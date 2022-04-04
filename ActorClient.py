import argparse
import contextlib
import functools
import getpass
import json
import logging
import os
import pprint
import random
import socket
import ssl
import struct
import sys
import warnings

THIS_DIR = os.path.abspath(os.path.dirname(__file__))

DEFAULT_HOST = os.getenv('IT3105_HOST', 'it3105.idi.ntnu.no')
DEFAULT_PORT = os.getenv('IT3105_PORT', '33000')
DEFAULT_CERT = os.getenv('IT3105_CERT', os.path.join(THIS_DIR, 'server.crt'))
DEFAULT_AUTH = os.getenv('IT3105_AUTH=aa5c0269e3254d9a95675779c36852ef')
DEFAULT_ECHO = os.getenv('IT3105_ECHO', '0').lower() in {'yes', '1', 'y'}
DEFAULT_QUALIFY = os.getenv('IT3105_QUALIFY')
DEFAULT_API_PORT = os.getenv('IT3105_API_PORT', '32000')
DEFAULT_LeAGUE_PORT = os.getenv('IT3105_LeAGUE_PORT', '30000')


class ActorClient:
    """Client for connecting to the gaming server (and parts of the API)

    Args:
        host (str): server hostname or ip address
        port (int): server port number for gaming (str is fine too)
        cert (str): path to server certificate (for tls)
        auth (str): your personal authentication token
        echo (bool): toggle debug logging (strings like "yes" is fine too)
        qualify (bool): automatically answer "y" when asked if you want
            to qualify (the alternative "n" will only play test games)
        api_port (int): server port number for api (str is fine too)
        api_port (int): server port number for the league
        log_fmt (str): optional format for the logger
    """

    class Error(Exception):
        """Error thrown (primarily) when the server is not happy with you"""
        pass

    def __init__(self,
                 host=DEFAULT_HOST,
                 port=DEFAULT_PORT,
                 cert=DEFAULT_CERT,
                 auth=DEFAULT_AUTH,
                 echo=DEFAULT_ECHO,
                 qualify=DEFAULT_QUALIFY,
                 api_port=DEFAULT_API_PORT,
                 league_port=DEFAULT_LeAGUE_PORT,
                 log_fmt=None):
        self.host = host
        self.port = port
        self.cert = cert
        self.auth = auth
        self.echo = echo
        self.api_port = api_port
        self.league_port = league_port
        self.log_fmt = log_fmt
        self.answers = {'qualify': qualify}
        self.logger = self.create_logger()
        self._sock = None

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   "Game loop"
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    @property
    def sock(self):
        """Returns the socket used for server communication (if connected)"""
        if self._sock is None:
            raise RuntimeError('Cannot access socket without being connected')
        return self._sock

    def run(self, mode='qualifiers'):
        """Main loop for the student client

        It begins by establishing an SSL connection with the server
        and then reacts to incoming messages.

        Args:
            mode (str): Which mode to connect to. 'qualifiers' plays against
                opponents hosted on the server, while 'league' will queue
                you up to play against other players.
        """
        try:
            port = {'qualifiers': self.port, 'league': self.league_port}[mode]
        except KeyError:
            raise ValueError('Illegal mode "{mode}"'.format(mode=mode))
        # Conenct to server
        with self.connect(host=self.host, port=port):
            # Receive and react to messages from server
            while self.handle(self.recv()):
                pass

    def handle(self, msg):
        """Handles an incoming json message

        There are a number of different messages that the client may react to,
        each identified by the "topic" field of the json payload.

        Topics:
            - message: generic info/warning/error log message
            - authentication: the server requests a token
            - question: the server asks a yes/no question
            - error: the server signals that the session is done (with error)
            - finish: the server signals that the session is done (without error)
            - series_start: beginning of a new set of games (player hook)
            - game_start: beginning of a new game (player hook)
            - request_action: the server requests a player action (player hook)
            - game_end: end of a game (player hook)
            - series_end: end of a set of games (player hook)
            - tournament_end: end of all games (player hook)

        Args:
            msg (dict): a dict containing a json message from the server.
                It should always have a "topic" string field, as well as
                a number of other fields depending on the topic.

        Returns:
            bool: indicating whether the connection should be closed.
        """
        topic = msg.get('topic')

        # === Logistics

        if topic == 'message':
            self.handle_message(message=msg['message'], level=msg['level'])

        elif topic == 'authentication':
            if self.auth is None:
                self.send(getpass.getpass('Enter API token: '))
            else:
                self.send(self.auth)
            self.logger.info('Authenticating...')

        elif topic == 'question':
            answer = self.handle_question(question=msg['question'],
                                          prompt=msg['prompt'])
            self.send(answer)

        elif topic == 'finish':
            self.logger.info('Connection closed')
            return False

        elif topic == 'error':
            self.handle_error(error=msg['error'])
            return False

        # === Hooks for player logic

        elif topic == 'series_start':
            unique_player_id = msg['unique_player_id']
            player_id_map = msg['player_id_map']
            series_player_id = [
                p[1] for p in player_id_map if p[0] == unique_player_id
            ]
            self.handle_series_start(
                unique_id=unique_player_id,
                series_id=series_player_id[0],
                player_map=player_id_map,
                num_games=msg['num_games'],
                game_params=msg['game_params'],
            )

        elif topic == 'game_start':
            self.handle_game_start(start_player=msg['start_player'], )

        elif topic == 'request_action':
            action = self.handle_get_action(state=msg['state'])
            self.send(action)

        elif topic == 'game_over':
            self.handle_game_over(
                winner=msg['winner'],
                end_state=msg['end_state'],
            )

        elif topic == 'series_over':
            self.handle_series_over(stats=msg['stats'])

        elif topic == 'tournament_over':
            self.handle_tournament_over(score=msg['score'])

        # === Wooups?

        else:
            raise RuntimeError(
                'Received unexpected message from server "{msg}"'.format(
                    msg=msg))

        return True

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Utilities
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def create_logger(self):
        """Creates a simple stream logger"""
        logger = logging.Logger('ActorClient')
        if str(self.echo).lower() in {'yes', '1', 'y', 'true', 't'}:
            level = logging.DEBUG
        else:
            level = logging.INFO
        logger.setLevel(level)
        log_fmt = self.log_fmt or '[%(asctime)s] %(levelname)s: %(message)s'
        fmt = logging.Formatter(log_fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        logger.addHandler(sh)
        return logger

    def create_socket(self):
        """Creates an SSL-wrapped ip socket"""
        raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(self.cert)
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = False  # We have no hostname for the server
        return context.wrap_socket(raw_sock)

    @contextlib.contextmanager
    def connect(self, host, port):
        """
        Context for opening a connection to the gaming server

        Args:
            host (str): server hostname
            port (str): server port (int is fine too)
        """
        if self._sock is not None:
            self.logger.warning('Overwriting existing socket')
        sock = self.create_socket()
        try:
            self._sock = sock
            self.logger.info('Connecting to %s:%s', host, port)
            sock.connect((host, int(port)))
            self.logger.debug(pprint.pformat(sock.getpeercert(), indent=4))
            self.logger.debug(sock.cipher())
            yield sock
        finally:
            self._sock = None
            sock.close()

    def recv(self):
        """Receives json data from the server"""
        # Receive header (an int specifying the size of the body)
        header_num = 0
        header = []
        i_size = struct.calcsize('i')
        while header_num < i_size:
            recv = self.sock.recv(i_size - header_num)
            if not recv: raise socket.error('Received empty string')
            header_num += len(recv)
            header.append(recv)
        data_size = struct.unpack('i', b''.join(header))[0]
        self.logger.debug('Receiving payload of size: %s', data_size)
        # Receive body
        body_num = 0
        body = []
        while body_num < data_size:
            recv = self.sock.recv(data_size - body_num)
            if not recv: raise socket.error('Received empty string')
            body_num += len(recv)
            body.append(recv)
        msg = b''.join(body)
        self.logger.debug('Received payload: %s', msg)
        # Decode byte string and parse json
        return json.loads(msg.decode('utf-8'))

    def send(self, msg):
        """Sends json data to the server"""
        # Dump json and encode to byte string
        self.logger.debug('Sending: %s', msg)
        byte_data = json.dumps(msg).encode('utf-8')
        # Send the size of the payload
        data_size = len(byte_data)
        self.sock.send(struct.pack('i', data_size))
        # And the the actual payload
        self.sock.send(byte_data)

    def get_random_action(self, state):
        """Utility for picking random Hex actions

        Args:
            state (list): current board configuration (see `handle_get_action`)

        Returns:
            tuple: random valid board coordinates as (row, col) ints

        Note:
            You can do better!
        """
        board_size = int((len(state) - 1)**0.5)
        return random.choice([(index // board_size, index % board_size)
                              for index, cell in enumerate(state[1:])
                              if cell == 0])

    def handle_question(self, question, prompt):
        """Hook for answering y/n questions from the server

        It will look through the "answers" member of this instance
        to see if there is a "canned" answer first. If not, the user
        will be prompted for a answer via standard in.

        Args:
            question (str): string identifying the question
            prompt (str): a more verbose question prompt that will be
                presented to the user if there is no canned answer.

        Returns:
            bool: a parsed version of the provided answer
        """
        if self.answers.get(question) is not None:
            answer = self.answers[question]
            self.logger.info('Using default answer: %s="%s"', question, answer)
        else:
            answer = input('{prompt} [y/n] '.format(prompt=prompt))
        answer = str(answer).lower().strip()
        if answer in {'yes', 'y', '1', 'true', 't'}:
            return True
        elif answer in {'no', 'n', '0', 'false', 'f'}:
            return False
        else:
            raise self.Error(
                'Bad answer ({question}="{answer}"): Not y/n'.format(
                    question=question, answer=answer))

    def handle_message(self, message, level):
        """Hook for handling incoming log messages from the server

        Args:
            message (str): Message body
            level (str): Logging level
        """
        self.logger.log(
            {
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
            }.get(level, logging.INFO),
            message,
        )

    def handle_error(self, error):
        """Hook for handling incoming server errors

        Whenever the server is angry (or crashes), it will attempt to
        send a final error message before closing the connection.

        Args:
            error (str): error message string
        """
        self.logger.error(error)
        raise self.Error(error)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   Game Interface
    #   TODO: override
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def handle_series_start(self, unique_id, series_id, player_map, num_games,
                            game_params):
        """Called at the start of each set of games against an opponent

        Args:
            unique_id (int): your unique id within the tournament
            series_id (int): whether you are player 1 or player 2
            player_map (list): (inique_id, series_id) touples for both players
            num_games (int): number of games that will be played
            game_params (list): game-specific parameters.

        Note:
            > For the qualifiers, your player_id should always be "-200",
              but this can change later
            > For Hex, game params will be a 1-length list containing
              the size of the game board ([board_size])
        """
        self.logger.info(
            'Series start: unique_id=%s series_id=%s player_map=%s num_games=%s'
            ', game_params=%s',
            unique_id,
            series_id,
            player_map,
            num_games,
            game_params,
        )

    def handle_game_start(self, start_player):
        """Called at the beginning of of each game

        Args:
            start_player (int): the series_id of the starting player (1 or 2)
        """
        self.logger.info('Game start: start_player=%s', start_player)

    def handle_get_action(self, state):
        """Called whenever it's your turn to pick an action

        Args:
            state (list): board configuration as a list of board_size^2 + 1 ints

        Returns:
            tuple: action with board coordinates (row, col) (a list is ok too)

        Note:
            > Given the following state for a 5x5 Hex game
                state = [
                    1,              # Current player (you) is 1
                    0, 0, 0, 0, 0,  # First row
                    0, 2, 1, 0, 0,  # Second row
                    0, 0, 1, 0, 0,  # ...
                    2, 0, 0, 0, 0,
                    0, 0, 0, 0, 0
                ]
            > Player 1 goes "top-down" and player 2 goes "left-right"
            > Returning (3, 2) would put a "1" at the free (0) position
              below the two vertically aligned ones.
            > The neighborhood around a cell is connected like
                  |/
                --0--
                 /|
        """
        self.logger.info('Get action: state=%s', state)
        row, col = self.get_random_action(state)
        self.logger.info('Picked random: row=%s col=%s', row, col)
        return row, col

    def handle_game_over(self, winner, end_state):
        """Called after each game

        Args:
            winner (int): the winning player (1 or 2)
            end_stats (tuple): final board configuration

        Note:
            > Given the following end state for a 5x5 Hex game
            state = [
                2,              # Current player is 2 (doesn't matter)
                0, 2, 0, 1, 2,  # First row
                0, 2, 1, 0, 0,  # Second row
                0, 0, 1, 0, 0,  # ...
                2, 2, 1, 0, 0,
                0, 1, 0, 0, 0
            ]
            > Player 1 has won here since there is a continuous
              path of ones from the top to the bottom following the
              neighborhood description given in `handle_get_action`
        """
        self.logger.info('Game over: winner=%s end_state=%s', winner,
                         end_state)

    def handle_series_over(self, stats):
        """Called after each set of games against an opponent is finished

        Args:
            stats (list): a list of lists with stats for the series players

        Example stats (suppose you have ID=-200, and playing against ID=999):
            [
                [-200, 1, 7, 3],  # id=-200 is player 1 with 7 wins and 3 losses
                [ 999, 2, 3, 7],  # id=+999 is player 2 with 3 wins and 7 losses
            ]
        """
        self.logger.info('Series over: stats=%s', stats)

    def handle_tournament_over(self, score):
        """Called after all series have finished

        Args:
            score (float): Your score (your win %) for the tournament
        """
        self.logger.info('Tournament over: score=%s', score)

    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #   API methods
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    def call_api(self, method, endpoint, data=None):
        """Calls and endpoint on the it3105 API service

        Args:
            method (str): which http method to user
            endpoint (str): url suffix / path
            data (json): request data

        Returns:
            json: server response
        """
        import requests
        req = {'get': requests.get, 'put': requests.put}[method]
        url = 'https://%s:%s/api/%s' % (self.host, self.api_port, endpoint)
        headers = {'X-Api-Token': self.auth}
        self.logger.debug('Calling %s : %s with data=%s', method, url, data)
        resp = req(url, headers=headers, verify=self.cert, json=data)
        if not resp.status_code == 200:
            try:
                e = resp.json()
                s = '%s (%s): %s' % (e['code'], e['name'], e['data'])
            except:
                s = 'Bad status code: %s' % resp.status_code
            self.logger.error(s)
            raise self.Error(s)
        return resp.json()

    def request_reset_token(self, email=None):
        """Creates a request for API token reset

        Args:
            email (str): email of account to reset token for

        Returns:
            json: server response

        """
        if email is None:
            email = input('Please enter your (NTNU) email: ')
        resp = self.call_api('put', 'reset_token', {'email': email})
        self.logger.info(resp['message'])
        return resp

    def reset_token(self, temp_token=None):
        """Activates an API token reset request

        Args:
            temp_token (str): temporary token received through email

        Returns:
            json: server response with new token
        """
        if temp_token is None:
            temp_token = input('Paste temporary token from email: ')
        resp = self.call_api('get', 'reset_token/%s' % temp_token)
        self.logger.info(resp['message'])
        self.logger.info('New API Token: %s', resp['token'])
        self.logger.info('Save this somewhere for later use')
        return resp

    def get_profile(self):
        """Queries information about the current user (as given by API token)

        Returns:
            json: server response with information about your user
        """
        resp = self.call_api('get', 'profile')
        self.logger.info(json.dumps(resp, indent=4))
        return resp

    def update_handle(self, handle=None):
        """Updates the player handle (nickname)

        Args:
            handle (str): New nickname. Empty string for random

        Returns:
            json: server response
        """
        if handle is None:
            handle = input('Enter new handle (leave blank for random): ')
        resp = self.call_api('put', 'handle', {'handle': handle})
        self.logger.info(resp['message'])
        return resp


if __name__ == '__main__':

    parser = argparse.ArgumentParser('API to it3105 gaming service')
    parser.add_argument('--host',
                        help='Server host hostess',
                        default=DEFAULT_HOST)
    parser.add_argument('--api-port',
                        help='Server API host port',
                        default=DEFAULT_API_PORT)
    parser.add_argument('--cert',
                        help='Server ssl certificate',
                        default=DEFAULT_CERT)
    parser.add_argument('--echo',
                        help='Log debug',
                        action='store_true',
                        default=DEFAULT_ECHO)

    subparsers = parser.add_subparsers(help='Commands')
    subparsers.required = True

    parser_rrt = subparsers.add_parser('request-reset-token',
                                       help='Request token reset email')
    parser_rrt.set_defaults(func='request_reset_token')
    parser_rrt.add_argument(
        '--email', help='Your email in the gaming system (NTNU probably)')

    parser_rt = subparsers.add_parser(
        'reset-token', help='Reset API token with temp token from email')
    parser_rt.set_defaults(func='reset_token')
    parser_rt.add_argument('--temp-token',
                           help='Temporary token received from email')

    parser_gp = subparsers.add_parser(
        'get-profile', help='Get information about you (requires API token)')
    parser_gp.set_defaults(func='get_profile')
    parser_gp.add_argument('--auth',
                           help='API authentication token',
                           default=DEFAULT_AUTH)

    parser_uh = subparsers.add_parser(
        'update-handle', help='Update your player handle (requires API token)')
    parser_uh.set_defaults(func='update_handle')
    parser_uh.add_argument('--auth',
                           help='API authentication token',
                           default=DEFAULT_AUTH)
    parser_uh.add_argument('--handle', help='New handle ("" for random)')

    args = vars(parser.parse_args())
    client = ActorClient(
        **{
            key: args.pop(key, None)
            for key in ['host', 'api_port', 'cert', 'auth', 'echo']
        })
    try:
        getattr(client, args.pop('func'))(**args)
        sys.exit(0)
    except ActorClient.Error:
        sys.exit(1)
