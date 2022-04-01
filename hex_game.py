from copy import deepcopy
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class HexGame:
    """
    Hex game
    """

    def __init__(self, K, black_to_play=True) -> None:
        self.K = K  # Row and column length
        self.black_to_play_start = black_to_play  # Whether black (player 1) begins the game
        self.black_to_play = black_to_play  # Whose turn it is
        self.board = np.zeros((self.K, self.K, 2))

    def play(self, piece_in_cell_num):
        """
        Take a turn where you place a piece in a cell.
        
        Input:
        piece_in_cell: Number of the cell where the player wants to place their piece
        """
        cell_coordinates = self.num_to_cell(piece_in_cell_num)
        # Check if move is legal
        if self.check_legal_move(self.board, cell_coordinates):

            # Placing the players, whose turn it is, piece in the cell
            self.place_piece(cell_coordinates)

            # Next time other players turn
            self.black_to_play = not self.black_to_play
        else:
            raise Exception("Invalid move")

    def check_legal_move(self, board, cell):
        """
        Checks whether the players attempt to place a piece in a given cell is legal
        """
        # Checks whether there already is a piece in the cell
        return np.sum(board[tuple(cell)]) == 0

    def place_piece(self, cell):
        """
        Placing piece in a cell given the player whose turn it is
        """
        if self.black_to_play:
            self.board[tuple(cell)] = np.array([1, 0])
        else:
            self.board[tuple(cell)] = np.array([0, 1])

    def game_over(self, display_winner=False):
        """
        Checks whether the game is over (by cheking whether black or red has won)
        """
        black_player_won = self.check_player_won(0)
        if black_player_won:
            if display_winner:
                print("Black won")
            return True
        red_player_won = self.check_player_won(1)
        if red_player_won:
            if display_winner:
                print("Red won")
            return True
        return False

    def check_player_won(self, player):
        """
        Checking whether a certain player has won (0 is black, 1 is red)
        """
        # Copying the current board
        board = deepcopy(self.board)

        # If red player, transpose the board
        if player == 1:
            board = np.transpose(board, (1, 0, 2))

        discovered = set()

        # Nodes in the first column where a players piece is placed
        start_nodes = board[:, 0, player]
        start_idx = np.where(start_nodes == 1)[0]

        # Doing DFS from all start nodes
        for idx in start_idx:
            v = (idx, 0, player)

            # If start node not discovered
            if v not in discovered:

                # Performing DFS from start node v
                finished = self.dfs(discovered, v, player, board)

                # If we have found a path
                if finished:
                    return True
        return False

    def dfs(self, discovered, v, player, board):
        """
        v is the index of the place a piece can be placed on the board
        """

        # If we have reached the last column, we have found a path and the game is over
        if v[1] == self.K - 1:
            return True

        discovered.add(v)
        for neighbour in self.dfs_neighbours(v, player, board):

            if neighbour not in discovered:

                finished = self.dfs(discovered, neighbour, player, board)

                if finished:
                    return True

        return False

    def dfs_neighbours(self, v, player, board):
        """
        v is the index of the place a piece can be placed on the board
        """
        r = v[0]
        c = v[1]
        board_len = self.K - 1
        neighbours = set()

        # If v not in first row, there is a neighbour above
        if r > 0:
            neighbours.add((r - 1, c, player))
            # If v not in last column either, there is a neighbour obliquely up to the right
            if c < board_len:
                neighbours.add((r - 1, c + 1, player))

        # If v not in last column, there is a neighbour to the right
        if c < board_len:
            neighbours.add((r, c + 1, player))

        # If v not in last row, there is a neighbour below
        if r < board_len:
            neighbours.add((r + 1, c, player))

            # If v not in first column either, there is a neighbour obliquely down to the left
            if c > 0:
                neighbours.add((r + 1, c - 1, player))

        # If v not in first column, there is a neighbour to the left
        if c > 0:
            neighbours.add((r, c - 1, player))

        # Filter out neighbours that are not 1 on the board
        neighbours_one = set()
        for neighbour in neighbours:
            if board[neighbour] == 1:
                neighbours_one.add(neighbour)
        return neighbours_one

    def reset(self, random_start_player=False):
        """
        Resets the game, and (optionally) randomly chooses who begins.
        """
        self.board = np.zeros((self.K, self.K, 2))
        if random_start_player:
            self.black_to_play = np.random.choice([True, False])
        else:
            self.black_to_play = self.black_to_play_start

    def get_legal_actions(self, one_hot_state: tuple):
        """
        Getting the legal actions for a given player
        
        state: one-hot encoding of player turn and board state
        """
        state = self.rev_one_hot(np.array(list(one_hot_state)))
        board = state[1]
        legal = np.where((board[:, :, 0] == 0) & (board[:, :, 1] == 0))
        legal = np.array(legal).T
        legal_num = []
        for legal_cell in legal:
            legal_num.append(self.cell_to_num(legal_cell))
        return np.array(legal_num)

    def get_input_dim(self):
        """
        Returns the dimension of the one-hot encoded state vector
        """
        return (self.K * self.K * 2) + 2

    def get_output_dim(self):
        """
        Returns the dimension of the one-hot encoded action vector
        """
        return self.K * self.K

    def set_position(self, s_0_one_hot: tuple):
        """
        Setting the position/state of the board to a given position/state
        """
        s_0 = self.rev_one_hot(np.array(list(s_0_one_hot)))

        self.black_to_play = s_0[0]
        self.board = s_0[1]

    def get_position(self):
        """
        Returning the current position/state of the game
        """
        one_hot_board = self.one_hot_board()
        one_hot_player = self.one_hot_player()
        return tuple(np.concatenate((one_hot_player, one_hot_board)))

    def black_wins(self):
        """
        Returns 1 if black won the game (which would mean that red would play next),
        and -1 if red won
        """
        if self.check_player_won(0):
            return 1
        elif self.check_player_won(1):
            return -1
        else:
            raise Exception("No player has won")

    def one_hot_board(self):
        """
        One-hot encoding the board
        """
        return self.board.flatten()

    def one_hot_player(self):
        """
        One-hot encoding player whose turn it is
        """
        if self.black_to_play:
            return np.array([1, 0])
        return np.array([0, 1])

    def rev_one_hot(self, one_hot_encoding: np.array):
        """
        Reversing one-hot encoding of player turn and board state.
        One-hot encoding needs to be np.array
        """
        if one_hot_encoding[0] == 1:
            black_to_play = True
        else:
            black_to_play = False
        one_hot_board = one_hot_encoding[2:].reshape((self.K, self.K, 2))
        return (black_to_play, one_hot_board)

    def num_to_cell(self, number):
        """
        Converting cell number to cell coordinates (f.ex. in 3*3 game, num=5 would return [1, 2])
        """
        return np.array([number // self.K, number % self.K])

    def cell_to_num(self, cell):
        """
        Converting cell coordinates to cell number (f.ex. in 3*3 game, [1, 1] would return 4)
        """
        return (cell[0] * self.K) + cell[1]

    def get_id(self):
        """
        Returns an ID of the game
        """
        return "k" + str(self.K)

    def display_state(self, one_hot_state):
        """
        Uses graphics to display a state of a Hex game
        """
        state = self.rev_one_hot(np.array(list(one_hot_state)))
        black_to_play = state[0]
        board = state[1]
        player = "Black" if black_to_play else "Red"
        plt.title(f"{player} players turn to play")

        graph = nx.Graph()

        pos = {}
        edgelist = []
        # Iterating over rows
        for i in range(self.K):

            # Iterating over columns
            for j in range(self.K):
                node_idx = i * self.K + j

                # If current node is not on the last row, add edge to the node below
                if i < self.K - 1:
                    edgelist.append([node_idx, node_idx + self.K])

                    # If current node is neither on the first column, add edge to the node below to the left
                    if j % self.K != 0:
                        edgelist.append([node_idx, node_idx + self.K - 1])

                # If current node is not on the first column, add edge to the node to the left
                if j % self.K != 0:
                    edgelist.append([node_idx, node_idx - 1])

                # Calculating position/coordinates of nodes
                x = self.K * 2 - i + j
                y = self.K * 2 - i - j
                pos[node_idx] = (x, y)

        graph.add_edges_from(edgelist)

        color_list = []
        for node in graph.nodes():
            # Selecting colors indicating if there is a piece placed on the node/space
            row = node // self.K
            col = node % self.K
            if board[row, col, 0] == 1:
                color_list.append("black")
            elif board[row, col, 1] == 1:
                color_list.append("red")
            else:
                color_list.append("white")

        nx.draw(graph, pos, node_color=color_list, edgecolors="black")

        plt.savefig("game_state.png")


if __name__ == "__main__":
    h = HexGame(3)
    h.set_position(
        (1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0))
    h.display_state(
        (1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0))
