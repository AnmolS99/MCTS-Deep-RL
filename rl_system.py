import copy
import random
import numpy as np
from mcts import MCTS


class RLSystem:
    """
    Reinforcement system that play multiple actual games, 
    where one move in an actual game is based on multiple MCTS simulations
    """

    def __init__(self, game, anet, num_search_games, c, eps, eps_delta,
                 num_actual_games, checkpoints) -> None:
        self.game = game
        self.anet = anet
        self.rbuf = []
        self.mcts = MCTS(copy.deepcopy(game), self.anet, num_search_games, c,
                         eps, eps_delta)
        self.num_actual_games = num_actual_games
        self.checkpoints = checkpoints

    def rl_algorithm(self, show_game=False):
        # 1. Need to save anet params (for untrained network)
        self.anet.nn.save(
            f"models/model_{self.game.get_id()}_{0}_of_{self.num_actual_games}"
        )
        print("Checkpoint!")

        # 2. Clearing RBUF
        self.rbuf = []

        # 3. Randomly initialize params for anet

        # 4. Iterate over number of actual games
        for g_a in range(self.num_actual_games + 1):

            # a) Initialize the actual game board (B_a) to an empty board
            self.game.reset(random_start_player=True)

            # b) Get starting board state
            s_init = self.game.get_position()

            # c) Initialize MCTS to root s_init
            self.mcts.reset()
            root = s_init

            # Showing the initial game state
            if show_game:
                self.game.display_state(root)

            # d) While B_a not in a final state
            while not self.game.game_over(display_winner=True):

                # Initialize Monte Carlo board (B_mc) to same state as root, and play number_search_games
                distribution = self.mcts.uct_search(root)

                # Adding case (root, D) to RBUF
                self.rbuf.append((root, distribution))

                # Chooses actual move based on distribution
                a_star = np.argmax(distribution)

                # Performing a_star to produce s_star
                self.game.play(a_star)
                s_star = self.game.get_position()

                root = s_star

                # Showing game state
                if show_game:
                    self.game.display_state(root)

            print(f"g_a: {g_a} | RBUF length: {len(self.rbuf)}")

            # e) Train ANet on a random minibatch of cases from RBUF
            if len(self.rbuf) <= 1024:
                states = [t[0] for t in self.rbuf]
                distibutions = [t[1] for t in self.rbuf]
            else:
                rbuf_samples = random.choices(self.rbuf, k=1024)
                states = [t[0] for t in rbuf_samples]
                distibutions = [t[1] for t in rbuf_samples]
            self.anet.nn.fit(np.array(states),
                             np.array(distibutions),
                             epochs=64)

            # f) If g_a is a checkpoint, save the parameters
            if g_a % (self.num_actual_games // (self.checkpoints - 1)) == 0:
                self.anet.nn.save(
                    f"models/model_{self.game.get_id()}_{g_a}_of_{self.num_actual_games}"
                )
                print("Checkpoint!")
