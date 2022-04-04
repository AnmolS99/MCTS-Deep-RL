import copy
import numpy as np
from mcts import MCTS
from lite_model import LiteModel
import random


class RLSystem:
    """
    Reinforcement system that play multiple actual games, 
    where one move in an actual game is based on multiple MCTS simulations
    """

    def __init__(self, game, anet, num_search_games, c, eps_rl, eps_rl_delta,
                 eps_mcts, num_actual_games, checkpoints, rbuf_size,
                 epochs) -> None:
        self.game = game
        self.anet = anet
        self.rbuf = []
        self.eps = eps_rl
        self.eps_delta = eps_rl_delta
        self.mcts = MCTS(copy.deepcopy(game),
                         LiteModel.from_keras_model(self.anet.nn),
                         num_search_games, c, eps_mcts)
        self.num_actual_games = num_actual_games
        self.checkpoints = checkpoints
        self.rbuf_size = rbuf_size
        self.epochs = epochs

    def rl_algorithm(self, show_game=False):
        # 1. Need to save anet params (for untrained network)
        self.anet.nn.save(
            f"models/model_{self.game.get_id()}_{0}_of_{self.num_actual_games}"
        )
        print("Checkpoint!")

        # Setting lmodel
        lmodel = LiteModel.from_keras_model(self.anet.nn)

        # 2. Clearing RBUF
        self.rbuf = []

        # 3. Randomly initialize params for anet

        # 4. Iterate over number of actual games
        for g_a in range(self.num_actual_games + 1):

            # a) Initialize the actual game board (B_a) to an empty board
            self.game.reset(random_start_player=False)

            # b) Get starting board state
            s_init = self.game.get_position()

            # c) Initialize MCTS to root s_init
            self.mcts.reset()
            root = s_init

            # Showing the initial game state
            if show_game:
                self.game.display_state(root)

            # Decreasing epsilon
            #self.eps = self.eps * self.eps_delta

            # d) While B_a not in a final state
            while not self.game.game_over(display_winner=True):

                # Initialize Monte Carlo board (B_mc) to same state as root, and play number_search_games
                distribution = self.mcts.uct_search(root, lmodel)

                # Adding case (root, D) to RBUF
                self.rbuf.append((root, distribution))

                # Chooses actual move, either randomly or the one with the highest probability
                if random.random() < self.eps:
                    distribution = distribution.reshape(1, -1)
                    non_zero_idx = np.nonzero(distribution)[1]
                    a_star = np.random.choice(non_zero_idx)
                else:
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
            if len(self.rbuf) <= self.rbuf_size:
                states = [t[0] for t in self.rbuf]
                distibutions = [t[1] for t in self.rbuf]
            else:
                rbuf_samples_idx = np.random.choice(a=len(self.rbuf),
                                                    size=self.rbuf_size,
                                                    replace=False)
                rbuf_samples = np.array(self.rbuf,
                                        dtype=object)[rbuf_samples_idx]
                states = [t[0] for t in rbuf_samples]
                distibutions = [t[1] for t in rbuf_samples]

            self.anet.nn.fit(np.array(states),
                             np.array(distibutions),
                             epochs=self.epochs)

            # Creating lmodel
            lmodel = LiteModel.from_keras_model(self.anet.nn)

            # f) If g_a is a checkpoint, save the parameters
            if g_a % (self.num_actual_games // (self.checkpoints - 1)) == 0:
                self.anet.nn.save(
                    f"models/model_{self.game.get_id()}_{g_a}_of_{self.num_actual_games}"
                )
                print("Checkpoint!")
