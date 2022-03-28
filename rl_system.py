import copy
import numpy as np
from mcts import MCTS


class RLSystem:
    """
    Reinforcement system that play multiple actual games, 
    where one move in an actual game is based on multiple MCTS simulations
    """

    def __init__(self, game, anet, num_search_games, c, eps, num_actual_games,
                 checkpoints) -> None:
        self.game = game
        self.anet = anet
        self.rbuf = []
        self.mcts = MCTS(copy.deepcopy(game), self.anet, num_search_games, c,
                         eps)
        self.num_actual_games = num_actual_games
        self.checkpoints = checkpoints

    def rl_algorithm(self, show_game=False):
        # 1. Need to save anet params
        i_s = self.checkpoints

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
            root = s_init

            # Showing the initial game state
            if show_game:
                self.game.display_state(root)

            # d) While B_a not in a final state
            while not self.game.game_over():

                # Initialize Monte Carlo board (B_mc) to same state as root, and play number_search_games
                distribution = self.mcts.uct_search(root)

                # Adding case (root, D) to RBUF
                self.rbuf.append((root, distribution))

                # Chooses actual move based on distribution
                a_star = np.random.choice(range(len(distribution)),
                                          p=distribution)

                # Performing a_star to produce s_star
                self.game.play(a_star)
                s_star = self.game.get_position()

                root = s_star

                # Showing game state
                if show_game:
                    self.game.display_state(root)

            print(f"g_a: {g_a} | RBUF length: {len(self.rbuf)}")

            # e) Train ANet on a random minibatch of cases from RBUF
            states = [t[0] for t in self.rbuf]
            distibutions = [t[1] for t in self.rbuf]
            self.anet.nn.fit(np.array(states), np.array(distibutions))

            # f) If g_a is a checkpoint, save the parameters
            if g_a % (self.num_actual_games // (self.checkpoints - 1)) == 0:
                self.anet.nn.save(
                    f"models/model_k4_{g_a}_of_{self.num_actual_games}")
                print("Checkpoint!")
