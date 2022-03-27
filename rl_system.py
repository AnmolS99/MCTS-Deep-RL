import copy
import numpy as np
import tensorflow as tf
from anet import ANet
from hex_game import HexGame
from mcts import MCTS
from nim_game import NimGame


class RLSystem:
    """
    Reinforcement system that play multiple actual games, 
    where one move in an actual game is based on multiple MCTS simulations
    """

    def __init__(self, game, anet_lr, num_search_games, c, eps,
                 num_actual_games, checkpoints) -> None:
        self.game = game
        self.anet_lr = anet_lr
        self.anet = self.create_anet()
        self.rbuf = []
        self.mcts = MCTS(copy.deepcopy(game), self.anet, num_search_games, c,
                         eps)
        self.num_actual_games = num_actual_games
        self.checkpoints = checkpoints

    def create_anet(self):
        """
        Creates anet based on the game that is being played
        """
        input_nodes = self.game.get_input_dim()
        output_nodes = self.game.get_output_dim()
        nn_specs = (input_nodes, 100, 100, 100, 100, output_nodes)
        anet = ANet(nn_specs, self.anet_lr)
        return anet

    def rl_algorithm(self, show_game=False):
        # 1. Need to save anet params
        i_s = self.checkpoints

        # 2. Clearing RBUF
        self.rbuf = []

        # 3. Randomly initialize params for anet

        # 4. Iterate over number of actual games
        for g_a in range(self.num_actual_games):

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

            # e) Train ANet on a random minibatch of cases from RBUF
            states = [t[0] for t in self.rbuf]
            distibutions = [t[1] for t in self.rbuf]
            self.anet.nn.fit(np.array(states), np.array(distibutions))

            # f) If g_a is a checkpoint, save the parameters
            if g_a % (self.num_actual_games // self.checkpoints) == 0:
                self.anet.nn.save(f"models/model_after_{g_a}")
                print("Checkpoint!")


if __name__ == "__main__":
    k = 3
    hex = HexGame(k)
    rls = RLSystem(hex,
                   anet_lr=0.03,
                   num_search_games=100,
                   c=1,
                   eps=0.1,
                   num_actual_games=10,
                   checkpoints=2)
    rls.rl_algorithm(show_game=False)
    print(
        f"State: Black player turn - {rls.anet.nn(np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(1, -1))}"
    )
    print(
        f"State: Black player turn - {rls.anet.nn(np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1))}"
    )