import random
import numpy as np


class NimGame:
    """
    Simple Nim game simulation
    """

    def __init__(self, N: int, K: int, black_to_play=True) -> None:
        self.N_start = N  # The number of pieces at the start of the game
        self.N = N  # The number of pieces currently in the game
        self.K = K  # The maximum amount of pieces allowed to remove
        self.black_to_play_start = black_to_play  # Whether it is black players turn to play the first time the game is initialized
        self.black_to_play = black_to_play  # Whether it is black players turn to play

    def play(self, pieces_remove: int):
        """
        Take a turn where you remove between 1 and min(N, K) pieces
        """
        # Check if valid move
        if pieces_remove in self.get_legal_actions(self.get_position()):

            # Removing pieces from the board
            self.N -= pieces_remove + 1

            # Next time other players turn
            self.black_to_play = not self.black_to_play
        else:
            raise Exception("Invalid number of pieces to remove")

    def game_over(self):
        """
        Checks if the current state is a final state
        """
        return self.N == 0

    def reset(self, random_start_player=False):
        """
        Resets the game, and (optionally) randomly chooses who begins.
        """
        self.N = self.N_start
        if random_start_player:
            self.black_to_play = random.choice([True, False])
        else:
            self.black_to_play = self.black_to_play_start

    def get_legal_actions(self, one_hot_state: tuple):
        """
        Getting the legal actions for a given player
        
        state: one-hot encoding of player turn and board state
        """
        state = self.rev_one_hot(np.array(list(one_hot_state)))
        n = state[1]
        return np.array([i for i in range(0, min(n, self.K))])

    def get_input_dim(self):
        """
        Returns the dimension of the one-hot encoded state vector
        """
        return self.N_start + 2

    def get_output_dim(self):
        """
        Returns the dimension of the one-hot encoded action vector
        """
        return self.K

    def set_position(self, s_0_one_hot: tuple):
        """
        Setting the position/state of the board to a given position/state
        """
        s_0 = self.rev_one_hot(np.array(list(s_0_one_hot)))

        self.black_to_play = s_0[0]
        self.N = s_0[1]

    def get_position(self):
        """
        Returning the current position/state of the game
        """
        one_hot_number = self.one_hot_number(self.N)
        one_hot_player = self.one_hot_player()
        return tuple(np.concatenate((one_hot_player, one_hot_number)))

    def black_wins(self):
        """
        Returns 1 if black won the game (which would mean that white would play next),
        and -1 if white won
        """
        if self.game_over():
            if (not self.black_to_play):
                return 1
            else:
                return -1
        else:
            raise Exception("Game is not over, cannot determine who won")

    def one_hot_number(self, number):
        """
        One-hot encoding a number
        """
        one_hot = np.zeros(self.N_start + 1)
        one_hot[number] = 1
        return one_hot

    def one_hot_player(self):
        """
        One-hot encoding player whose turn it is
        """
        if self.black_to_play:
            return np.array([0])
        return np.array([1])

    def rev_one_hot(self, one_hot_encoding):
        """
        Reversing one-hot encoding of player turn and board state
        """
        if one_hot_encoding[0] == 0:
            black_to_play = True
        else:
            black_to_play = False
        one_hot_board = one_hot_encoding[1:]
        n = np.where(one_hot_board == 1)[0][0]
        return (black_to_play, n)
