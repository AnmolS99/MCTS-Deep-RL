import random


class NimGame:
    """
    Simple Nim game siulation
    """

    def __init__(self, N: int, K: int, black_to_play=True) -> None:
        self.N = N
        self.K = K
        self.black_to_play = black_to_play

    def play(self, pieces_remove: int):
        """
        Take a turn where you remove between 1 and min(N, K) pieces
        """
        # Check if valid move
        if pieces_remove in self.get_legal_actions(self.get_current_state()):

            # Removing pieces from the board
            self.N -= pieces_remove

            # Next time other players turn
            self.black_to_play = not self.black_to_play
        else:
            raise Exception("Invalid number of pieces to remove")

    def game_over(self):
        """
        Checks if the current state is a final state
        """
        return self.N == 0

    def get_legal_actions(self, state: tuple):
        """
        Getting the legal actions for a given player
        
        state: (board_state, black_to_play)
        """
        board_state = state[0]
        return [i for i in range(1, min(board_state, self.K))]

    def get_current_state(self):
        """
        Getting the current state
        """
        return (self.N, self.black_to_play)

    def set_position(self, s_0):
        """
        Setting the position/state of the board to a given position/state
        """
        board_state = s_0[0]
        black_to_play = s_0[1]

        self.N = board_state
        self.black_to_play = black_to_play

    def get_position(self):
        """
        Returning the current position/state of the game
        """
        return self.N

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


def play_game():
    nsg = NimGame(20, 2)
    not_finished = True
    i = 0
    while not_finished:
        print(f"Player {i%2}'s turn")
        pieces_remove = random.randint(1, min(nsg.N, nsg.K))
        print(
            f"Remaining pieces: {nsg.N}, and player {i%2} wants to remove {pieces_remove} pieces"
        )
        nsg.take_turn(pieces_remove)
        if nsg.game_over():
            not_finished = False
            print(f"Player {i%2} wins")
        i += 1


if __name__ == "__main__":
    play_game()
