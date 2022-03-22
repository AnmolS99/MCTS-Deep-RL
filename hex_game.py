import numpy as np


class HexGame:
    """
    Hex game
    """

    def __init__(self, K, black_to_play=True) -> None:
        self.K = K  # Row and column length
        self.black_to_play_start = black_to_play  # Whether black begins the game
        self.black_to_play = black_to_play  # Whose turn it is
        self.board = np.zeros((self.K, self.K, 2))

    def play(self, stone_in_cell):
        """
        Take a turn where you place a piece in a cell.
        
        Input:
        stone_in_cell: Coordinates of the cell where the player wants to place their piece
        """
        if self.check_legal_move(self.board, stone_in_cell):
            pass

    def check_legal_move(self, board, cell):
        """
        Checks whether the players attempt to place a piece in a given cell is legal
        """
        return np.sum(board[tuple(cell)]) == 0


if __name__ == "__main__":
    h = HexGame(3)
    h.board[0, 0, 0] = 1
    h.board[0, 0, 1] = 1
    print(h.board)
    print(h.check_legal_move(h.board, np.array([0, 0])))
