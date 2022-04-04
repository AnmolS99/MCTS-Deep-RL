import tensorflow as tf
import numpy as np
from hex_game import HexGame


class HexActor():

    def __init__(self, k, model_path) -> None:
        self.k = k
        self.game_board = HexGame(K=self.k)
        self.model = tf.keras.models.load_model(model_path)

    def get_action(self, state):
        # Preprocessing state
        state_list = state
        player = self.parse_player(state_list[0])
        board_state = self.parse_board_state(state_list[1:])
        parsed_state = np.concatenate((player, board_state)).reshape(1, -1)
        parsed_state_tuple = tuple(parsed_state[0])

        # Sending it through model
        probs = self.model(parsed_state).numpy()

        # Getting indices of all legal actions
        legal_actions_idx = self.game_board.get_legal_actions(
            parsed_state_tuple)

        # Getting the probabilities and normalizing over them
        legal_actions_probs = normalize_vector(probs[:, legal_actions_idx])

        # Creating a new array of zeroes
        legal_actions = np.zeros_like(probs)

        legal_actions[:, legal_actions_idx] = legal_actions_probs
        a = np.argmax(legal_actions)

        # Postprocess action

        row = a // self.k
        col = a % self.k

        return int(row), int(col)

    def parse_player(self, number):
        # Player 1 is RED and supposed to go from top-right to bottom-left
        if number == 1:
            return [0, 1]
        # Player 2 is BLACK and supposed to go from top-left to bottom-right
        elif number == 2:
            return [1, 0]
        else:
            raise Exception("Invalid player number")

    def parse_board_state(self, board_state):
        parsed_board = []
        for cell in board_state:
            # If empty cell
            if cell == 0:
                parsed_board.append(0)
                parsed_board.append(0)
            # If player 1 (RED) has placed in cell, we append [0, 1]
            elif cell == 1:
                parsed_board.append(0)
                parsed_board.append(1)
            # If player 2 (BLACK) has placed in cell, we append [1, 0]
            elif cell == 2:
                parsed_board.append(1)
                parsed_board.append(0)
            else:
                raise Exception("Invalid cell number")
        return parsed_board


def normalize_vector(vector):
    """
    Normalizes a vector (np.array)
    """
    return vector / np.sum(vector)