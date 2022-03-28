import tensorflow as tf
import numpy as np


class TOPP:
    """
    Tournament Of Progressive Policies
    """

    def __init__(self, game, g, k, num_actual_games, checkpoints) -> None:
        self.game = game
        self.g = g
        self.k = k
        self.num_actual_games = num_actual_games
        self.checkpoints = checkpoints

    def load_models(self):
        """
        Loading in all the models/ANets
        """
        models = []
        for i in range(self.checkpoints):
            checkpoint = i * (self.num_actual_games // (self.checkpoints - 1))
            model = tf.keras.models.load_model(
                f"models/model_k{self.k}_{checkpoint}_of_{self.num_actual_games}"
            )
            models.append(model)

    def play(self, model1, model2):
        """
        Two models playing G games against each other
        """
        wins_model_1 = 0
        for i in range(self.g):

            # Resetting the board and randomly choosing starting player
            self.game.reset(random_start_player=True)

            # Getting start position
            state = self.game.get_position()

            black_player_turn = self.game.black_to_play

            while not self.game.game_over():
                if black_player_turn:
                    distribution = model1(state)
                else:
                    distribution = model2(state)

                a = np.argmax(distribution)

                self.game.play(a)

                black_player_turn = not black_player_turn

            if self.game.black_won:
                wins_model_1 += 1

        return wins_model_1