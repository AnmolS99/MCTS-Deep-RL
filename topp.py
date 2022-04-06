import time
import tensorflow as tf
import numpy as np
from hex_game import HexGame
import matplotlib.pyplot as plt

from lite_model import LiteModel


class TOPP:
    """
    Tournament Of Progressive Policies
    """

    def __init__(self, g, k, num_actual_games, checkpoints) -> None:
        self.game = HexGame(K=k)
        self.g = g
        self.k = k
        self.num_actual_games = num_actual_games
        self.checkpoints = checkpoints

    def load_models(self):
        """
        Loading in all the models/ANets
        """
        models = dict()
        for i in range(self.checkpoints):
            checkpoint = i * (self.num_actual_games // (self.checkpoints - 1))
            model = tf.keras.models.load_model(
                f"models/model_k{self.k}_{checkpoint}_of_{self.num_actual_games}"
            )
            # Creating Litemodel
            lmodel = LiteModel.from_keras_model(model)

            # Saving the lmodel to dictionary
            models[f"player{checkpoint}"] = lmodel
        return models

    def play(self,
             model1,
             model2,
             g,
             model1_name="Model1",
             model2_name="Model2",
             display=False,
             delay=1):
        """
        Two models playing G games against each other
        """
        wins_model_1 = 0
        wins_model_2 = 0
        for i in range(g):

            # Resetting the board and randomly choosing starting player
            self.game.reset(random_start_player=False)

            if i % 2 == 0:
                self.game.black_to_play = True
            else:
                self.game.black_to_play = False

            if display:
                self.game.display_state(
                    self.game.get_position(),
                    info=f"{model1_name} vs. {model2_name} - Game {i+1}")
                time.sleep(delay)

            # Getting start position
            state = np.array(list(self.game.get_position())).reshape(1, -1)

            black_player_turn = self.game.black_to_play

            while not self.game.game_over(display_winner=True):
                if black_player_turn:
                    probs = model1.predict(state)
                else:
                    probs = model2.predict(state)

                # Getting indices of all legal actions
                legal_actions_idx = self.game.get_legal_actions(
                    self.game.get_position())

                # Getting the probabilities and normalizing over them
                legal_actions_probs = normalize_vector(
                    probs[:, legal_actions_idx])

                # Creating a new array of zeroes
                legal_actions = np.zeros_like(probs)

                legal_actions[:, legal_actions_idx] = legal_actions_probs

                legal_actions = legal_actions.flatten()

                a = np.argmax(legal_actions)

                self.game.play(a)

                black_player_turn = not black_player_turn

                if display:
                    self.game.display_state(
                        self.game.get_position(),
                        info=f"{model1_name} vs. {model2_name} - Game {i+1}")
                    time.sleep(delay)

            if self.game.black_wins() == 1:
                wins_model_1 += 1
            else:
                wins_model_2 += 1

        return wins_model_1, wins_model_2

    def play_topp(self, show_game=False, delay=1):
        """
        Playing TOPP Tournament between all players
        """
        # Getting dictionary of models
        models_dict = self.load_models()
        # Instantiating a new dictionary counting model wins
        model_wins = dict()
        for model_name in models_dict.keys():
            model_wins[model_name] = 0

        # Converting dict of models to list of tuples
        models_list = [(name, model) for name, model in models_dict.items()]

        # Iterating over all models (except last)
        for i in range(len(models_list) - 1):
            # Current model
            curr_model_name, curr_model_nn = models_list[i]

            # Other models
            rest_of_models = models_list[i + 1:]

            # Iterating over other models
            for rest_model in rest_of_models:

                rest_model_name, rest_model_nn = rest_model

                # Playing to sets of games, where they switch who plays as black
                curr_model_wins, rest_model_wins = self.play(
                    curr_model_nn,
                    rest_model_nn,
                    self.g,
                    model1_name=curr_model_name,
                    model2_name=rest_model_name,
                    display=show_game,
                    delay=delay)

                print(
                    f"{curr_model_name} plays {rest_model_name}: result {curr_model_wins} - {rest_model_wins}"
                )

                model_wins[curr_model_name] += curr_model_wins
                model_wins[rest_model_name] += rest_model_wins

        # Plotting the results
        lists = model_wins.items()
        players, wins = zip(*lists)
        players = [x.replace("player", "") for x in players]
        fig = plt.figure()
        plt.xlabel("No. episodes Actor has trained")
        plt.ylabel("Wins")
        plt.title("TOPP results")
        plt.bar(players, wins)
        plt.show()

        return model_wins

    def play_against_model(self, model):
        # Resetting the board and randomly choosing starting player
        self.game.reset(random_start_player=True)

        # Getting start position
        state = np.array(list(self.game.get_position())).reshape(1, -1)

        black_player_turn = self.game.black_to_play

        while not self.game.game_over():
            self.game.display_state(self.game.get_position())

            if black_player_turn:
                probs = model.predict(state)
                # Getting indices of all legal actions
                legal_actions_idx = self.game.get_legal_actions(
                    self.game.get_position())

                # Getting the probabilities and normalizing over them
                legal_actions_probs = normalize_vector(
                    probs[:, legal_actions_idx])

                # Creating a new array of zeroes
                legal_actions = np.zeros_like(probs)

                legal_actions[:, legal_actions_idx] = legal_actions_probs

                a = np.argmax(legal_actions)
            else:
                a = int(input("You choose action: "))

            self.game.play(a)

            black_player_turn = not black_player_turn

        self.game.display_state(self.game.get_position())
        if self.game.black_wins() == 1:
            return "Black player (player 1) WON"
        else:
            return "Red player (player 2) WON"


def normalize_vector(vector):
    """
    Normalizes a vector (np.array)
    """
    return vector / np.sum(vector)
