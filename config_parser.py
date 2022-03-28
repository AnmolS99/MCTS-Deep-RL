import configparser
from anet import ANet

from hex_game import HexGame
from rl_system import RLSystem


class ConfigParser:
    """
    Configuration file parser
    """

    def __init__(self, config_file) -> None:
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def create_rl_system(self):
        """
        Creates the RL System along with ANet, MCTS and Hex game
        """
        # Creating Hex game
        k = int(self.config["hex"]["k"])
        black_to_play = self.config["hex"]["black_to_play"].lower() == "true"
        hex_game = HexGame(K=k, black_to_play=black_to_play)

        # Creating ANet
        lr = float(self.config["anet"]["lr"])
        optimizer = self.config["anet"]["optimizer"]

        # Parsing the layer vairables for each layer
        input_nodes = hex_game.get_input_dim()
        output_nodes = hex_game.get_output_dim()
        layers = [input_nodes]
        for section in self.config.sections()[4:]:
            neurons = int(self.config[section]["n"])
            act_func = self.config[section]["act"]
            layers.append((neurons, act_func))
        layers.append(output_nodes)
        anet = ANet(layers, lr, optimizer)

        num_search_games = int(self.config["mcts"]["num_search_games"])
        c = float(self.config["mcts"]["c"])
        eps = float(self.config["mcts"]["eps"])

        num_actual_games = int(self.config["rl_system"]["num_actual_games"])
        checkpoints = int(self.config["rl_system"]["checkpoints"])

        return RLSystem(game=hex_game,
                        anet=anet,
                        num_search_games=num_search_games,
                        c=c,
                        eps=eps,
                        num_actual_games=num_actual_games,
                        checkpoints=checkpoints)
