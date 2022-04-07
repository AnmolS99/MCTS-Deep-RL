import configparser
import copy
from anet import ANet

from hex_game import HexGame
from lite_model import LiteModel
from mcts import MCTS
from rl_system import RLSystem
from topp import TOPP


class ConfigParser:
    """
    Configuration file parser
    """

    def __init__(self, config_file) -> None:
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def create_rls_topp(self):
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
        for section in self.config.sections()[5:]:
            neurons = int(self.config[section]["n"])
            act_func = self.config[section]["act"]
            layers.append((neurons, act_func))
        layers.append(output_nodes)
        anet = ANet(layers, lr, optimizer)

        # Creating MCTS
        num_search_games = int(self.config["mcts"]["num_search_games"])
        c = float(self.config["mcts"]["c"])
        eps_mcts = float(self.config["mcts"]["eps_mcts"])
        mcts = MCTS(copy.deepcopy(hex_game),
                    LiteModel.from_keras_model(anet.nn), num_search_games, c,
                    eps_mcts)

        # Parsing RL variables and creating RLSystem
        num_actual_games = int(self.config["rl_system"]["num_actual_games"])
        checkpoints = int(self.config["rl_system"]["checkpoints"])
        rbuf = int(self.config["rl_system"]["rbuf"])
        epochs = int(self.config["rl_system"]["epochs"])
        eps_rl = float(self.config["rl_system"]["eps_rl"])
        rls = RLSystem(game=hex_game,
                       anet=anet,
                       mcts=mcts,
                       eps_rl=eps_rl,
                       num_actual_games=num_actual_games,
                       checkpoints=checkpoints,
                       rbuf_size=rbuf,
                       epochs=epochs)

        # Creating TOPP
        games = int(self.config["topp"]["games"])
        topp = TOPP(g=games,
                    k=k,
                    num_actual_games=num_actual_games,
                    checkpoints=checkpoints)

        return rls, topp
