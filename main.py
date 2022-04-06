from config_parser import ConfigParser
from topp import TOPP


def main(show_game=False):
    cfg_parser = ConfigParser("configs/config_hex_4_short.ini")
    rls = cfg_parser.create_rl_system()
    rls.rl_algorithm(show_game=show_game)


def topp(show_game=False, delay=1):
    topp = TOPP(g=2, k=5, num_actual_games=200, checkpoints=11)
    print(topp.play_topp(show_game=show_game, delay=delay))


if __name__ == "__main__":
    # Main function to train a actor network given a config file
    # main(show_game=False)

    # Running TOPP on
    topp(show_game=False, delay=0.05)