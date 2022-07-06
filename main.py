from config_parser import ConfigParser


def main(training, play_topp, show_game=False, delay=0.1):
    cfg_parser = ConfigParser("configs/config_hex_5_long.ini")
    rls, topp = cfg_parser.create_rls_topp()
    if training:
        rls.rl_algorithm(show_game=show_game)
    if play_topp:
        topp.play_topp(show_game=show_game, delay=delay)


if __name__ == "__main__":
    # Remember to set "training" to False when showing long run results
    main(training=False, play_topp=True, show_game=False, delay=0.01)