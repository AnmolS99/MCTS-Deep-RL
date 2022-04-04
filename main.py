from config_parser import ConfigParser


def main():
    cfg_parser = ConfigParser("configs/config_hex_7.ini")
    rls = cfg_parser.create_rl_system()
    rls.rl_algorithm(show_game=False)


if __name__ == "__main__":
    main()