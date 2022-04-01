from config_parser import ConfigParser


def main():
    cfg_parser = ConfigParser("config_hex_4.ini")
    rls = cfg_parser.create_rl_system()
    rls.rl_algorithm(show_game=True)


if __name__ == "__main__":
    main()