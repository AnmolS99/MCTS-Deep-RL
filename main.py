from config_parser import ConfigParser

if __name__ == "__main__":
    cfg_parser = ConfigParser("config_hex_4.ini")
    rls = cfg_parser.create_rl_system()
    rls.rl_algorithm(show_game=False)