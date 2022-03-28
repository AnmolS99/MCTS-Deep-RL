from config_parser import ConfigParser
import cProfile, pstats


def main():
    cfg_parser = ConfigParser("config_hex_4.ini")
    rls = cfg_parser.create_rl_system()
    rls.rl_algorithm(show_game=False)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()