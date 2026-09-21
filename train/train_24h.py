from .common import build_parser, run_training

if __name__ == "__main__":
    run_training(build_parser(24).parse_args())
