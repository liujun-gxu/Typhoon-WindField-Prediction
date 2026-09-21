from .common import build_parser, run_training

if __name__ == "__main__":
    run_training(build_parser(12).parse_args())
