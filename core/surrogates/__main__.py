"""Run: python -m core.surrogates train config.optical.resum.yaml"""

import argparse

from core.surrogates.experiment import run_experiment
from schemas.surrogates import load_surrogate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train any configured event surrogate")
    train.add_argument("config")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    config = load_surrogate_config(args.config)
    print(f"Training {config.model.kind}; selecting by {config.selection}", flush=True)
    print(run_experiment(config), flush=True)


if __name__ == "__main__":
    main()
