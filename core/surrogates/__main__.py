"""Run: python -m core.surrogates train config.optical.resum.yaml
python -m core.surrogates predict RUN/checkpoints/best DATA_DIR gp_unlabeled"""

import argparse

from core.surrogates.experiment import run_experiment
from schemas.surrogates import load_surrogate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train any configured event surrogate")
    train.add_argument("config")
    predict = commands.add_parser(
        "predict", help="Mean predicted p per voxel of a prepared partition (no labels needed)"
    )
    predict.add_argument("checkpoint")
    predict.add_argument("data_directory")
    predict.add_argument("partition")
    predict.add_argument("--fidelity", default="lf", choices=["lf", "hf"])
    predict.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    if args.command == "predict":
        from core.surrogates.inference import predict_partition

        print(
            predict_partition(
                args.checkpoint,
                args.data_directory,
                args.partition,
                args.fidelity,
                threads=args.threads,
            ),
            flush=True,
        )
        return
    config = load_surrogate_config(args.config)
    print(f"Training {config.model.kind}; selecting by {config.selection}", flush=True)
    print(run_experiment(config), flush=True)


if __name__ == "__main__":
    main()
