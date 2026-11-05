"""Run: python -m core.surrogates train TRAINING.yaml
python -m core.surrogates predict RUN/checkpoints/best DATA_DIR PARTITION
python -m core.surrogates mfgp GP.yaml
python -m core.surrogates binomial BINOMIAL_GP.yaml
python -m core.surrogates photon PHOTON_FIELD.yaml   (torchrun for several GPUs)"""

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
    predict.add_argument(
        "--data-provider",
        default="checkpoint",
        help="'module:Class' of the dataset provider (default: the one the model was trained on)",
    )
    predict.add_argument(
        "--photons-from",
        metavar="PARTITION/FIDELITY",
        help="use the event inputs of seeded random donor voxels of this partition (a fresh "
        "emission sample) instead of each voxel's own",
    )
    predict.add_argument("--seed", type=int, default=0, help="donor draw seed")
    mfgp = commands.add_parser("mfgp", help="Fit a standalone multi-fidelity GP")
    mfgp.add_argument("config")
    binomial = commands.add_parser("binomial", help="Fit a binomial count GP on hit counts")
    binomial.add_argument("config")
    photon = commands.add_parser("photon", help="Train a neural field on a photon store")
    photon.add_argument("config")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    if args.command == "photon":
        from core.surrogates.photon_training import run_photon_field
        from schemas.surrogates import load_photon_field_config

        config = load_photon_field_config(args.config)
        print(f"Training a neural field on photons of {config.photon_store}", flush=True)
        report = run_photon_field(config)
        print({k: report[k] for k in ("steps", "photons_seen", "best")}, flush=True)
        return
    if args.command == "binomial":
        from core.surrogates.binomial_stage import run_binomial_config
        from schemas.surrogates import load_binomial_config

        config = load_binomial_config(args.config)
        print(f"Fitting a binomial GP on {len(config.observations)} count sets", flush=True)
        print(run_binomial_config(config), flush=True)
        return
    if args.command == "mfgp":
        from core.surrogates.mfgp_stage import run_mfgp_config
        from schemas.surrogates import load_mfgp_config

        config = load_mfgp_config(args.config)
        print(f"Fitting a {len(config.levels)}-level MFGP", flush=True)
        print(run_mfgp_config(config), flush=True)
        return
    if args.command == "predict":
        from core.surrogates.inference import predict_partition

        print(
            predict_partition(
                args.checkpoint,
                args.data_directory,
                args.partition,
                args.fidelity,
                threads=args.threads,
                data_provider=args.data_provider,
                photons_from=tuple(args.photons_from.split("/")) if args.photons_from else None,
                seed=args.seed,
            ),
            flush=True,
        )
        return
    config = load_surrogate_config(args.config)
    print(f"Training {config.model.kind}; selecting by {config.selection}", flush=True)
    print(run_experiment(config), flush=True)


if __name__ == "__main__":
    main()
