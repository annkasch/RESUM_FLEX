"""Run: python -m core.surrogates train TRAINING.yaml
python -m core.surrogates predict RUN/checkpoints/best DATA_DIR PARTITION
python -m core.surrogates mfgp GP.yaml
python -m core.surrogates binomial BINOMIAL_GP.yaml
python -m core.surrogates stream STREAM.yaml   (torchrun for several GPUs)
python -m core.surrogates residual RESIDUAL.yaml   (residual test on a finished run)
python -m core.surrogates within_point WITHIN_POINT.yaml   (within-design-point metrics on a
    finished run)"""

import argparse

from core.surrogates.experiment import run_experiment
from schemas.surrogates import load_surrogate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train any configured event surrogate")
    train.add_argument("config")
    predict = commands.add_parser(
        "predict",
        help="Mean predicted p per design point of a prepared partition (no labels needed)",
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
        "--events-from",
        metavar="PARTITION/FIDELITY",
        help="use the event inputs of seeded random donor design points of this partition (a fresh "
        "event sample, never the design point itself; may be the predicted partition) instead of "
        "each design point's own; moved by the provider's transfer_events if it has one",
    )
    predict.add_argument("--seed", type=int, default=0, help="donor draw seed")
    predict.add_argument(
        "--event-logits",
        action="store_true",
        help="also write every event's logit (float32 shards + index) next to the result",
    )
    mfgp = commands.add_parser("mfgp", help="Fit a standalone multi-fidelity GP")
    mfgp.add_argument("config")
    binomial = commands.add_parser("binomial", help="Fit a binomial count GP on hit counts")
    binomial.add_argument("config")
    stream = commands.add_parser("stream", help="Train a neural field on an event store")
    stream.add_argument("config")
    residual = commands.add_parser(
        "residual", help="Residual learnability test on a finished training or stream run"
    )
    residual.add_argument("config")
    within_point = commands.add_parser(
        "within_point", help="Within-design-point metrics on a finished training or stream run"
    )
    within_point.add_argument("config")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    if args.command == "residual":
        from core.surrogates.residual_stage import run_residual
        from schemas.surrogates import load_residual_config

        output = run_residual(load_residual_config(args.config))
        print(f"Residual test written to {output}", flush=True)
        return
    if args.command == "within_point":
        from core.surrogates.within_point_stage import run_within_point
        from schemas.surrogates import load_within_point_config

        output = run_within_point(load_within_point_config(args.config))
        print(f"Within-design-point metrics written to {output}", flush=True)
        return
    if args.command == "stream":
        from core.surrogates.stream_training import run_stream
        from schemas.surrogates import load_stream_config

        config = load_stream_config(args.config)
        print(f"Training a neural field on events of {config.event_store}", flush=True)
        report = run_stream(config)
        print({k: report[k] for k in ("steps", "events_seen", "best")}, flush=True)
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
                events_from=tuple(args.events_from.split("/")) if args.events_from else None,
                seed=args.seed,
                event_logits=args.event_logits,
            ),
            flush=True,
        )
        return
    config = load_surrogate_config(args.config)
    print(f"Training {config.model.kind}; selecting by {config.selection}", flush=True)
    print(run_experiment(config), flush=True)


if __name__ == "__main__":
    main()
