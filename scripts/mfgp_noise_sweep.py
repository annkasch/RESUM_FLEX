"""MFGP rate bias and coverage vs. hits per design point, on synthetic counts.

Runs :func:`core.mfgp_bias.noise_study` for every (output transform, noise, hits, seed) and
prints one row each: relative bias of the predicted rate (all / dim / bright held-out
points), 1σ/2σ/3σ coverage of fresh observed fractions (all; 1σ/2σ for dim and bright
points) and, in log mode, the bias terms
(pseudocount, Jensen, back-conversion with the learned top-level noise; the
fit corrects the first two unless --max-refits 0). Writes the rows
to ``--out`` (JSON); a fit that fails (e.g. predicted rates outside (0, 1) for binomial
noise) is printed and recorded with ``failed``.

    python scripts/mfgp_noise_sweep.py --hits 1,5,50 --transforms identity,log --seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.mfgp_bias import noise_study  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hits", default="1,5,50", help="mean hits per top-level point")
    parser.add_argument("--transforms", default="identity,log")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-restarts", type=int, default=2)
    parser.add_argument(
        "--max-refits",
        type=int,
        default=None,
        help="default 8 where needed; 0 = log correction off",
    )
    parser.add_argument(
        "--noise", default="learned", help="observation noise(s), e.g. learned,binomial"
    )
    parser.add_argument("--out", type=Path, default=Path("viz_output/mfgp_noise_sweep.json"))
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", module=r"GPy.*|paramz.*")
    rows = []
    header = (
        f"{'transform':9} {'noise':8} {'hits':>5} {'dim':>5} {'seed':>4} {'bias':>7} {'dim':>7} "
        f"{'bright':>7} {'cov1':>5} {'cov2':>5} {'cov3':>5} {'dim1':>5} {'dim2':>5} "
        f"{'brt1':>5} {'brt2':>5} {'pseudo':>7} {'jensen':>7} "
        f"{'back':>7}"
    )
    print(header)
    settings = [
        (transform, noise)
        for transform in args.transforms.split(",")
        for noise in args.noise.split(",")
    ]
    for transform, noise in settings:
        for hits in map(float, args.hits.split(",")):
            for seed in map(int, args.seeds.split(",")):
                try:
                    row = noise_study(
                        hits,
                        output_transform=transform,
                        n_restarts=args.n_restarts,
                        max_refits=args.max_refits
                        if transform == "log" or noise == "binomial"
                        else None,
                        observation_noise=noise,
                        seed=seed,
                    )
                except ValueError as error:  # e.g. predicted rates outside (0, 1)
                    rows.append(
                        dict(
                            output_transform=transform,
                            observation_noise=noise,
                            hits=hits,
                            seed=seed,
                            failed=str(error),
                        )
                    )
                    print(f"{transform:9} {noise:8} {hits:5g} {'':5} {seed:4d} failed: {error}")
                    continue
                rows.append(row)
                terms = "".join(
                    f" {row[key]:+7.1%}" if key in row else f" {'':>7}"
                    for key in ("pseudocount_shift", "jensen_shift", "back_conversion_shift")
                )
                cov = row["coverage"]
                print(
                    f"{transform:9} {noise:8} {hits:5g} {row['mean_hits_dim']:5.1f} {seed:4d} "
                    f"{row['rate_bias']:+7.1%} {row['rate_bias_dim']:+7.1%} "
                    f"{row['rate_bias_bright']:+7.1%} {cov['1']:5.2f} {cov['2']:5.2f} "
                    f"{cov['3']:5.2f} {row['coverage_dim']['1']:5.2f} "
                    f"{row['coverage_dim']['2']:5.2f} {row['coverage_bright']['1']:5.2f} "
                    f"{row['coverage_bright']['2']:5.2f}{terms}",
                    flush=True,
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
