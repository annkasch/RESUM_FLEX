"""MFGP rate bias and coverage vs. hits per design point, on synthetic counts.

Runs :func:`core.mfgp_bias.noise_study` for every (output transform, hits, seed) and
prints one row each: relative bias of the predicted rate (all / dim / bright held-out
points), 1σ/2σ/3σ coverage of fresh observed fractions and, in log mode, the bias terms
(pseudocount, Jensen, back-conversion with the learned top-level noise). Writes the rows
to ``--out`` (JSON).

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
    parser.add_argument("--out", type=Path, default=Path("viz_output/mfgp_noise_sweep.json"))
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore", module=r"GPy.*|paramz.*")
    rows = []
    header = (
        f"{'transform':9} {'hits':>5} {'dim':>5} {'seed':>4} {'bias':>7} {'dim':>7} "
        f"{'bright':>7} {'cov1':>5} {'cov2':>5} {'cov3':>5} {'pseudo':>7} {'jensen':>7} "
        f"{'back':>7}"
    )
    print(header)
    for transform in args.transforms.split(","):
        for hits in map(float, args.hits.split(",")):
            for seed in map(int, args.seeds.split(",")):
                row = noise_study(
                    hits, output_transform=transform, n_restarts=args.n_restarts, seed=seed
                )
                rows.append(row)
                terms = "".join(
                    f" {row[key]:+7.1%}" if key in row else f" {'':>7}"
                    for key in ("pseudocount_shift", "jensen_shift", "back_conversion_shift")
                )
                cov = row["coverage"]
                print(
                    f"{transform:9} {hits:5g} {row['mean_hits_dim']:5.1f} {seed:4d} "
                    f"{row['rate_bias']:+7.1%} {row['rate_bias_dim']:+7.1%} "
                    f"{row['rate_bias_bright']:+7.1%} {cov['1']:5.2f} {cov['2']:5.2f} "
                    f"{cov['3']:5.2f}{terms}",
                    flush=True,
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
