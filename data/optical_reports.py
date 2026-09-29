"""Per-split audits, kept separate from training inputs and selection logic."""

import csv
import json
from collections import Counter

import numpy as np

from data.optical_features import center, make_labels
from data.optical_split import FIDELITIES, SPLITS


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_reports(runs, assignments, excluded, config, manifest):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = config.output_directory / "reports"
    vocabulary = sorted({int(d) for r in runs for d in r.detector_ids})
    root.mkdir(parents=True, exist_ok=True)
    (root / "integrity.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "validated_files": len(runs),
                "excluded": excluded,
                "split_overlap_groups": manifest["overlapping_groups"],
                "channel_vocabulary": vocabulary,
                "channel_vocabulary_note": (
                    "Observed IDs across included files; not a full channel roster."
                ),
                "test_outcomes": (
                    "Stored separately under reports/test; not used for split assignment."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    for split in SPLITS:
        target = root / ("test" if split == "test" else f"development/{split}")
        target.mkdir(parents=True, exist_ok=True)
        summaries = []
        for fidelity in FIDELITIES:
            selected = [r for r in runs if assignments[r.file] == split and r.fidelity == fidelity]
            per_file, channels = [], []
            for r in selected:
                counts = Counter(map(int, r.detector_ids))
                c = center(r, config)
                positive = int(make_labels(r, config).sum())
                per_file.append(
                    {
                        "file": r.file,
                        "fidelity": fidelity,
                        "x_m": c[0],
                        "y_m": c[1],
                        "z_m": c[2],
                        "radius_m": np.hypot(c[0], c[1]),
                        "events": r.n_events,
                        "hits": len(r.detector_ids),
                        "events_with_hits": len(np.unique(r.hit_event_ids)),
                        "channels_hit": len(counts),
                        **{
                            f"channels_{k}_hits": sum(v == k for v in counts.values())
                            for k in (1, 2, 3, 4)
                        },
                        "channels_ge5_hits": sum(v >= 5 for v in counts.values()),
                        "target_positive": positive,
                        "target_negative": r.n_events - positive,
                    }
                )
            for detector in vocabulary:
                hits = sum(int(np.sum(r.detector_ids == detector)) for r in selected)
                contributing = sum(bool(np.any(r.detector_ids == detector)) for r in selected)
                channels.append(
                    {
                        "detector_id": detector,
                        "hits": hits,
                        "contributing_voxels": contributing,
                        "support": "none"
                        if contributing == 0
                        else "few (<3 voxels)"
                        if contributing < 3
                        else ">=3 voxels",
                    }
                )
            write_csv(
                target / f"{fidelity}_voxels.csv",
                per_file,
                list(per_file[0]) if per_file else ["file", "fidelity"],
            )
            write_csv(
                target / f"{fidelity}_channels.csv",
                channels,
                ["detector_id", "hits", "contributing_voxels", "support"],
            )
            summaries.append(
                {
                    "fidelity": fidelity,
                    "files": len(selected),
                    "events": sum(r.n_events for r in selected),
                    "hits": sum(len(r.detector_ids) for r in selected),
                    "events_with_hits": sum(len(np.unique(r.hit_event_ids)) for r in selected),
                    "zero_hit_voxels": sum(len(r.detector_ids) == 0 for r in selected),
                    "target_positive": sum(r["target_positive"] for r in per_file),
                    "target_negative": sum(r["target_negative"] for r in per_file),
                }
            )
            if not selected:
                for ext in ("png", "pdf"):
                    stale = target / f"{fidelity}_coverage.{ext}"
                    if stale.exists():
                        stale.unlink()
                continue
            centers = np.array([center(r, config) for r in selected])
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            values = [centers[:, i] for i in range(3)] + [np.hypot(*centers[:, :2].T)]
            for ax, values_, label in zip(
                axes.flat, values, ["x (m)", "y (m)", "z (m)", "r (m)"], strict=False
            ):
                ax.hist(values_, bins=min(10, len(selected)), color="#087E8B")
                ax.set(xlabel=label, ylabel="Voxel count")
            axes[1, 1].hist([r["channels_hit"] for r in per_file], bins="auto", color="#D26935")
            axes[1, 1].set(xlabel="Distinct channels hit per voxel", ylabel="Voxel count")
            cats = ["1", "2", "3", "4", "≥5"]
            heights = [sum(r[f"channels_{k}_hits"] for r in per_file) for k in (1, 2, 3, 4)]
            heights += [sum(r["channels_ge5_hits"] for r in per_file)]
            axes[1, 2].bar(cats, heights, color="#087E8B")
            axes[1, 2].set(xlabel="Hits in detector within voxel", ylabel="Detector–voxel pairs")
            fig.suptitle(f"{split} / {fidelity.upper()} — {len(selected)} files")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(target / f"{fidelity}_coverage.{ext}", dpi=150)
            plt.close(fig)
        write_csv(target / "summary.csv", summaries)
