"""Prepared-data acquisition; preparation never replaces an existing snapshot."""

import tempfile
from pathlib import Path

import yaml

from core.experiments.storage import snapshot_dataset


def materialize(spec, store):
    if spec.prepared is not None:
        return snapshot_dataset(spec.prepared, store)
    path = spec.preparation
    raw = yaml.safe_load(path.read_text())
    with tempfile.TemporaryDirectory(prefix="resum-prepare-") as temp:
        root = Path(temp)
        if spec.format == "counts":
            from data.optical_counts import prepare_optical_counts

            if "backend" in raw:
                from schemas.count_gp import load_count_gp_config

                cfg = load_count_gp_config(path).data
            else:
                from schemas.optical import load_optical_config

                cfg = load_optical_config(path)
            prepare = prepare_optical_counts
        elif "test_source" in raw:
            from data.optical_transfer import prepare_optical_transfer_data
            from schemas.optical_transfer import load_optical_transfer_config

            cfg = load_optical_transfer_config(path)
            prepare = prepare_optical_transfer_data
        else:
            from data.optical_pipeline import prepare_optical_data
            from schemas.optical import load_optical_config

            cfg = load_optical_config(path)

            def prepare(c):
                return prepare_optical_data(c, update_manifest=True)

        # Preserve existing assignments, without modifying their source manifest.
        if cfg.split.manifest.exists():
            import shutil

            shutil.copyfile(cfg.split.manifest, root / "split.json")
        cfg.output_directory = root / "prepared"
        cfg.split.manifest = root / "split.json"
        prepare(cfg)
        # Preparation paths are operational details, not dataset identity.
        import json

        for artifact in cfg.output_directory.rglob("*.json"):
            text = artifact.read_text().replace(str(root), "<preparation>")
            json.loads(text)
            artifact.write_text(text)
        return snapshot_dataset(cfg.output_directory, store)
