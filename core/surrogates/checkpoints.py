"""Versioned, model-independent checkpoint directories.

As with existing PyTorch/joblib checkpoints, load only trusted local artifacts.
"""

import json
from pathlib import Path

from core.surrogates.models import build_surrogate

FORMAT = "resum_flex.event_surrogate"
VERSION = 1


def save_surrogate(path, model, *, metadata=None):
    path = Path(path)
    manifest = dict(
        format=FORMAT,
        version=VERSION,
        model=model.config.model_dump(mode="json"),
        dim_theta=model.dim_theta,
        dim_phi=model.dim_phi,
        metadata={**model.metadata, **(metadata or {})},
    )
    if model.config.kind == "bdt":
        import sklearn

        manifest["backend_version"] = sklearn.__version__
    else:
        import torch

        manifest["backend_version"] = torch.__version__
    serialized = json.dumps(manifest, indent=2, allow_nan=False)
    if path.exists() and any(path.iterdir()) and not (path / "model.json").exists():
        raise FileExistsError(f"Refusing to overwrite a non-checkpoint directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if model.config.kind == "bdt":
        import joblib

        filename = "estimator.joblib"
        joblib.dump(model.estimator, path / (filename + ".tmp"))
    else:
        import torch

        filename = "weights.pt"
        torch.save(
            {k: v.detach().cpu() for k, v in model.module.state_dict().items()},
            path / (filename + ".tmp"),
        )
    (path / (filename + ".tmp")).replace(path / filename)
    (path / "model.json.tmp").write_text(serialized)
    (path / "model.json.tmp").replace(path / "model.json")


def load_surrogate(path):
    path = Path(path)
    manifest = json.loads((path / "model.json").read_text())
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise ValueError("Unsupported event-surrogate checkpoint format/version")
    model = build_surrogate(manifest["model"], manifest["dim_theta"], manifest["dim_phi"])
    if model.config.kind == "bdt":
        import joblib

        model.estimator = joblib.load(path / "estimator.joblib")
        if model.estimator.n_features_in_ != (model.dim_theta or 0) + (model.dim_phi or 0):
            raise ValueError("Checkpoint feature dimensions do not match estimator")
    else:
        import torch

        model.module.load_state_dict(
            torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        )
        model.module.eval()
    model.metadata = manifest["metadata"]
    return model
