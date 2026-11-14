"""Versioned, model-independent checkpoint directories.

As with existing PyTorch/joblib checkpoints, load only trusted local artifacts.
"""

import json
from pathlib import Path

from core.provenance import code_provenance
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
        code=code_provenance(),
    )
    if model.config.kind == "bdt":
        import sklearn

        manifest["backend_version"] = sklearn.__version__
        if model.stream_position is not None:
            manifest["stream_position"] = list(model.stream_position)
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
    posterior = getattr(model, "last_layer", None)
    if posterior is not None:
        import numpy as np

        with (path / "last_layer.npz.tmp").open("wb") as stream:
            np.savez(stream, **posterior.to_arrays())
        (path / "last_layer.npz.tmp").replace(path / "last_layer.npz")
    elif (path / "last_layer.npz").exists():
        (path / "last_layer.npz").unlink()  # a stale posterior must not outlive its weights
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
        model.stream_position = manifest.get("stream_position")
        if model.estimator.n_features_in_ != model.n_features:
            raise ValueError("Checkpoint feature dimensions do not match estimator")
    else:
        import torch

        model.module.load_state_dict(
            torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
        )
        model.module.eval()
        if (path / "last_layer.npz").exists():
            import numpy as np

            from core.last_layer import LastLayerPosterior

            with np.load(path / "last_layer.npz") as arrays:
                model.last_layer = LastLayerPosterior.from_arrays(arrays)
    model.metadata = manifest["metadata"]
    return model
