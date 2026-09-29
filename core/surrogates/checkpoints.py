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


def import_legacy_surrogate(path, *, dim_theta=None, dim_phi=None):
    """Adapt existing single-logit/BDT experiment checkpoints without retraining.

    Old Gaussian CNP checkpoints remain supported by core.training.load_checkpoint;
    they cannot be silently treated as single-logit classifiers. For tree files
    lacking feature metadata, supply dimensions explicitly.
    """
    path = Path(path)
    if path.suffix == ".joblib":
        from core.surrogate_bdt import load_bdt

        estimator, payload = load_bdt(path)
        metadata = payload.get("metadata", {})
        features = metadata.get("features", {})
        if dim_theta is None:
            dim_theta = features.get("theta", {}).get("dimension")
        if dim_phi is None:
            dim_phi = features.get("phi", {}).get("dimension")
        if dim_theta is None and dim_phi is None:
            raise ValueError("Supply dim_theta/dim_phi for a tree without feature metadata")
        from schemas.bdt import BDTConfig

        params = estimator.get_params()
        architecture = {key: params[key] for key in BDTConfig.model_fields if key in params}
        architecture["eval_every"] = min(25, estimator.max_iter)
        architecture["seed"] = params.get("random_state", 0)
        model = build_surrogate({"kind": "bdt", "architecture": architecture}, dim_theta, dim_phi)
        if estimator.n_features_in_ != (dim_theta or 0) + (dim_phi or 0):
            raise ValueError("Supplied feature dimensions do not match the saved tree")
        model.estimator = estimator
    else:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        kinds = {
            "bernoulli_cnp_single_logit": "cnp",
            "bernoulli_mlp_single_logit": "mlp",
            "bernoulli_transformer_single_logit": "transformer",
        }
        kind = kinds.get(payload.get("model_kind"))
        if kind is None:
            raise ValueError("Only single-logit legacy neural checkpoints can be imported")
        spec = {"kind": kind}
        if kind == "transformer":
            spec["architecture"] = payload["transformer_config"]
        elif kind == "mlp":
            spec["architecture"] = {
                k: payload["encoder_config"][k] for k in ("hidden_dims", "dropout")
            }
        else:
            spec["encoder"] = payload["encoder_config"]
        model = build_surrogate(spec, payload["dim_theta"], payload["dim_phi"])
        model.module.load_state_dict(payload["model_state"])
        model.module.eval()
    model.metadata = {
        "legacy_source": str(path.resolve()),
        "legacy_metadata": payload.get("metadata", {}),
    }
    return model
