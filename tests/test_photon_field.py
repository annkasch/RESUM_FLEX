"""Neural field trained on ungrouped photons: units, voxel evaluation, training run, schema."""

import json

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from core.photon_data import photon_labels
from core.surrogates import load_surrogate
from core.surrogates.photon_training import run_photon_field
from schemas.surrogates import PhotonFieldRunConfig
from tests.test_bernoulli_field import (
    NORMALIZATION,
    SPEC,
    _field,
    _physical_batch,
    _random_physical,
)

# Store normalization of x, y, z (m) and px, py, pz (MeV), deliberately unlike the voxel one.
PHOTON_NORMALIZATION = {
    "offset": [0.45, -0.21, 1.02, 0.0, 0.0, 0.0],
    "scale": [0.08, 0.04, 0.09, 5.6e-6, 5.6e-6, 5.6e-6],
}


def _records(position, momentum, label=None):
    """Raw photon records [M, 7] float32 with label bits in the last column."""
    out = np.zeros((len(position), 7), dtype=np.float32)
    out[:, :3], out[:, 3:6] = position, momentum
    bits = np.zeros(len(position), dtype=np.int32) if label is None else label.astype(np.int32)
    out[:, 6] = bits.view(np.float32)
    return out


class FakePhotonSource:
    """In-memory photon source over ``<store>/<split>.npy`` records (see core.photon_data)."""

    def __init__(self, store, split, *, seed=0, rank=0, world_size=1, **_loading):
        self.records = np.load(f"{store}/{split}.npy")
        self.seed, self.rank, self.world_size = seed, rank, world_size
        self.n_photons = len(self.records)

    def normalization(self):
        return PHOTON_NORMALIZATION

    def n_batches(self, batch_size, pass_index, *, drop_last=False):
        n = len(self.records) // self.world_size
        return n // batch_size if drop_last else -(-n // batch_size)

    def batches(self, batch_size, pass_index=0, *, start_batch=0, drop_last=False, device="cpu"):
        order = np.random.default_rng([self.seed, pass_index]).permutation(len(self.records))
        order = order[self.rank :: self.world_size]
        for b in range(start_batch, self.n_batches(batch_size, pass_index, drop_last=drop_last)):
            chunk = self.records[order[b * batch_size : (b + 1) * batch_size]]
            yield torch.as_tensor(chunk, device=device)

    def load(self):
        return self.records


def test_photon_trained_field_gives_the_same_logits_on_voxel_inputs():
    model = _field()
    model.module.set_photon_normalization(PHOTON_NORMALIZATION)
    model.configure_normalization(NORMALIZATION)  # voxel data of the evaluation
    rng = np.random.default_rng(1)
    centre, offset, direction = _random_physical(rng)
    batch = _physical_batch(centre, offset, direction)
    with torch.no_grad():
        voxel = model.module(None, batch)  # [B, N]
        position = (centre[:, None, :] + offset).reshape(-1, 3)
        photons = torch.as_tensor(_records(position, direction.reshape(-1, 3)))
        direct = model.module.forward_photons(photons).reshape(voxel.shape)
    torch.testing.assert_close(voxel, direct, rtol=1e-4, atol=1e-4)


def test_voxel_trained_field_is_unchanged_and_old_weights_load():
    model = _field()
    rng = np.random.default_rng(2)
    batch = _physical_batch(*_random_physical(rng))
    state = model.module.state_dict()
    old = {k: v for k, v in state.items() if k not in model.module._OPTIONAL_BUFFERS}
    restored = _field(seed=5)
    restored.module.load_state_dict(old)
    with torch.no_grad():
        torch.testing.assert_close(restored.module(None, batch), model.module(None, batch))
    assert not restored.module.uses_photon_units
    with pytest.raises(ValueError, match="set_photon_normalization"):
        model.module.forward_photons(torch.zeros((2, 7)))


def _photon_split(rng, n):
    """Photons in a box with p(detected) rising along x (a learnable rate)."""
    offset = np.asarray(PHOTON_NORMALIZATION["offset"][:3])
    position = offset + rng.uniform(-0.1, 0.1, (n, 3))
    momentum = rng.normal(size=(n, 3)) * 9.7e-6
    p = 1 / (1 + np.exp(-(-3.0 + 25.0 * (position[:, 0] - offset[0]))))
    return _records(position, momentum, rng.random(n) < p)


def _run_config(tmp_path, **training):
    store = tmp_path / "store"
    store.mkdir()
    rng = np.random.default_rng(0)
    for split, n in (("train", 6000), ("validation", 2000)):
        np.save(store / f"{split}.npy", _photon_split(rng, n))
    return PhotonFieldRunConfig.model_validate(
        {
            "model": {**SPEC, "last_layer": {"prior_precision": "evidence", "draws": 100}},
            "training": {
                "batch_size": 256,
                "n_passes": 3,
                "eval_every": 20,
                "learning_rate": 3e-3,
                "device": "cpu",
                "validation_photons": 1500,
                **training,
            },
            "photon_store": str(store),
            "photon_source": "tests.test_photon_field:FakePhotonSource",
            "output_directory": str(tmp_path / "run"),
        }
    )


def test_photon_training_learns_selects_and_fits_the_last_layer(tmp_path):
    config = _run_config(tmp_path)
    report = run_photon_field(config)
    history = report["history"]
    assert report["steps"] == 3 * -(-6000 // 256)
    assert history[-1]["validation_loss"] < history[0]["validation_loss"]
    run = tmp_path / "run"
    experiment = json.loads((run / "experiment.json").read_text())
    assert experiment["photon_normalization"] == PHOTON_NORMALIZATION
    assert experiment["validation_photons"] == 1500
    assert (run / "compute.json").is_file()
    best = load_surrogate(run / "checkpoints" / "best")
    assert best.module.uses_photon_units and best.last_layer is not None
    assert best.metadata["last_layer"]["events"] == 6000
    assert best.metadata["best_validation_loss"] == pytest.approx(report["best"]["loss"])
    # The selected checkpoint reproduces its recorded validation loss.
    validation = torch.as_tensor(np.load(tmp_path / "store" / "validation.npy")[:1500])
    with torch.no_grad():
        logits = best.module.forward_photons(validation)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, photon_labels(validation))
    assert float(loss) == pytest.approx(report["best"]["loss"], rel=1e-5)
    with pytest.raises(FileExistsError):
        run_photon_field(config)


def test_photon_field_config_needs_position_and_direction():
    base = {"photon_store": "s", "output_directory": "o"}
    PhotonFieldRunConfig.model_validate({"model": SPEC, **base})
    with pytest.raises(ValidationError, match="direction"):
        PhotonFieldRunConfig.model_validate(
            {"model": {k: v for k, v in SPEC.items() if k != "direction"}, **base}
        )
    with pytest.raises(ValidationError, match="all dim_phi"):
        PhotonFieldRunConfig.model_validate({"model": SPEC, "dim_phi": 7, **base})
