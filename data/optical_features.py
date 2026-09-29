"""Feature construction and training-only affine normalization."""

from dataclasses import asdict, dataclass

import numpy as np

from data.optical_reader import OpticalRun
from schemas.optical import OpticalDataConfig


def center(run: OpticalRun, config: OpticalDataConfig) -> np.ndarray:
    return run.nominal_center if config.theta.center_source == "filename" else run.measured_center


def cylindrical(xyz, *, include_azimuth=True, encoding="sin_cos"):
    x, y, z = np.moveaxis(np.asarray(xyz), -1, 0)
    r = np.hypot(x, y)
    angle = np.where(r == 0, 0, np.arctan2(y, x))
    if not include_azimuth:
        return np.stack([r, z], axis=-1), ["r", "z"], ["m", "m"]
    if encoding == "sin_cos":
        return (
            np.stack([r, np.sin(angle), np.cos(angle), z], axis=-1),
            ["r", "sin_azimuth", "cos_azimuth", "z"],
            ["m", "1", "1", "m"],
        )
    return np.stack([r, angle, z], axis=-1), ["r", "azimuth", "z"], ["m", "rad", "m"]


def make_features(run: OpticalRun, config: OpticalDataConfig):
    c = center(run, config)
    angle_config = config.theta.cylindrical
    theta, tn, tu = c.copy(), ["x", "y", "z"], ["m"] * 3
    if config.theta.coordinates == "cylindrical":
        theta, tn, tu = cylindrical(
            c, include_azimuth=angle_config.include_azimuth, encoding=angle_config.angle_encoding
        )
    parts, pn, pu = [], [], []
    vertex = config.phi.vertex
    if vertex.representation != "omit":
        v = run.positions.copy()
        names, units = ["x", "y", "z"], ["m"] * 3
        if vertex.representation == "offset":
            v -= c
            names = ["dx", "dy", "dz"]
            if vertex.coordinates == "cylindrical":
                angle = 0 if np.hypot(c[0], c[1]) == 0 else np.arctan2(c[1], c[0])
                co, si = np.cos(angle), np.sin(angle)
                v = np.column_stack(
                    [v[:, 0] * co + v[:, 1] * si, -v[:, 0] * si + v[:, 1] * co, v[:, 2]]
                )
                names = ["dr_local", "dt_local", "dz"]
        elif vertex.coordinates == "cylindrical":
            v, names, units = cylindrical(v, encoding=angle_config.angle_encoding)
        parts.append(v)
        pn.extend(names)
        pu.extend(units)
    momentum = config.phi.momentum.representation
    if momentum != "omit":
        m = run.momenta.copy()
        names, units = ["px", "py", "pz"], ["MeV"] * 3
        if momentum == "direction_and_magnitude":
            magnitude = np.linalg.norm(m, axis=1)
            if np.any(magnitude == 0):
                raise ValueError(f"{run.file}: zero momentum cannot define direction")
            m = np.column_stack([m / magnitude[:, None], magnitude])
            names, units = ["ux", "uy", "uz", "momentum_magnitude"], ["1", "1", "1", "MeV"]
        parts.append(m)
        pn.extend(names)
        pu.extend(units)
    phi = np.concatenate(parts, axis=1) if parts else None
    return (
        theta,
        phi,
        {
            "theta": {"names": tn, "units": tu, "dimension": len(tn)},
            "phi": {"names": pn, "units": pu, "dimension": len(pn)},
        },
    )


def make_labels(run: OpticalRun, config: OpticalDataConfig):
    ids = run.hit_event_ids
    if config.target.kind == "single_channel":
        ids = ids[run.detector_ids == config.target.detector_id]
    return np.isin(run.event_ids, ids).astype(np.int8)


@dataclass
class AffineTransform:
    method: str
    offset: list[float]
    scale: list[float]

    @classmethod
    def fit(cls, values, method):
        if not len(values):
            raise ValueError("Cannot fit normalization without training values")
        if method == "standard":
            offset, scale = values.mean(0), values.std(0)
        elif method == "minmax":
            offset, scale = values.min(0), np.ptp(values, axis=0)
        elif method == "none":
            offset, scale = np.zeros(values.shape[-1]), np.ones(values.shape[-1])
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        scale = np.where(np.ptp(values, axis=0) == 0, 1, scale)
        return cls(method, offset.tolist(), scale.tolist())

    def transform(self, values):
        return (values - np.asarray(self.offset)) / np.asarray(self.scale)

    def to_dict(self):
        return asdict(self)
