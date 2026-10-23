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


def theta_features(centers, config: OpticalDataConfig):
    """Design features of voxel centers [B, 3] -> ([B, D_theta], names, units)."""
    centers = np.asarray(centers, dtype=float)
    if config.theta.coordinates == "cylindrical":
        angle_config = config.theta.cylindrical
        return cylindrical(
            centers,
            include_azimuth=angle_config.include_azimuth,
            encoding=angle_config.angle_encoding,
        )
    return centers.copy(), ["x", "y", "z"], ["m"] * 3


def phi_features(positions, momenta, centers, config: OpticalDataConfig):
    """Event features -> ([E, D_phi] or None, names, units).

    positions, momenta: [E, 3] vertex (m) and momentum (MeV); centers: [E, 3] center of
    each event's voxel. Computed in float64 whatever the input precision.
    """
    parts, pn, pu = [], [], []
    vertex = config.phi.vertex
    if vertex.representation != "omit":
        v = np.asarray(positions, dtype=float)
        names, units = ["x", "y", "z"], ["m"] * 3
        if vertex.representation == "offset":
            c = np.asarray(centers, dtype=float)
            v = v - c
            names = ["dx", "dy", "dz"]
            if vertex.coordinates == "cylindrical":
                angle = np.where(np.hypot(c[:, 0], c[:, 1]) == 0, 0, np.arctan2(c[:, 1], c[:, 0]))
                co, si = np.cos(angle), np.sin(angle)
                v = np.column_stack(
                    [v[:, 0] * co + v[:, 1] * si, -v[:, 0] * si + v[:, 1] * co, v[:, 2]]
                )
                names = ["dr_local", "dt_local", "dz"]
        elif vertex.coordinates == "cylindrical":
            v, names, units = cylindrical(v, encoding=config.theta.cylindrical.angle_encoding)
        parts.append(v)
        pn.extend(names)
        pu.extend(units)
    momentum = config.phi.momentum.representation
    if momentum != "omit":
        m = np.asarray(momenta, dtype=float)
        names, units = ["px", "py", "pz"], ["MeV"] * 3
        if momentum == "direction_and_magnitude":
            magnitude = np.linalg.norm(m, axis=1)
            if np.any(magnitude == 0):
                raise ValueError("zero momentum cannot define direction")
            m = np.column_stack([m / magnitude[:, None], magnitude])
            names, units = ["ux", "uy", "uz", "momentum_magnitude"], ["1", "1", "1", "MeV"]
        parts.append(m)
        pn.extend(names)
        pu.extend(units)
    return (np.concatenate(parts, axis=1) if parts else None), pn, pu


def feature_info(config: OpticalDataConfig) -> dict:
    """Feature names, units and dimensions, without data."""
    _, tn, tu = theta_features(np.zeros((0, 3)), config)
    empty = np.zeros((0, 3))
    _, pn, pu = phi_features(empty, empty, empty, config)
    return {
        "theta": {"names": tn, "units": tu, "dimension": len(tn)},
        "phi": {"names": pn, "units": pu, "dimension": len(pn)},
    }


def make_features(run: OpticalRun, config: OpticalDataConfig):
    c = center(run, config)
    theta = theta_features(c[None], config)[0][0]
    try:
        phi = phi_features(
            run.positions, run.momenta, np.broadcast_to(c, (run.n_events, 3)), config
        )[0]
    except ValueError as exc:
        raise ValueError(f"{run.file}: {exc}") from exc
    return theta, phi, feature_info(config)


def make_labels(run: OpticalRun, config: OpticalDataConfig):
    ids = run.hit_event_ids
    if config.target.kind == "single_channel":
        ids = ids[run.detector_ids == config.target.detector_id]
    return np.isin(run.event_ids, ids).astype(np.int8)


class RunningMoments:
    """Count, mean, variance, minimum and maximum of feature rows, added in chunks.

    Chunks are merged with Chan's parallel update, so the result does not depend on how the
    rows are divided and the full training set never has to be in memory at once.
    """

    def __init__(self):
        self.n, self.mean, self.m2, self.min, self.max = 0, None, None, None, None

    def add(self, values) -> "RunningMoments":
        values = np.asarray(values, dtype=float)  # [E, D]
        n = len(values)
        if not n:
            return self
        mean = values.mean(0)
        m2 = np.square(values - mean).sum(0)
        low, high = values.min(0), values.max(0)
        if self.n == 0:
            self.n, self.mean, self.m2, self.min, self.max = n, mean, m2, low, high
            return self
        total = self.n + n
        delta = mean - self.mean
        self.mean = self.mean + delta * (n / total)
        self.m2 = self.m2 + m2 + np.square(delta) * (self.n * n / total)
        self.min, self.max = np.minimum(self.min, low), np.maximum(self.max, high)
        self.n = total
        return self


@dataclass
class AffineTransform:
    method: str
    offset: list[float]
    scale: list[float]

    @classmethod
    def fit(cls, values, method):
        return cls.from_moments(RunningMoments().add(values), method)

    @classmethod
    def from_moments(cls, moments: RunningMoments, method):
        if not moments.n:
            raise ValueError("Cannot fit normalization without training values")
        ptp = moments.max - moments.min
        if method == "standard":
            offset, scale = moments.mean, np.sqrt(moments.m2 / moments.n)
        elif method == "minmax":
            offset, scale = moments.min, ptp
        elif method == "none":
            offset, scale = np.zeros_like(ptp), np.ones_like(ptp)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        scale = np.where(ptp == 0, 1, scale)
        return cls(method, offset.tolist(), scale.tolist())

    def transform(self, values):
        return (values - np.asarray(self.offset)) / np.asarray(self.scale)

    def to_dict(self):
        return asdict(self)
