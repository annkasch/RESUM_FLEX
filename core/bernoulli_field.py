"""Neural field: one Bernoulli logit per event from encoded continuous coordinates.

Inputs per event (all optional, from schemas.surrogates.FieldModel):

* position = theta + selected phi columns, summed in physical units (for example voxel
  centre + emission offset = emission point), expressed in normalized theta units and
  Fourier-encoded: [x, sin(2π f_k x), cos(2π f_k x)] for octave-spaced f_k up to f_max;
* direction = a 3-vector of phi columns, normalized to unit length and encoded by real
  orthonormal spherical harmonics up to the configured degree, or (``encoding: momentum``)
  the normalized vector itself, which keeps its length (the photon wavelength);
* every other phi column unencoded.

With ``encoding: {kind: none}`` and a momentum direction the field is a plain MLP on
(position, momentum): the photon-trained MLP (``kind: mlp`` in a photon run config).

An MLP maps the concatenated encodings to the logit. Without the encodings an MLP on raw
coordinates learns mainly smooth (low-frequency) functions; the Fourier features let it
represent structure down to a half-period of 1 / (2 f_max).

Position units. By default the field position is in normalized theta units of the voxel
data it is trained on. A field trained on ungrouped photons (``forward_photons``, e.g. a
homogeneous photon store) instead fixes its own units with ``set_photon_normalization``:
position = (x - origin) / unit in physical coordinates. ``set_normalization`` then maps voxel
inputs (theta + offset) into those same units, so the photon-trained field is evaluated on
voxel data unchanged.
"""

import math

import torch
from torch import nn

from core.networks import MLPEncoder


class FourierFeatures(nn.Module):
    """x [..., D] -> [..., D (1 + 2 n)]: raw value, then sin and cos per frequency."""

    def __init__(self, n_frequencies: int, max_frequency: float):
        super().__init__()
        exponents = torch.arange(n_frequencies, dtype=torch.float32) - (n_frequencies - 1)
        self.register_buffer("frequencies", max_frequency * 2.0**exponents)  # [n]
        self.n_frequencies = n_frequencies

    def output_dim(self, dim: int) -> int:
        return dim * (1 + 2 * self.n_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * x[..., None] * self.frequencies  # [..., D, n]
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [..., D, 2n]
        return torch.cat([x, encoded.flatten(-2)], dim=-1)


class SphericalHarmonics(nn.Module):
    """Unit vectors u [..., 3] -> real orthonormal spherical harmonics [..., (L + 1)²].

    Ordered by degree l, then m = -l..l. Uses P_l^m(z) / sin^m θ (a polynomial in z) times
    Re/Im (x + i y)^m, so it is smooth at the poles; orthonormal over the sphere with
    respect to the area element (∫ Y_a Y_b dΩ = δ_ab).
    """

    def __init__(self, degree: int):
        super().__init__()
        if degree < 0:
            raise ValueError("degree must be nonnegative")
        self.degree = degree

    @property
    def output_dim(self) -> int:
        return (self.degree + 1) ** 2

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        x, y, z = u[..., 0], u[..., 1], u[..., 2]
        L = self.degree
        # Re/Im of (x + i y)^m, m = 0..L.
        cos_m, sin_m = [torch.ones_like(x)], [torch.zeros_like(x)]
        for _ in range(L):
            c, s = cos_m[-1], sin_m[-1]
            cos_m.append(c * x - s * y)
            sin_m.append(s * x + c * y)
        # P[l][m] = P_l^m(z) / sin^m θ via the standard recurrence (no Condon-Shortley).
        P = [[None] * (L + 1) for _ in range(L + 1)]
        for m in range(L + 1):
            P[m][m] = torch.full_like(z, float(math.prod(range(1, 2 * m, 2))))  # (2m-1)!!
            if m + 1 <= L:
                P[m + 1][m] = z * (2 * m + 1) * P[m][m]
            for l in range(m + 2, L + 1):  # noqa: E741
                P[l][m] = ((2 * l - 1) * z * P[l - 1][m] - (l + m - 1) * P[l - 2][m]) / (l - m)
        out = []
        for l in range(L + 1):  # noqa: E741
            for m in range(-l, l + 1):
                a = abs(m)
                k = math.sqrt(
                    (2 * l + 1) / (4 * math.pi) * math.factorial(l - a) / math.factorial(l + a)
                )
                if m == 0:
                    out.append(k * P[l][0])
                elif m > 0:
                    out.append(math.sqrt(2) * k * P[l][a] * cos_m[a])
                else:
                    out.append(math.sqrt(2) * k * P[l][a] * sin_m[a])
        return torch.stack(out, dim=-1)


class BernoulliField(nn.Module):
    """Neural field over (position, direction, other event inputs); see module docstring."""

    def __init__(self, config, dim_theta, dim_phi):
        super().__init__()
        self.dim_theta, self.dim_phi = dim_theta, dim_phi
        position, direction = config.position, config.direction
        used = [c for g in (position, direction) if g is not None for c in g.phi_columns]
        if used and (dim_phi is None or max(used) >= dim_phi):
            raise ValueError(f"phi_columns {used} need phi with more than {max(used)} columns")
        if position is not None:
            if dim_theta is None or len(position.phi_columns) != dim_theta:
                raise ValueError(
                    "position.phi_columns must match theta one to one (same dimension)"
                )
        self.position_columns = [] if position is None else list(position.phi_columns)
        self.direction_columns = [] if direction is None else list(direction.phi_columns)
        self.extra_columns = [c for c in range(dim_phi or 0) if c not in used]
        dim_position = dim_theta or 0
        # Affine maps from normalized inputs to physical-unit combinations (set_normalization):
        # field position = theta * theta_coefficient + phi[position] * position_scale
        # + position_shift; direction (up to scale) = phi * direction_scale + direction_shift.
        n_pos, n_dir = len(self.position_columns), len(self.direction_columns)
        self.register_buffer("theta_coefficient", torch.ones(dim_position))
        self.register_buffer("position_scale", torch.ones(n_pos))
        self.register_buffer("position_shift", torch.zeros(n_pos))
        # Field units from photon data (set_photon_normalization); unused while photon_units = 0.
        self.register_buffer("position_origin", torch.zeros(dim_position))
        self.register_buffer("position_unit", torch.ones(dim_position))
        self.register_buffer("photon_units", torch.zeros(()))
        self.register_buffer("direction_scale", torch.ones(n_dir))
        self.register_buffer("direction_shift", torch.zeros(n_dir))
        # Momentum encoding: (vector - momentum_origin) / momentum_unit in physical units; the
        # normalization of the training data (voxel phi or photon store).
        self.register_buffer("momentum_origin", torch.zeros(n_dir))
        self.register_buffer("momentum_unit", torch.ones(n_dir))
        encoding = config.encoding
        self.fourier = (
            FourierFeatures(encoding.n_frequencies, encoding.max_frequency)
            if encoding.kind == "fourier"
            else None
        )
        self.momentum = direction is not None and direction.encoding == "momentum"
        self.harmonics = (
            SphericalHarmonics(direction.degree)
            if direction is not None and not self.momentum
            else None
        )
        width_position = self.fourier.output_dim(dim_position) if self.fourier else dim_position
        width_direction = (
            0 if direction is None else 3 if self.momentum else self.harmonics.output_dim
        )
        width = width_position + width_direction + len(self.extra_columns)
        if width == 0:
            raise ValueError("The field has no inputs")
        arch = config.architecture
        self.decoder = MLPEncoder(width, list(arch.hidden_dims), 1, arch.dropout)

    # Buffers added after the first field checkpoints; older weights load with their defaults.
    _OPTIONAL_BUFFERS = (
        "theta_coefficient",
        "position_origin",
        "position_unit",
        "photon_units",
        "momentum_origin",
        "momentum_unit",
    )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in self._OPTIONAL_BUFFERS:
            state_dict.setdefault(prefix + name, getattr(self, name).detach().clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def uses_photon_units(self) -> bool:
        return bool(self.photon_units)

    def set_photon_normalization(self, normalization):
        """Fix the field units from photon data: position = (x - offset[:3]) / scale[:3].

        ``normalization``: dict with ``offset`` and ``scale`` of x, y, z, px, py, pz (physical
        units, as from a photon store). Needs a 3-D position (theta + position columns) and
        a direction; call before ``set_normalization`` of any voxel data.
        """
        if self.dim_theta != 3 or len(self.position_columns) != 3 or not self.direction_columns:
            raise ValueError("Photon training needs a 3-D position and a direction encoding")
        if self.extra_columns:
            raise ValueError("Photon training supports no unencoded extra phi columns")
        offset = torch.as_tensor(normalization["offset"], dtype=torch.float64)
        scale = torch.as_tensor(normalization["scale"], dtype=torch.float64)
        if offset.shape != (6,) or scale.shape != (6,) or torch.any(scale <= 0):
            raise ValueError("Photon normalization needs offset and positive scale of 6 columns")
        self.position_origin.copy_(offset[:3].float())
        self.position_unit.copy_(scale[:3].float())
        self.momentum_origin.copy_(offset[3:].float())
        self.momentum_unit.copy_(scale[3:].float())
        self.photon_units.fill_(1.0)

    def forward_photons(self, photons: torch.Tensor) -> torch.Tensor:
        """Logits [M] of raw photon records [M, >= 6]: x, y, z (m), px, py, pz (any unit)."""
        if not self.uses_photon_units:
            raise ValueError("Call set_photon_normalization before forward_photons")
        position = (photons[:, :3] - self.position_origin) / self.position_unit
        parts = [self._position_features(position), self._direction_features(photons[:, 3:6])]
        return self.decoder(torch.cat(parts, dim=-1)).squeeze(-1)

    def _position_features(self, position):
        return position if self.fourier is None else self.fourier(position)

    def _direction_features(self, vector):
        """Encoded direction of nonzero physical vectors [..., 3]."""
        if self.momentum:
            return (vector - self.momentum_origin) / self.momentum_unit
        return self.harmonics(vector / vector.norm(dim=-1, keepdim=True))

    def set_normalization(self, normalization):
        """Read the affine normalization x_n = (x - offset) / scale of theta and phi.

        None (unnormalized data) keeps the identity maps. With photon units the voxel inputs
        are mapped into those units; otherwise normalized theta units are the field units.
        """
        if normalization is None or not (self.position_columns or self.direction_columns):
            if self.uses_photon_units:
                raise ValueError("A photon-trained field needs the voxel data normalization")
            return

        def affine(group, size):
            entry = normalization.get(group) if isinstance(normalization, dict) else None
            if not isinstance(entry, dict) or not {"offset", "scale"} <= entry.keys():
                raise ValueError(
                    f"The field needs an affine {group} normalization with offset and scale"
                )
            offset = torch.as_tensor(entry["offset"], dtype=torch.float64)
            scale = torch.as_tensor(entry["scale"], dtype=torch.float64)
            if offset.shape != (size,) or scale.shape != (size,) or torch.any(scale == 0):
                raise ValueError(f"{group} normalization does not match {size} input columns")
            return offset, scale

        phi_offset, phi_scale = affine("phi", self.dim_phi)
        if self.position_columns:
            theta_offset, theta_scale = affine("theta", self.dim_theta)
            cols = self.position_columns
            if self.uses_photon_units:
                origin = self.position_origin.detach().cpu().double()
                unit = self.position_unit.detach().cpu().double()
            else:
                origin, unit = theta_offset, theta_scale
            self.theta_coefficient.copy_((theta_scale / unit).float())
            self.position_scale.copy_((phi_scale[cols] / unit).float())
            self.position_shift.copy_(((theta_offset + phi_offset[cols] - origin) / unit).float())
        if self.direction_columns:
            cols = self.direction_columns
            self.direction_scale.copy_(phi_scale[cols].float())
            self.direction_shift.copy_(phi_offset[cols].float())
            if not self.uses_photon_units:  # momentum in the units of the training data
                self.momentum_origin.copy_(phi_offset[cols].float())
                self.momentum_unit.copy_(phi_scale[cols].float())

    def encode(self, target) -> torch.Tensor:
        """Concatenated encodings [B, N, width] of a StandardBatch."""
        parameter = self.position_scale
        n_events = target.n_events
        parts = []
        phi = None
        if target.phi is not None:
            phi = torch.as_tensor(target.phi, dtype=torch.float32, device=parameter.device)
        if self.dim_theta:
            theta = torch.as_tensor(target.theta, dtype=torch.float32, device=parameter.device)
            position = theta[:, None, :].expand(-1, n_events, -1) * self.theta_coefficient
            if self.position_columns:
                offset = phi[..., self.position_columns] * self.position_scale
                position = position + offset + self.position_shift
            parts.append(self._position_features(position))
        if self.direction_columns:
            vector = phi[..., self.direction_columns] * self.direction_scale + self.direction_shift
            norm = vector.norm(dim=-1, keepdim=True)  # [B, N, 1]
            valid = torch.as_tensor(target.event_mask, device=norm.device)[..., None]
            if torch.any((norm == 0) & valid):
                raise ValueError("Direction vectors of valid events must be nonzero")
            # Padded slots of ragged batches are not events; any unit vector will do.
            vector = torch.where(valid, vector, torch.ones_like(vector))
            parts.append(self._direction_features(vector))
        if self.extra_columns:
            parts.append(phi[..., self.extra_columns])
        return torch.cat(parts, dim=-1)

    def forward(self, context, target):
        for name, dimension in (("theta", self.dim_theta), ("phi", self.dim_phi)):
            if (getattr(target, name) is None) != (dimension is None):
                raise ValueError(f"{name} presence must match model input dimensions")
        return self.decoder(self.encode(target)).squeeze(-1)

    def predict_beta(self, context, target):
        return self(context, target).sigmoid()
