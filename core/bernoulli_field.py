"""Neural field: one Bernoulli logit per event from encoded continuous coordinates.

Inputs per event (all optional, from schemas.surrogates.FieldModel):

* encoded columns (``encoding.columns``) = theta (default) or a list of phi columns taken as
  they are (phi holds the event's own inputs, e.g. the emission point; theta is then not an
  input), in normalized units and Fourier-encoded: [x, sin(2π f_k x), cos(2π f_k x)] for
  octave-spaced f_k up to f_max;
* direction = a 3-vector of phi columns, normalized to unit length and encoded by real
  orthonormal spherical harmonics up to the configured degree, or (``encoding: momentum``)
  the normalized vector itself, which keeps its length (e.g. energy or wavelength);
* every other phi column unencoded.

The toolkit never adds theta to phi: an application whose events are stored relative to
their design point makes them absolute in its dataset provider.

With ``encoding: {kind: none, columns: []}`` and no direction the field is a plain MLP on its
normalized phi columns: the stream-trained MLP (``kind: mlp`` in a stream run config), for
any number of columns. With a transformer ``architecture`` it is the stream-trained
transformer (``kind: transformer``), one token per normalized column.

An MLP (or the transformer) maps the concatenated encodings to the logit. Without the
encodings an MLP on raw coordinates learns mainly smooth (low-frequency) functions; the
Fourier features let it represent structure down to a half-period of 1 / (2 f_max).

Units. By default the field works in the normalized units of the design-point data it is
trained on. A field trained on ungrouped events (``forward_events``) instead fixes its own
units with ``set_stream_normalization``: column = (x - origin) / unit in physical units of
the stream. ``set_normalization`` then maps each phi column of design-point data into
those units, so the stream-trained field is evaluated on design-point data unchanged.

Stream record layout (``forward_events``): the ``dim_phi`` event inputs in phi order, then the
label column; e.g. x, y, z, px, py, pz, label.
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
    """Neural field over (encoded columns, direction, other event inputs); see module docstring."""

    def __init__(self, config, dim_theta, dim_phi):
        super().__init__()
        self.dim_theta, self.dim_phi = dim_theta, dim_phi
        direction, columns = config.direction, config.encoding.columns
        self.encodes_theta = columns == "theta"
        self.encoded_columns = [] if self.encodes_theta else list(columns)
        self.direction_columns = [] if direction is None else list(direction.phi_columns)
        used = self.encoded_columns + self.direction_columns
        if used and (dim_phi is None or max(used) >= dim_phi):
            raise ValueError(f"phi_columns {used} need phi with more than {max(used)} columns")
        self.extra_columns = [c for c in range(dim_phi or 0) if c not in used]
        n_phi, n_dir = dim_phi or 0, len(self.direction_columns)
        # phi in field units = phi * phi_scale + phi_shift: the identity for design-point
        # training (normalized phi are the field units); data units -> stream units for a
        # stream-trained field (set_normalization).
        self.register_buffer("phi_scale", torch.ones(n_phi))
        self.register_buffer("phi_shift", torch.zeros(n_phi))
        # Stream units of the phi record columns: (x - phi_origin) / phi_unit in physical
        # units (set_stream_normalization); unused while stream_units = 0.
        self.register_buffer("phi_origin", torch.zeros(n_phi))
        self.register_buffer("phi_unit", torch.ones(n_phi))
        self.register_buffer("stream_units", torch.zeros(()))
        # Direction in physical units = phi * direction_scale + direction_shift (the harmonics
        # need the true vector, not its normalized components).
        self.register_buffer("direction_scale", torch.ones(n_dir))
        self.register_buffer("direction_shift", torch.zeros(n_dir))
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
        dim_encoded = (dim_theta or 0) if self.encodes_theta else len(self.encoded_columns)
        width_encoded = self.fourier.output_dim(dim_encoded) if self.fourier else dim_encoded
        width_direction = (
            0 if direction is None else 3 if self.momentum else self.harmonics.output_dim
        )
        width = width_encoded + width_direction + len(self.extra_columns)
        if width == 0:
            raise ValueError("The field has no inputs")
        arch = config.architecture
        if hasattr(arch, "token_dim"):  # tabular transformer, one token per encoded column
            from core.bernoulli_transformer import FeatureTransformerDecoder

            self.decoder = FeatureTransformerDecoder(arch, width)
        else:
            self.decoder = MLPEncoder(width, list(arch.hidden_dims), 1, arch.dropout)

    @property
    def record_width(self) -> int:
        """Input columns of a stream record: the phi columns."""
        return self.dim_phi or 0

    @property
    def uses_stream_units(self) -> bool:
        return bool(self.stream_units)

    def set_stream_normalization(self, normalization):
        """Fix the field units from stream data: column = (x - offset) / scale.

        ``normalization``: dict with ``offset`` and ``scale`` of the record's ``dim_phi`` input
        columns (physical units, as from an event store). Call before ``set_normalization``
        of any design-point data.
        """
        if self.encodes_theta and self.dim_theta:
            raise ValueError("Stream records hold phi only; a stream field encodes phi columns")
        offset = torch.as_tensor(normalization["offset"], dtype=torch.float64)
        scale = torch.as_tensor(normalization["scale"], dtype=torch.float64)
        width = self.record_width
        if offset.shape != (width,) or scale.shape != (width,) or torch.any(scale <= 0):
            raise ValueError(
                f"Stream normalization needs offset and positive scale of {width} columns "
                "(the phi columns)"
            )
        self.phi_origin.copy_(offset.float())
        self.phi_unit.copy_(scale.float())
        self.stream_units.fill_(1.0)

    def forward_events(self, events: torch.Tensor) -> torch.Tensor:
        """Logits [M] of raw event records [M, >= record_width] (the phi columns in order)."""
        if not self.uses_stream_units:
            raise ValueError("Call set_stream_normalization before forward_events")
        units = (events[:, : self.record_width] - self.phi_origin) / self.phi_unit  # [M, P]
        encoded = units[:, self.encoded_columns] if self.encoded_columns else None
        vector = events[:, self.direction_columns]  # physical units  [M, 3]
        return self.decoder(self._features(encoded, vector, units)).squeeze(-1)

    def _features(self, encoded, vector, units):
        """Concatenated encodings [..., width]: encoded columns, direction, other columns.

        ``encoded``: the encoded columns in field units (or None); ``vector``: the direction
        in physical units; ``units``: all phi columns in field units.
        """
        parts = []
        if encoded is not None:
            parts.append(encoded if self.fourier is None else self.fourier(encoded))
        if self.direction_columns:
            if self.momentum:
                parts.append(units[..., self.direction_columns])
            else:
                parts.append(self.harmonics(vector / vector.norm(dim=-1, keepdim=True)))
        if self.extra_columns:
            parts.append(units[..., self.extra_columns])
        return torch.cat(parts, dim=-1)

    def set_normalization(self, normalization):
        """Read the affine normalization x_n = (x - offset) / scale of the design-point data.

        None (unnormalized data) keeps the identity maps. The direction is mapped back to
        physical units; with stream units every phi column is mapped into those units.
        """
        stream = self.uses_stream_units
        if normalization is None:
            if stream:
                raise ValueError("A stream-trained field needs the design-point data normalization")
            return
        if not (self.direction_columns or (stream and self.dim_phi)):
            return
        entry = normalization.get("phi") if isinstance(normalization, dict) else None
        if not isinstance(entry, dict) or not {"offset", "scale"} <= entry.keys():
            raise ValueError("The field needs an affine phi normalization with offset and scale")
        offset = torch.as_tensor(entry["offset"], dtype=torch.float64)
        scale = torch.as_tensor(entry["scale"], dtype=torch.float64)
        size = self.dim_phi
        if offset.shape != (size,) or scale.shape != (size,) or torch.any(scale == 0):
            raise ValueError(f"phi normalization does not match {size} input columns")
        if self.direction_columns:
            self.direction_scale.copy_(scale[self.direction_columns].float())
            self.direction_shift.copy_(offset[self.direction_columns].float())
        if stream:  # data units -> stream units, column by column
            origin = self.phi_origin.detach().cpu().double()
            unit = self.phi_unit.detach().cpu().double()
            self.phi_scale.copy_((scale / unit).float())
            self.phi_shift.copy_(((offset - origin) / unit).float())

    def encode(self, target) -> torch.Tensor:
        """Concatenated encodings [B, N, width] of a StandardBatch."""
        device = self.phi_scale.device
        units = vector = encoded = None
        if target.phi is not None:
            phi = torch.as_tensor(target.phi, dtype=torch.float32, device=device)  # [B, N, P]
            units = phi * self.phi_scale + self.phi_shift
        if self.encodes_theta and self.dim_theta:
            theta = torch.as_tensor(target.theta, dtype=torch.float32, device=device)  # [B, T]
            encoded = theta[:, None, :].expand(-1, target.n_events, -1)
        elif self.encoded_columns:
            encoded = units[..., self.encoded_columns]
        if self.direction_columns and not self.momentum:
            vector = phi[..., self.direction_columns] * self.direction_scale + self.direction_shift
            norm = vector.norm(dim=-1, keepdim=True)  # [B, N, 1]
            valid = torch.as_tensor(target.event_mask, device=norm.device)[..., None]
            if torch.any((norm == 0) & valid):
                raise ValueError("Direction vectors of valid events must be nonzero")
            # Padded slots of ragged batches are not events; any unit vector will do.
            vector = torch.where(valid, vector, torch.ones_like(vector))
        return self._features(encoded, vector, units)

    def forward(self, context, target):
        for name, dimension in (("theta", self.dim_theta), ("phi", self.dim_phi)):
            if (getattr(target, name) is None) != (dimension is None):
                raise ValueError(f"{name} presence must match model input dimensions")
        return self.decoder(self.encode(target)).squeeze(-1)

    def predict_beta(self, context, target):
        return self(context, target).sigmoid()
