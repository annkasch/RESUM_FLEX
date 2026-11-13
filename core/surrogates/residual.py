"""Residual learnability test: can gradient-boosted trees improve the model's p?

LightGBM (``binary`` objective = BCE, ``init_score`` = model logit) learns a correction
``logit p' = logit p + f(inputs, logit p)`` by cross-fitting: every unit (design point, or
event for ungrouped streams) is scored once by trees that never saw it. The held-out gain
``T = L(p) - L(p')`` in nats per positive is about 0 if p is right and positive if the
trees found structure the event model missed. For design-point data it splits exactly into
a φ part ``F = W(p') - W(p)`` (W: gain over each design point's own mean, as
``within_point_log_loss_gain``) and a rate part ``T - F``.

LightGBM is imported here only (optional ``lgbm`` extra); inputs and outputs are numpy.
"""

import numpy as np
from scipy.special import expit, logit

from core.surrogates.evaluation import (
    CALIBRATION_BINS,
    bootstrap_bands,
    calibration_bins,
    feature_columns,
)

# SeedSequence key under the evaluation seed (read-out keys 0-3 are taken).
RESIDUAL_KEY = 4
# Sub-streams under RESIDUAL_KEY.
_FOLD_KEY, _BOOTSTRAP_KEY, _TREE_KEY, _BLOCK_KEY, _SUBSAMPLE_KEY = 0, 1, 2, 3, 4
BOOTSTRAP_DRAWS = 200  # as the other read-out intervals
# Ungrouped events are bootstrapped in this many seeded random blocks.
BOOTSTRAP_BLOCKS = 1000


def require_lightgbm():
    """Fail before training if the test is configured but LightGBM is missing."""
    try:
        import lightgbm  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "residual_test needs LightGBM: uv sync --extra lgbm (with your other extras)"
        ) from error


def _rng(seed, key):
    return np.random.default_rng(np.random.SeedSequence([seed, RESIDUAL_KEY, key]))


def _bce(labels, logits):
    """Per-event Bernoulli log loss from logits (stable, never clipped)."""
    return np.where(labels == 1, np.logaddexp(0, -logits), np.logaddexp(0, logits))


def balanced_folds(unit_positives, k, rng):
    """Fold per unit, balanced by positive count: seeded order, then snake assignment."""
    n = len(unit_positives)
    shuffled = rng.permutation(n)
    order = shuffled[np.argsort(-np.asarray(unit_positives)[shuffled], kind="stable")]
    rank = np.arange(n)
    column, row = rank % k, rank // k
    folds = np.empty(n, dtype=np.int64)
    folds[order] = np.where(row % 2 == 0, column, k - 1 - column)
    return folds


def within_gain(groups, labels, logits):
    """Per-group gain over the group's own mean p: W_b = Σ_i [loss(q̄_b) - loss(q_i)].

    groups: [E] in 0..B-1; labels, logits: [E]. Returns [B].
    """
    count = np.bincount(groups)
    with np.errstate(invalid="ignore"):  # empty groups are never indexed
        mean = np.bincount(groups, expit(logits)) / count  # [B]
    if np.any(mean[count > 0] <= 0) or np.any(mean[count > 0] >= 1):
        raise ValueError("Design-point mean probabilities must lie strictly inside (0, 1)")
    baseline = logit(mean[groups])  # [E]
    gain = _bce(labels, baseline) - _bce(labels, logits)
    return np.bincount(groups, gain, minlength=len(count))


def _params(settings, seed):
    return dict(
        objective="binary",
        num_leaves=settings.num_leaves,
        learning_rate=settings.learning_rate,
        min_data_in_leaf=settings.min_data_in_leaf,
        lambda_l2=settings.lambda_l2,
        num_threads=settings.threads,
        seed=seed,
        deterministic=True,
        force_row_wise=True,
        feature_pre_filter=False,
        verbosity=-1,
    )


def _fit_fold(x, y, z, train_units, unit, unit_positives, settings, rng):
    """Trees of one outer fold: inner early stopping, then refit on all training units.

    x: [E, F+1] features with logit p; y, z: [E] labels and model logits; unit: [E].
    Returns (booster or None, number of trees).
    """
    import lightgbm as lgb

    if len(train_units) < 2:  # no unit left for early stopping
        return None, 0
    inner = balanced_folds(unit_positives[train_units], max(2, settings.folds - 1), rng)
    stop_units = train_units[inner == 0]
    stop = np.isin(unit, stop_units)
    fit = np.isin(unit, train_units) & ~stop
    params = _params(settings, int(rng.integers(2**31 - 1)))
    fit_set = lgb.Dataset(x[fit], y[fit], init_score=z[fit])
    stop_set = lgb.Dataset(x[stop], y[stop], init_score=z[stop], reference=fit_set)
    history = {}
    lgb.train(
        {**params, "metric": "binary_logloss"},
        fit_set,
        num_boost_round=settings.max_trees,
        valid_sets=[stop_set],
        valid_names=["stop"],
        callbacks=[
            lgb.early_stopping(settings.patience, verbose=False),
            lgb.record_evaluation(history),
        ],
    )
    curve = np.asarray(history["stop"]["binary_logloss"])
    start = _bce(y[stop], z[stop]).mean()  # zero trees: the model's own p
    trees = int(np.argmin(curve)) + 1 if curve.size and curve.min() < start else 0
    if trees == 0:
        return None, 0
    every = np.isin(unit, train_units)
    refit = lgb.Dataset(x[every], y[every], init_score=z[every])
    return lgb.train(params, refit, num_boost_round=trees), trees


def correction_bins(names, features, labels, logits, corrected, n_bins=CALIBRATION_BINS):
    """Predicted hits before and after the trees' correction, in equal-count bins.

    Columns: ``logit_p`` first, then each input column. names: F; features: [E, F];
    labels, logits (logit p), corrected (logit p'): [E] events the test scored. Arrays
    ``residual_correction_*`` [1+F, bins]: events, hits, predicted (Σp), corrected (Σp'),
    value (median column value per bin); ``residual_correction_names`` [1+F].
    """
    features = np.asarray(features, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    if features.shape != (len(z), len(names)):
        raise ValueError("features must be [E, F] with one name per column")
    columns = np.column_stack([z, features])  # [E, 1+F]
    n_bins = min(n_bins, len(z))
    keys = ("events", "hits", "predicted", "corrected", "value")
    arrays = {f"residual_correction_{k}": np.zeros((columns.shape[1], n_bins)) for k in keys}
    p, p_corrected = expit(z), expit(np.asarray(corrected, dtype=np.float64))
    for c in range(columns.shape[1]):
        before = calibration_bins(labels, p, n_bins, sort_by=columns[:, c])
        after = calibration_bins(labels, p_corrected, n_bins, sort_by=columns[:, c])
        order = np.argsort(columns[:, c], kind="stable")  # the bins of calibration_bins
        arrays["residual_correction_events"][c] = before["calibration_events"]
        arrays["residual_correction_hits"][c] = before["calibration_hits"]
        arrays["residual_correction_predicted"][c] = before["calibration_predicted"]
        arrays["residual_correction_corrected"][c] = after["calibration_predicted"]
        arrays["residual_correction_value"][c] = [
            np.median(columns[b, c]) for b in np.array_split(order, n_bins)
        ]
    arrays["residual_correction_names"] = np.asarray(["logit_p", *names])
    return arrays


def design_point_residual(target, labels, logits, settings, *, seed, bins=CALIBRATION_BINS):
    """residual_test on a design-point batch, split by design point.

    target: StandardBatch (θ, φ, mask); labels, logits: [B, N]. Per-event arrays come back
    as [B, N] with padding NaN (``residual_logits``) or -1 (``residual_fold``), plus the
    ``correction_bins`` arrays (``bins`` bins per column).
    """
    mask = target.event_mask  # [B, N]
    names, values = feature_columns(target)  # [B, N, F]
    rows = np.nonzero(mask)[0]  # [E] design point of each valid event
    metrics, arrays = residual_test(
        values[mask], labels[mask], logits[mask], rows, settings, seed=seed
    )
    if "residual_logits" in arrays:
        arrays.update(
            correction_bins(
                names, values[mask], labels[mask], logits[mask], arrays["residual_logits"], bins
            )
        )
    for key, fill in (("residual_logits", np.nan), ("residual_fold", -1)):
        if key in arrays:
            full = np.full(mask.shape, fill, dtype=float if key == "residual_logits" else int)
            full[mask] = arrays[key]
            arrays[key] = full  # [B, N]
    return metrics, arrays


def stream_residual(features, labels, logits, settings, *, seed, bins=CALIBRATION_BINS):
    """residual_test on independent stream records, split by event.

    With ``settings.max_positives``, a seeded uniform fraction min(1, max_positives /
    positives) of all events is kept: the same fraction for hits and non-hits, so the gain
    needs no weights. features: [M, W]; labels, logits: [M]. Per-event arrays come back as
    [M] with NaN (``residual_logits``) or -1 (``residual_fold``) for events not kept; the
    ``correction_bins`` arrays (columns ``record_j``) cover the kept events.
    """
    labels = np.asarray(labels)
    positives = int(labels.sum())
    fraction = 1.0
    if settings.max_positives is not None and positives > settings.max_positives:
        fraction = settings.max_positives / positives
    keep = (
        _rng(seed, _SUBSAMPLE_KEY).random(len(labels)) < fraction
        if fraction < 1
        else np.ones(len(labels), dtype=bool)
    )  # [M]
    metrics, arrays = residual_test(
        np.asarray(features)[keep],
        labels[keep],
        np.asarray(logits)[keep],
        None,
        settings,
        seed=seed,
    )
    if "residual_logits" in arrays:
        kept = np.asarray(features, dtype=np.float64)[keep]  # [E, W]
        names = [f"record_{j}" for j in range(kept.shape[1])]
        arrays.update(
            correction_bins(
                names, kept, labels[keep], np.asarray(logits)[keep], arrays["residual_logits"], bins
            )
        )
    metrics.update(
        events_available=int(len(labels)),
        positives_available=positives,
        subsample_fraction=fraction,
    )
    for key, fill in (("residual_logits", np.nan), ("residual_fold", -1)):
        if key in arrays:
            full = np.full(len(labels), fill, dtype=np.float32 if key == "residual_logits" else int)
            full[keep] = arrays[key]
            arrays[key] = full  # [M]
    return metrics, arrays


def residual_test(features, labels, logits, groups, settings, *, seed):
    """Cross-fitted residual learnability test on valid events.

    features: [E, F] model inputs; labels, logits: [E]; groups: [E] design-point index in
    0..B-1, or None for independent events (split and bootstrapped by event).
    settings: ResidualTestConfig. Returns (metrics, arrays): ``residual_logits`` [E] (logit
    p'), ``residual_fold`` [E], and for groups ``residual_gain_per_point`` and
    ``residual_phi_gain_per_point`` [B] (nats, not per positive).
    """
    features = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels).astype(np.float64)
    z = np.asarray(logits, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(y) or len(z) != len(y):
        raise ValueError("features must be [E, F] with one row per label and logit")
    if not (np.isfinite(features).all() and np.isfinite(z).all()):
        raise ValueError("Residual test inputs and logits must be finite")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Residual test needs binary labels")
    grouped = groups is not None
    unit = np.asarray(groups, dtype=np.int64) if grouped else np.arange(len(y))
    n_units = int(unit.max()) + 1 if len(unit) else 0
    unit_positives = np.bincount(unit, y, minlength=n_units)
    positives = int(y.sum())
    metrics = dict(
        gain=None,
        gain_ci68=None,
        gain_ci95=None,
        phi_gain=None,
        phi_gain_ci68=None,
        phi_gain_ci95=None,
        rate_gain=None,
        rate_gain_ci68=None,
        rate_gain_ci95=None,
        events=int(len(y)),
        positives=positives,
        units="design_points" if grouped else "events",
        folds=None,
        trees=None,
        skipped=None,
    )
    present = np.flatnonzero(np.bincount(unit, minlength=n_units) > 0)
    if positives < settings.min_positives or len(present) < 2:
        metrics["skipped"] = (
            f"{positives} positives in {len(present)} units; needs >= "
            f"{settings.min_positives} positives and >= 2 units"
        )
        return metrics, {}
    k = min(settings.folds, len(present))
    fold_of_unit = np.full(n_units, -1)
    fold_of_unit[present] = balanced_folds(unit_positives[present], k, _rng(seed, _FOLD_KEY))
    fold = fold_of_unit[unit]  # [E]
    x = np.column_stack([features, z])  # [E, F+1]
    corrected = z.copy()
    trees = []
    tree_rng = _rng(seed, _TREE_KEY)
    for f in range(k):
        held = fold == f
        train_units = present[fold_of_unit[present] != f]
        booster, n_trees = _fit_fold(x, y, z, train_units, unit, unit_positives, settings, tree_rng)
        trees.append(n_trees)
        if booster is not None:
            corrected[held] = z[held] + booster.predict(x[held], raw_score=True)
    gain_event = _bce(y, z) - _bce(y, corrected)  # [E]
    if grouped:
        total_b = np.bincount(unit, gain_event, minlength=n_units)  # [B]
        phi_b = within_gain(unit, y, corrected) - within_gain(unit, y, z)  # [B]
        sample_units = present
    else:
        blocks = min(BOOTSTRAP_BLOCKS, len(y))
        block = _rng(seed, _BLOCK_KEY).permutation(len(y)) % blocks  # [E]
        total_b = np.bincount(block, gain_event, minlength=blocks)
        phi_b = None
        sample_units = np.arange(blocks)
        unit_positives = np.bincount(block, y, minlength=blocks)  # per block from here
    rng = _rng(seed, _BOOTSTRAP_KEY)
    samples = {"gain": [], "phi_gain": [], "rate_gain": []}
    for _ in range(BOOTSTRAP_DRAWS):
        index = rng.choice(sample_units, len(sample_units))
        k_draw = unit_positives[index].sum()
        if not k_draw:
            continue
        samples["gain"].append(total_b[index].sum() / k_draw)
        if phi_b is not None:
            samples["phi_gain"].append(phi_b[index].sum() / k_draw)
            samples["rate_gain"].append((total_b[index] - phi_b[index]).sum() / k_draw)
    metrics.update(
        gain=float(total_b.sum() / positives),
        **bootstrap_bands("gain", samples["gain"]),
        folds=k,
        trees=trees,
    )
    arrays = dict(residual_logits=corrected, residual_fold=fold)
    if phi_b is not None:
        metrics.update(
            phi_gain=float(phi_b.sum() / positives),
            rate_gain=float((total_b - phi_b).sum() / positives),
            **bootstrap_bands("phi_gain", samples["phi_gain"]),
            **bootstrap_bands("rate_gain", samples["rate_gain"]),
        )
        arrays.update(residual_gain_per_point=total_b, residual_phi_gain_per_point=phi_b)
    return metrics, arrays
