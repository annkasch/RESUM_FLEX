import numpy as np
import pytest

from core.surrogates.models import build_surrogate
from core.surrogates.stages import fit_stages
from data.pseudo_generator import for_scenario
from schemas.surrogates import SurrogateRunConfig, TrainingStage
from tests.test_fine_tuning import settings
from tests.test_surrogates import spec


def test_legacy_schedule_and_explicit_stages_are_identical(tmp_path):
    from core.surrogate_cnp import split_context_target
    from core.surrogates.base import Episode
    from core.surrogates.fine_tuning import fine_tuning_stages

    source = for_scenario("S1", seed=4)
    batch = source.generate(n_trials=4, n_events=24)
    ctx, target = split_context_target(batch, 4, seed=7)
    a = build_surrogate(spec("legacy_cnp"), source.dim_theta, source.dim_phi, seed=13)
    b = build_surrogate(spec("legacy_cnp"), source.dim_theta, source.dim_phi, seed=13)
    cfg = settings("legacy_cnp")
    a.fit(batch, cfg, validation=Episode(ctx, target), checkpoints=tmp_path / "old")
    fit_stages(
        b,
        batch,
        fine_tuning_stages(cfg, "legacy_cnp"),
        validation=Episode(ctx, target),
        checkpoints=tmp_path / "new",
    )
    np.testing.assert_array_equal(
        a.predict(target, context=ctx).logits, b.predict(target, context=ctx).logits
    )


def test_invalid_stage_dependencies_and_tree_freezing(tmp_path):
    with pytest.raises(ValueError, match="BDT"):
        TrainingStage(name="fine", training={"backend": "bdt"}, trainable="output_layer")
    cfg = settings("cnp").model_copy(update={"fine_tuning": None})
    with pytest.raises(ValueError, match="earlier"):
        SurrogateRunConfig(
            model=spec("cnp"),
            training=cfg,
            stages=[TrainingStage(name="first", start_from="missing/best", training=cfg)],
            data_directory=tmp_path,
            output_directory=tmp_path / "out",
        )
