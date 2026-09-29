import pytest
import torch
from core.bernoulli_cnp import evaluate_bernoulli
from core.bernoulli_transformer import (BernoulliTransformer, save_transformer_checkpoint, load_transformer_checkpoint)
from core.surrogate_cnp import split_context_target
from core.binary_losses import binary_focal_loss_with_logits
from data.pseudo_generator import for_scenario
from schemas.tabular_transformer import TabularTransformerConfig

@pytest.mark.parametrize('architecture', ['legacy', 'ft_transformer'])
@pytest.mark.parametrize('scenario',[f'S{i}' for i in range(1,9)])
def test_single_output_and_checkpoint(scenario,tmp_path,architecture):
    source=for_scenario(scenario,seed=0)
    batch=source.generate(n_trials=3,n_events=24)
    context,target=split_context_target(batch,8,seed=2)
    config=TabularTransformerConfig(architecture=architecture, token_dim=8, n_heads=2, feedforward_dim=12)
    model=BernoulliTransformer(config,source.dim_theta,source.dim_phi)
    logits=model(context,target)
    assert logits.shape==(3,16)
    assert model.head[-1].out_features==1
    assert not any('sigma' in name for name,_ in model.named_parameters())
    altered = target.model_copy(update={'labels': 1-target.labels})
    torch.testing.assert_close(model(None, altered), logits)
    loss=binary_focal_loss_with_logits(logits,target.labels,gamma=1)
    loss.backward()
    assert torch.isfinite(model.head[-1].weight.grad).all()
    summary,arrays=evaluate_bernoulli(model,context,target)
    assert model.training
    assert arrays['predicted'].shape==(3,)
    assert summary['bernoulli_log_loss']>=0
    path=tmp_path/'cnp.ckpt'
    save_transformer_checkpoint(path,model,encoder_config=config,dim_theta=source.dim_theta,
                             dim_phi=source.dim_phi,metadata={'step':2})
    loaded,payload=load_transformer_checkpoint(path)
    torch.testing.assert_close(loaded(context,target),logits)
    assert payload['metadata']['step']==2


def test_attention_is_within_events_and_inference_chunks_agree():
    import numpy as np
    source = for_scenario('S1', seed=4)
    batch = source.generate(n_trials=2, n_events=12)
    config = TabularTransformerConfig(token_dim=8, n_heads=2, feedforward_dim=12,
                                       inference_chunk_size=3)
    model = BernoulliTransformer(config, source.dim_theta, source.dim_phi).eval()
    with torch.no_grad():
        expected = model(None, batch)
        model.config.inference_chunk_size = 100
        torch.testing.assert_close(model(None, batch), expected, atol=1e-6, rtol=1e-5)
        phi = batch.phi.copy()
        phi[:, 0] += .2
        changed = batch.model_copy(update={'phi': phi})
        actual = model(None, changed)
        torch.testing.assert_close(actual[:, 1:], expected[:, 1:], atol=1e-6, rtol=1e-5)
        assert not np.allclose(actual[:, 0].numpy(), expected[:, 0].numpy())


def test_ft_last_cls_matches_full_attention_and_gradients():
    from core.bernoulli_transformer import FeatureTransformerBlock
    cfg = TabularTransformerConfig(architecture='ft_transformer', token_dim=8,
                                  n_heads=2, feedforward_dim=12)
    block = FeatureTransformerBlock(cfg, first=False).eval()
    tokens = torch.randn(3, 10, 8, requires_grad=True)
    full = block(tokens)[:, :1]
    efficient = block(tokens, cls_only=True)
    torch.testing.assert_close(full, efficient)
    full_grad = torch.autograd.grad(full.sum(), tokens, retain_graph=True)[0]
    efficient_grad = torch.autograd.grad(efficient.sum(), tokens)[0]
    torch.testing.assert_close(full_grad, efficient_grad)


def test_matched_ft_capacity_and_first_norm():
    import yaml
    from pathlib import Path
    cfg = TabularTransformerConfig(**yaml.safe_load(
        (Path(__file__).parents[1]/'config.tabular_transformer.matched.yaml').read_text()))
    model = BernoulliTransformer(cfg, 3, 6)
    assert abs(sum(p.numel() for p in model.parameters())/125633-1) < .05
    assert isinstance(model.blocks[0].attention_norm, torch.nn.Identity)
    assert isinstance(model.blocks[1].attention_norm, torch.nn.LayerNorm)
