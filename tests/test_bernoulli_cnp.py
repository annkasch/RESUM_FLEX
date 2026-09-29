import pytest
import torch
from core.bernoulli_cnp import (BernoulliCNP,evaluate_bernoulli,
                               save_bernoulli_checkpoint,load_bernoulli_checkpoint)
from core.surrogate_cnp import split_context_target
from core.binary_losses import binary_focal_loss_with_logits
from data.pseudo_generator import for_scenario
from schemas.config import EncoderConfig

@pytest.mark.parametrize('scenario',[f'S{i}' for i in range(1,9)])
def test_single_output_and_checkpoint(scenario,tmp_path):
    source=for_scenario(scenario,seed=0)
    batch=source.generate(n_trials=3,n_events=24)
    context,target=split_context_target(batch,8,seed=2)
    config=EncoderConfig(type='mlp',latent_dim=8,hidden_dims=[12],dropout=0)
    model=BernoulliCNP(config,source.dim_theta,source.dim_phi)
    logits=model(context,target)
    assert logits.shape==(3,16)
    assert model.decoder.net[-1].out_features==1
    assert not any('sigma' in name for name,_ in model.named_parameters())
    loss=binary_focal_loss_with_logits(logits,target.labels,gamma=1)
    loss.backward()
    assert torch.isfinite(model.decoder.net[-1].weight.grad).all()
    summary,arrays=evaluate_bernoulli(model,context,target)
    assert model.training
    assert arrays['predicted'].shape==(3,)
    assert summary['bernoulli_log_loss']>=0
    path=tmp_path/'cnp.ckpt'
    save_bernoulli_checkpoint(path,model,encoder_config=config,dim_theta=source.dim_theta,
                             dim_phi=source.dim_phi,metadata={'step':2})
    loaded,payload=load_bernoulli_checkpoint(path)
    torch.testing.assert_close(loaded(context,target),logits)
    assert payload['metadata']['step']==2
