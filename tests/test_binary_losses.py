import pytest
import torch
from torch.nn import functional as F
from core.binary_losses import binary_focal_loss_with_logits
from core.surrogate_cnp import CnpOutput, resum_binary_logits, resum_binary_moments, theory_truth_loss

@pytest.mark.parametrize('gamma',[0.,2.])
def test_extreme_wrong_predictions_retain_gradient(gamma):
    logits=torch.tensor([-100.,100.],requires_grad=True)
    loss=binary_focal_loss_with_logits(logits,torch.tensor([1.,0.]),gamma=gamma)
    loss.backward()
    assert torch.isfinite(loss)
    torch.testing.assert_close(logits.grad,torch.tensor([-.5,.5]))

def test_gamma_zero_matches_weighted_bce():
    z=torch.tensor([-3.,1.,4.],dtype=torch.float64)
    y=torch.tensor([0.,1.,.3],dtype=torch.float64)
    w=torch.tensor([1.,.1,2.],dtype=torch.float64)
    torch.testing.assert_close(binary_focal_loss_with_logits(z,y,gamma=0,weights=w),
                               (F.binary_cross_entropy_with_logits(z,y,reduction='none')*w).mean())

def test_focal_downweights_easy_negatives():
    z=torch.tensor([-8.]);y=torch.zeros_like(z)
    assert binary_focal_loss_with_logits(z,y,gamma=2)<binary_focal_loss_with_logits(z,y,gamma=0)*1e-5

def test_effective_logit_preserves_mean_and_theory_loss_recovers():
    out=CnpOutput(mu_logit=torch.tensor([[-100.,1.]],requires_grad=True),log_sigma=torch.tensor([[0.,2.]]))
    mean,_=resum_binary_moments(out)
    torch.testing.assert_close(resum_binary_logits(out).sigmoid(),mean)
    theory_truth_loss(out,torch.ones_like(mean)).backward()
    assert out.mu_logit.grad[0,0]<-.1
