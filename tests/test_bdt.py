import numpy as np
import pytest
from schemas.bdt import BDTConfig
from schemas.data_models import StandardBatch,InputMode
from core.surrogate_bdt import event_features,build_bdt,score_bdt,save_bdt,load_bdt


def test_features_exclude_labels_and_preserve_voxel_event_order():
    theta=np.array([[1.,2.],[3.,4.]])
    phi=np.arange(12,dtype=float).reshape(2,3,2)
    b=StandardBatch(mode=InputMode.FULL,theta=theta,phi=phi,labels=np.zeros((2,3),int))
    expected=np.concatenate([np.repeat(theta,3,axis=0),phi.reshape(-1,2)],axis=1)
    np.testing.assert_array_equal(event_features(b),expected)
    b.labels[:]=1
    np.testing.assert_array_equal(event_features(b),expected)


def test_fit_and_roundtrip(tmp_path):
    pytest.importorskip('sklearn')
    rng=np.random.default_rng(42)
    phi=rng.normal(size=(4,100,2));labels=(phi[...,0]>.5).astype(int)
    b=StandardBatch(mode=InputMode.EVENT_ONLY,theta=None,phi=phi,labels=labels)
    cfg=BDTConfig(max_iter=40,eval_every=40,min_samples_leaf=5,l2_regularization=0)
    model=build_bdt(cfg);model.fit(event_features(b),labels.ravel())
    metrics,arrays=score_bdt(model,b)
    assert metrics['average_precision']>.95
    assert arrays['predicted'].shape==(4,)
    path=tmp_path/'model.joblib'
    save_bdt(path,model,{'feature_order':['phi0','phi1']})
    loaded,payload=load_bdt(path)
    np.testing.assert_array_equal(loaded.predict_proba(event_features(b)),model.predict_proba(event_features(b)))
    assert payload['metadata']['feature_order']==['phi0','phi1']
