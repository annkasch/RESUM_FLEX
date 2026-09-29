import numpy as np
from core.mixup import ClassAwareMixupSource
from schemas.data_models import StandardBatch, InputMode


def test_disjoint_pools_epoch_exhaustion_and_pairing():
    y=np.zeros((3,25),dtype=int)
    y[0,-4:]=1;y[1,-1]=1
    b=StandardBatch(mode=InputMode.FULL,theta=np.arange(3)[:,None],phi=y[...,None].astype(float),labels=y)
    def make():return ClassAwareMixupSource(b,alpha=.2,seed=42,batch_size=2,n_events=8,n_context=3)
    source=make();same=make()
    for pools in source.pools:
        assert not set(np.concatenate(pools[0])) & set(np.concatenate(pools[1]))
    for epoch in (1,2):
        seen=[]
        while True:
            ctx,tgt=source.next();ctx2,tgt2=same.next()
            for a,other in [(ctx,ctx2),(tgt,tgt2)]:
                np.testing.assert_allclose(a.labels,a.phi[...,0])
                np.testing.assert_array_equal(a.labels,other.labels)
            for r in source.last_provenance:
                assert r['epoch']==epoch
                seen.extend((r['voxel'],int(i)) for i in r['negatives'])
                if r['voxel']==1 and r['side']==0:
                    assert np.all(b.labels[r['voxel'],r['partners']]==0)
            if not source.pending:break
        expected={(i,int(j)) for i in range(3) for j in np.flatnonzero(y[i]==0)}
        assert len(seen)==len(set(seen)) and set(seen)==expected


def test_real_context_preserves_events_and_matched_targets():
    y = np.zeros((3, 100), dtype=int)
    y[0, -10:] = 1
    y[1, -1] = 1
    phi = np.broadcast_to(np.arange(100)[None, :, None], (3, 100, 1)).astype(float).copy()
    batch = StandardBatch(mode=InputMode.FULL, theta=np.arange(3)[:, None], phi=phi, labels=y)
    options = dict(alpha=.2, seed=42, batch_size=2, n_events=16, n_context=6)
    mixed = ClassAwareMixupSource(batch, **options)
    real = ClassAwareMixupSource(batch, mix_context=False, **options)
    for _ in range(30):
        _, mixed_target = mixed.next()
        context, target = real.next()
        np.testing.assert_array_equal(target.labels, mixed_target.labels)
        np.testing.assert_array_equal(target.phi, mixed_target.phi)
        np.testing.assert_array_equal(target.theta, mixed_target.theta)
        assert type(context) is StandardBatch
        for j, theta in enumerate(context.theta):
            voxel = int(theta[0])
            indices = context.phi[j, :, 0].astype(int)
            assert len(set(indices)) == len(indices)
            assert set(indices) <= set(np.concatenate(real.pools[voxel][0]))
            assert not set(indices) & set(np.concatenate(real.pools[voxel][1]))
            np.testing.assert_array_equal(context.labels[j], y[voxel, indices])
