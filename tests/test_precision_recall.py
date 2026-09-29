import numpy as np
import pytest
from core.precision_recall import precision_recall


def test_ties_and_average_precision():
    r=precision_recall([1,0,1,0],[3,2,2,1])
    np.testing.assert_allclose(r['precision'],[1,1,2/3,1/2])
    np.testing.assert_allclose(r['recall'],[0,.5,1,1])
    assert r['average_precision']==pytest.approx(5/6)


def test_constant_score_has_prevalence_ap():
    r=precision_recall([0,1,0,0],[-100]*4)
    assert r['average_precision']==.25
    np.testing.assert_allclose(r['recall'],[0,1])


def test_perfect_ranking():
    assert precision_recall([0,1,1,0],[0,2,3,1])['average_precision']==1
