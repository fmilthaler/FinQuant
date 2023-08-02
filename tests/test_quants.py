import pdb

import numpy as np
import pandas as pd
import pytest

from finquant.quants import (
    annualised_portfolio_quantities,
    downside_risk,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
    weighted_mean,
    weighted_std,
)


def test_weighted_mean():
    means = np.array([1])
    weights = np.array([1])
    assert weighted_mean(means, weights) == 1
    means = np.array(range(5))
    weights = np.array(range(5, 10))
    assert weighted_mean(means, weights) == 80


def test_weighted_std():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
    Sigma = np.cov(x, y)
    weights = np.array([1, 1])
    assert weighted_std(Sigma, weights) == 0.0
    weights = np.array([-3, 5])
    assert weighted_std(Sigma, weights) ** 2 == 480.0


def test_sharpe_ratio():
    assert sharpe_ratio(0.5, 0.2, 0.02) == 2.4
    assert sharpe_ratio(0.5, 0.22, 0.005) == 2.25


def test_sortino_ratio():
    assert sortino_ratio(0.5, 0.0, 0.02) is np.NaN
    assert sortino_ratio(0.005, 8.5, 0.005) == 0.0


def test_value_at_risk():
    assert abs(value_at_risk(1e2, 0.5, 0.25, 0.95) - 91.12) <= 1e-1
    assert abs(value_at_risk(1e3, 0.8, 0.5, 0.99) - 1963.17) <= 1e-1
    assert abs(value_at_risk(1e4, -0.1, 0.25, 0.9) - 2203.88) <= 1e-1
    assert abs(value_at_risk(1e4, 0.1, -0.25, 0.9) - (-2203.88)) <= 1e-1
    assert abs(value_at_risk(1e4, -0.1, -0.25, 0.9) - (-4203.88)) <= 1e-1
    assert value_at_risk(0, 0.1, 0.5, 0.9) == 0
    assert abs(value_at_risk(1e4, 0, 0.5, 0.9) - 6407.76) <= 1e-1
    assert abs(value_at_risk(1e4, 0.1, 0, 0.9) - 1000) <= 1e-1
    assert value_at_risk(1e4, 0, 0, 0.9) == 0


def test_value_at_risk_invalid_types():
    with pytest.raises(ValueError):
        value_at_risk("10000", 0.05, 0.02, 0.95)

    with pytest.raises(ValueError):
        value_at_risk(10000, 0.05, "0.02", 0.95)

    with pytest.raises(ValueError):
        value_at_risk(10000, [0.05], 0.02, 0.95)

    with pytest.raises(ValueError):
        value_at_risk(10000, 0.05, 0.02, "0.95")

    with pytest.raises(ValueError):
        value_at_risk(10000, 0.05, 0.02, 1.5)

    with pytest.raises(ValueError):
        value_at_risk(10000, 0.05, 0.02, -0.5)


def test_annualised_portfolio_quantities():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
    Sigma = np.cov(x, y)
    mean = np.array([1, 2])
    weights = np.array([-3, 5])
    res = annualised_portfolio_quantities(weights, mean, Sigma, 0, 252)
    orig = (1764.0, 347.79304190854657, 5.071981861166303)
    for i in range(len(res)):
        assert abs(res[i] - orig[i]) <= 1e-15


def test_downside_risk():
    data1 = pd.DataFrame({"1": [1, 2, 4, 8], "2": [1, 2, 3, 4]})
    weights = np.array([0.25, 0.75])
    rf_rate = 0.005
    dr1 = downside_risk(data1, weights, rf_rate)
    assert dr1 == 0

    data2 = pd.DataFrame({"1": [7, 6, 5, 4, 3]})
    weights = np.array([1])
    rf_rate = 0.0
    dr2 = downside_risk(data2, weights, rf_rate)
    assert abs(dr2 - 0.19409143531019335) <= 1e-15
