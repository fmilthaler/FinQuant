import numpy as np
from finquant.quants import weighted_mean, weighted_std
from finquant.quants import sharpe_ratio, annualised_portfolio_quantities


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


def test_annualised_portfolio_quantities():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
    Sigma = np.cov(x, y)
    weights = np.array([1, 1])
    mean = np.array([1, 2])
    weights = np.array([-3, 5])
    res = annualised_portfolio_quantities(weights, mean, Sigma, 0, 252)
    orig = (1764, 347.79304190854657, 5.071981861166303)
    for i in range(len(res)):
        assert abs(res[i] - orig[i]) <= 1e-15
