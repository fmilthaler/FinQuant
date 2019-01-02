import numpy as np


def weightedMean(means, weights):
    return np.sum(means * weights)


def weightedStd(cov_matrix, weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def SharpeRatio(exproi, riskfreerate, volatility):
    return (exproi - riskfreerate)/float(volatility)
