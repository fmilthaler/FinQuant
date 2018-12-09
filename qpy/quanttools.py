import numpy as np

def weightedMean(means, weights):
    return np.sum(means * weights)

def weightedStd(cov_matrix, weights):
    weighted_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return weighted_std

def SharpeRatio(exproi, riskfreerate, volatility):
    sharpe = (exproi - riskfreerate)/float(volatility)
    return sharpe
