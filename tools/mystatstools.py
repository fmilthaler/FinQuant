import pylab

# define a few functions for statistics
def compMean(population):
    return sum(population)/len(population)

def compVariance(population):
    mean = float(sum(population))/len(population)
    diffs = 0.0
    for x in population:
        diffs += (x - mean)**2
    return diffs/len(population)

def compStd(population):
    import numpy as np
    return np.std(population)

def compMedian(population):
    import numpy as np
    return np.median(population)

def compSem(popSD, sampleSize):
    return popSD/sampleSize**0.5

# Compare models
def aveMeanSquareError(data, predicted):
    error = 0.0
    for i in range(len(data)):
        error = error + (data[i] - predicted[i])**2
    return error/len(data)

def L2residual(observed, predicted):
    residual = ((observed - predicted)**2).sum()
    return residual

def rSquared(observed, predicted):
    import numpy as np
    observed = np.array(observed)
    predicted = np.array(predicted)
    error = ((observed - predicted)**2).sum()
    meanError = error/len(observed)
    R2 = 1 - (meanError/np.var(observed))
    return R2

def gaussian(x, mu, sigma):
    factor1 = 1.0/(sigma*(2*pylab.pi)**0.5)
    factor2 = pylab.e**(-0.5*((x-mu)/sigma)**2)
    return factor1*factor2
