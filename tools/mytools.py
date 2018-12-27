# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

def plotData(x,y,label=None,xlabel=None,ylabel=None,
             title=None,fmt='-'):
    import numpy as np
    import pylab
    x = np.array(x)
    y = np.array(y)
    ax = pylab.plot(x, y, fmt, label=label)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)
    pylab.title(title)
    if (label):
        pylab.legend()
    return ax

# <codecell>

# provide functions to get deal reference date
# from the portfolio datafile, and furthermore get 
# the year from that extracted date.

def getReferenceDate(filename):
    with open(filename, "r") as f:
        for line in f:
            if ("Deal Reference Date" in line):
                refline = line.replace(',','').strip()
                break
        refDate = refline.split('Deal Reference Date')
        refDate = ''.join(refline.split('Deal Reference Date'))
    return refDate

def getReferenceYear(refDate):
    if ('/' in refDate):
        splitchar = '/'
    elif ('.' in refDate):
        splitchar = '.'
    elif ('-' in refDate):
        splitchar = '-'
    else:
        print("Could not correctly identify reference year. "
              +"2017 will be used as the reference year.")
        refYear = 2017
        return refYear
    refYear = refDate.split(splitchar)[-1]
    if (len(refYear)<4):
        refYear = '20'+refYear
    try:
        refYear = int(refYear)
    except:
        raise ValueError("Could not convert the found reference date '"
                         +str(refDate)+"' to an integer (year).")
    return refYear

# <codecell>

# adding functions to manipulate bar plots
# add text to bars
def addValToBarPlot(ax, values, xshift=.15, yshift=-.12,
                     color='black', precision=3):
    for i in range(len(ax.patches)):
        bar = ax.patches[i]
        # get_x pulls left or right; get_height pushes up or down
        ax.text(bar.get_x()+xshift, bar.get_height()+yshift,
                str(round((values[i]), precision)),
                color='black')

# change width of existing bar (in bar plot)
def changeBarWidth(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value
        # we change the bar width
        patch.set_width(new_value)
        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

# <codecell>

# get random sample from dataset
def getSample(data, sampleSize, replace=False):
    import pandas as pd
    import numpy as np
    if (data.__class__.__name__ in ["DataFrame","Series"]):
        sample = data.sample(n=sampleSize)
    elif (data.__class__.__name__ in ["ndarray", "list"]):
        sample = np.random.choice(data, sampleSize, replace=replace)
    else:
        raise ValueError("data must be either pandas DataFrame/Series or a numpy array or list.")
    return sample

# define a few functions for statistics
def compMean(population):
    return sum(population)/len(population)

def compStd(population):
    import numpy as np
    return np.std(population)

def compMedian(population):
    import numpy as np
    return np.median(population)

def sem(popSD, sampleSize):
    return popSD/sampleSize**0.5

# function for monte carlo simulation
def monteCarlo(population, sampleSizes, numTrials, 
               bootstrap=False, returnTrialVals=False):
    # loop over different sample sizes, and number of trials,
    # extract a random sample from the given population,
    # compute sample mean and sample std for each trial and sample size.
    if (bootstrap):
        replace=True
        if (len(sampleSizes) == 1 and sampleSizes[0] != len(population)):
            raise ValueError("Please set sampleSizes to be '[len(population)]'")
    else: replace=False
    sampleMeans, sampleStds, sampleMedians = [], [], []
    for sampleSize in sampleSizes:
        trialMeans, trialStds, trialMedians = [], [], []
        for trial in range(numTrials):
            # get a random sample from population
            sample = getSample(population, sampleSize, replace)
            # compute mean:
            sampleMean = compMean(sample)
            sampleStd = compStd(sample)
            sampleMedian = compMedian(sample)
            trialMeans.append(sampleMean)
            trialStds.append(sampleStd)
            trialMedians.append(sampleMedian)
        sampleMeans.append(compMean(trialMeans))
        sampleStds.append(compStd(trialMeans))
        sampleMedians.append(compMedian(trialMedians))
    if (returnTrialVals):
        return sampleMeans, sampleStds, sampleMedians, trialMeans, trialStds, trialMedians
    else:
        return sampleMeans, sampleStds, sampleMedians
