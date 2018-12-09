import pandas as pd
#from qpy.fund import Fund

class Portfolio(object):
    ''' Object that contains information about a investment portfolio.
    To initialise the object, it requires a name, reference year.
    To fill the portfolio with investment information and daily return of investments
    (ROI) data, the function addFund(fund) should be used, in which `fund` is a `Fund`
    object. a pandas.DataFrame
    of the portfolio investment information. The corresponding daily return of investments
    (ROI) are stored in the Fund object.
    '''
    def __init__(self, name, ref_year):
        # initilisating instance variables
        self.name = name
        self.ref_year = ref_year
        # initialise some more instance variables that do not have a value yet
        self.portfolio = pd.DataFrame()
        self.funds = {}
        self.pf_roi_data = pd.DataFrame()
        self.pf_means = None
        self.pf_weights = None
        self.expectedRoi = None
        self.volatility = None
        self.covPf = None

    def addFund(self, fund):
        # adding fund to dictionary containing all funds provided
        self.funds.update({fund.name : fund})
        # adding information of fund to the portfolio
        self.portfolio = self.portfolio.append(fund.getInvestmentInfo(), ignore_index=True)
        # also add ROI data of fund to the dataframe containing all roi data points
        self.__addRoiData(fund.name, fund.roi_data.ROI)

    def __addRoiData(self, name, df):
        # get length of columns in pf_roi_data, in order to create a new column
        cols = len(self.pf_roi_data.columns)
        # add roi data to overall dataframe of roi data:
        self.pf_roi_data.insert(loc=cols, column=name, value=df.values)#, inplace=True)#, ignore_index=True)

    # get functions:
    def getPortfolio(self):
        return self.portfolio

    def getRefYear(self):
        return self.ref_year

    def getPfRoiData(self):
        return self.pf_roi_data

    def getFund(self, name):
        return self.getFunds()[name]

    def getFunds(self):
        return self.funds

    def getPfMeans(self):
        return self.pf_means

    def getPfWeights(self):
        return self.pf_weights

    def getPfExpectedRoi(self):
        return self.expectedRoi

    def getVolatility(self):
        return self.volatility

    def getCovPf(self):
        return self.covPf

    # set functions:
    def setPfMeans(self, pf_means):
        self.pf_means = pf_means

    def setPfWeights(self, pf_weights):
        self.pf_weights = pf_weights

    def setPfExpectedRoi(self, expectedRoi):
        self.expectedRoi = expectedRoi

    def setVolatility(self, volatility):
        self.volatility = volatility

    def setCovPf(self, covPf):
        self.covPf = covPf

    # functions to compute quantities
    def compPfMeans(self):
        pf_means = self.getPfRoiData().mean().values
        # set instance variable
        self.setPfMeans(pf_means)
        return pf_means

    def compPfWeights(self):
        import numpy as np
        # computes the weights of the funds in the given portfolio
        # in respect of the total investment
        total = self.portfolio.FMV.sum()
        pf_weights = self.portfolio.FMV/total
        # set instance variable
        self.setPfWeights(pf_weights)
        return pf_weights

    def compPfExpectedRoi(self):
        # computing portfolio ROI
        pf_means = self.compPfMeans()
        pf_weights = self.compPfWeights()
        expectedRoi = weightedMean(pf_means, pf_weights)
        # set instance variable
        self.setPfExpectedRoi(expectedRoi)
        return expectedRoi

    def compPfVolatility(self):
        # computing the volatility of a portfolio
        pf_weights = self.compPfWeights()
        covPf = self.compCovPf()
        volatility = weightedStd(covPf, pf_weights)
        # set instance variable
        self.setVolatility(volatility)
        return volatility

    def compCovPf(self):
        # get the covariance matrix of the roi of the portfolio
        covPf = self.pf_roi_data.cov()
        # set instance variable
        self.setCovPf(covPf)
        return covPf

    def compSharpe(self, riskfreerate):
        expected_roi = self.getPfExpectedRoi()
        volatility = self.getVolatility()
        return SharpeRatio(expected_roi, riskfreerate, volatility)

    def compSkew(self):
        return self.getPfRoiData().skew()

    def compKurtosis(self):
        return self.getPfRoiData().kurt()

    # optimising the investments based on the sharpe ratio
    def optimisePortfolio(self, num_trials, riskfreerate=0, plot=True):
        # optimise the portfolio by doing a monte carlo simulation:
        # trying num_trials different weights of the investment in the portfolio
        # return values are:
        # (portfolio with highest sharpe ratio, portfolio with minimum volatility)
        # both are returned as a pandas.Series
        import numpy as np
        if (plot): import matplotlib.pyplot as plt
        # set number of funds in the portfolio
        num_funds = len(self.getFunds())
        #set up array to hold results
        res_columns = ['roi','volatility','sharpe']
        res_columns.extend(self.getFunds().keys())
        results = np.zeros((len(res_columns),num_trials))
        # compute means and covariance matrix
        pf_means = self.compPfMeans()
        cov_matrix = self.compCovPf()
        # monte carlo simulation
        for i in range(num_trials):
            # select random weights for portfolio
            weights = np.array(np.random.random(num_funds))
            # rebalance weights
            weights = weights/np.sum(weights)
            # compute portfolio roi and volatility
            #pf_roi = np.sum(pf_means * weights)
            pf_roi = weightedMean(pf_means, weights)
            #pf_volatility = np.sqrt(np.dot(weights.T,np.dot(cov_matrix, weights)))
            pf_volatility = weightedStd(cov_matrix, weights)

            #store results in results array
            results[0,i] = pf_roi
            results[1,i] = pf_volatility
            # store Sharpe Ratio
            results[2,i] = SharpeRatio(pf_roi, riskfreerate, pf_volatility)
            # add weights to results array
            results[3:, i] = weights

        # convert to pandas.DataFrame
        df_results = pd.DataFrame(results.T,columns=res_columns)
        # get portfolio with highest Sharpe Ratio
        pf_max_sharpe = df_results.iloc[df_results['sharpe'].idxmax()]
        # get portfolio with minimum volatility
        pf_min_volatility = df_results.iloc[df_results['volatility'].idxmin()]

        # plot results
        if (plot):
            # create scatter plot coloured by Sharpe Ratio
            plt.scatter(df_results.volatility, df_results.roi, c=df_results.sharpe, cmap='RdYlBu', label=None)
            plt.title('Monte Carlo simulation to optimise the investments')
            plt.xlabel('Volatility')
            plt.ylabel('ROI')
            plt.colorbar()
            # mark in red the highest sharpe ratio
            plt.scatter(pf_max_sharpe[1], pf_max_sharpe[0], marker='o', color='r', s=100, label='max Sharpe Ratio')
            # mark in green the minimum volatility
            plt.scatter(pf_min_volatility[1], pf_min_volatility[0], marker="o", color='g', s=100, label='min volatility')
            plt.legend()

        return (pf_max_sharpe, pf_min_volatility)

    def __str__(self):
        return str(self.getPortfolio())

def weightedMean(means, weights):
    import numpy as np
    return np.sum(means * weights)

def weightedStd(cov_matrix, weights):
    import numpy as np
    weighted_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return weighted_std

def SharpeRatio(exproi, riskfreerate, volatility):
    sharpe = (exproi - riskfreerate)/float(volatility)
    return sharpe