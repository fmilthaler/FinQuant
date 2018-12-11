import pandas as pd
from qpy.quanttools import weightedMean, weightedStd, SharpeRatio

class Stock(object):
    ''' Object that contains information about a stock/fund.
    To initialise the object, it requires a name, information about
    the stock/fund given as one of the following data structures:
     - pandas.Series
     - pandas.DataFrame
    The investment information can contain as little information as its name,
    and the amount invested in it, the column labels must be "Name" and "FMV"
    respectively, but it can also contain more information, such as
     - Year
     - Strategy
     - CCY
     - etc
    It also requires daily return of investments (ROI) as a pandas.DataFrame or
    pandas.Series. If it is a DataFrame, the "roi_data" data structure is required
    to contain the following label
     - ROI
    '''
    def __init__(self, investmentinfo, roi_data):
        self.name = investmentinfo.Name
        self.investmentinfo = investmentinfo
        self.roi_data = roi_data
    def getInvestmentInfo(self):
        return self.investmentinfo
    def getRoiData(self):
        return self.roi_data
    def compSkew(self):
        return self.roi_data.skew().values[0]
    def compKurtosis(self):
        return self.roi_data.kurt().values[0]
    def __str__(self):
        return str(pd.DataFrame([self.investmentinfo]))

class Portfolio(object):
    ''' Object that contains information about a investment portfolio.
    To initialise the object, it requires a name, reference year.
    To fill the portfolio with investment information and daily return of investments
    (ROI) data, the function addStock(stock) should be used, in which `stock` is a `Stock`
    object. a pandas.DataFrame
    of the portfolio investment information. The corresponding daily return of investments
    (ROI) are stored in the Stock object.
    '''
    def __init__(self, name, ref_year):
        # initilisating instance variables
        self.name = name
        self.ref_year = ref_year
        # initialise some more instance variables that do not have a value yet
        self.portfolio = pd.DataFrame()
        self.stocks = {}
        self.pf_roi_data = pd.DataFrame()
        self.pf_means = None
        self.pf_weights = None
        self.expectedRoi = None
        self.volatility = None
        self.covPf = None

    def addStock(self, stock):
        # adding stock to dictionary containing all stocks provided
        self.stocks.update({stock.name : stock})
        # adding information of stock to the portfolio
        self.portfolio = self.portfolio.append(stock.getInvestmentInfo(), ignore_index=True)
        # also add ROI data of stock to the dataframe containing all roi data points
        self.__addRoiData(stock.name, stock.roi_data.ROI)

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

    def getStock(self, name):
        return self.getStocks()[name]

    def getStocks(self):
        return self.stocks

    def getPfMeans(self):
        return self.pf_means

    def getTotalFMV(self):
        return self.portfolio.FMV.sum()

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
        # computes the weights of the stocks in the given portfolio
        # in respect of the total investment
        pf_weights = self.portfolio.FMV/self.getTotalFMV()
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
        # set number of stocks in the portfolio
        num_stocks = len(self.getStocks())
        #set up array to hold results
        res_columns = list(self.getStocks().keys())
        res_columns.extend(['ROI','Volatility','Sharpe'])
        results = np.zeros((len(res_columns),num_trials))
        # compute means and covariance matrix
        pf_means = self.compPfMeans()
        cov_matrix = self.compCovPf()
        # monte carlo simulation
        for i in range(num_trials):
            # select random weights for portfolio
            weights = np.array(np.random.random(num_stocks))
            # rebalance weights
            weights = weights/np.sum(weights)
            # compute portfolio roi and volatility
            pf_roi = weightedMean(pf_means, weights)
            pf_volatility = weightedStd(cov_matrix, weights)

            # add weights times total FMV to results array
            results[0:num_stocks, i] = weights*self.getTotalFMV()
            #store results in results array
            results[num_stocks,i] = pf_roi
            results[num_stocks+1,i] = pf_volatility
            # store Sharpe Ratio
            results[num_stocks+2,i] = SharpeRatio(pf_roi, riskfreerate, pf_volatility)

        # convert to pandas.DataFrame
        df_results = pd.DataFrame(results.T,columns=res_columns)
        # get portfolio with highest Sharpe Ratio
        pf_max_sharpe = df_results.iloc[df_results['Sharpe'].idxmax()]
        # get portfolio with minimum volatility
        pf_min_volatility = df_results.iloc[df_results['Volatility'].idxmin()]

        # plot results
        if (plot):
            # create scatter plot coloured by Sharpe Ratio
            plt.figure()
            plt.scatter(df_results['Volatility'], df_results['ROI'], c=df_results['Sharpe'], cmap='RdYlBu', label=None)
            plt.title('Monte Carlo simulation to optimise the investments')
            plt.xlabel('Volatility')
            plt.ylabel('ROI')
            plt.colorbar()
            # mark in red the highest sharpe ratio
            plt.scatter(pf_max_sharpe[num_stocks+1], pf_max_sharpe[num_stocks], marker='o', color='r', s=100, label='max Sharpe Ratio')
            # mark in green the minimum volatility
            plt.scatter(pf_min_volatility[num_stocks+1], pf_min_volatility[num_stocks], marker="o", color='g', s=100, label='min Volatility')
            plt.legend()
            plt.show()

        return (pf_max_sharpe, pf_min_volatility)

    def __str__(self):
        return str(self.getPortfolio())

