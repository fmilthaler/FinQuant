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
    def __init__(self, investmentinfo, roi_data=None, stock_data=None):
        self.name = investmentinfo.Name
        self.investmentinfo = investmentinfo
        self.roi_data = roi_data
        self.stock_data = stock_data
    def getInvestmentInfo(self):
        return self.investmentinfo
    def getStockData(self):
        return self.stock_data
    def getRoiData(self):
        if (self.roi_data is None):
            self.compRoiData()
        return self.roi_data
    def compRoiData(self):
        # self.roi_data = computation with self.stock_data here
        return self.roi_data
    def compSkew(self):
        return self.roi_data.skew().values[0]
    def compKurtosis(self):
        return self.roi_data.kurt().values[0]
    def __str__(self):
        return str(self.investmentinfo.to_frame().transpose())

class Portfolio(object):
    ''' Object that contains information about a investment portfolio.
    To initialise the object, it does not require any input.
    To fill the portfolio with investment information and daily return of investments
    (ROI) data, the function addStock(stock) should be used, in which `stock` is a `Stock`
    object, a pandas.DataFrame of the portfolio investment information. The corresponding
    daily return of investments (ROI) are stored in the Stock object.
    '''
    def __init__(self):
        # initilisating instance variables
        self.portfolio = pd.DataFrame()
        self.stocks = {}
        self.pf_stock_data = pd.DataFrame()
        self.pf_roi_data = pd.DataFrame()
        self.pf_means = None
        self.pf_weights = None
        self.expectedRoi = None
        self.volatility = None
        self.covPf = None

    def addStocks(self, stocks, data):
        #self.stocks

        for i in range(len(df_pf)):
            self.addStock(Stock(stocks.loc[i], roi))

        for i in range(len(df_pf)):
            #print(df_pf.loc[i])
            age = df_pf.loc[i].Age
            strategy = df_pf.loc[i].Strategy
            #data = extractRoiData()
            pf.addStock(Stock(df_pf.loc[i], extractRoiData(df_data, age, strategy)))

    def addStock(self, stock):
        # adding stock to dictionary containing all stocks provided
        self.stocks.update({stock.name : stock})
        # adding information of stock to the portfolio
        self.portfolio = self.portfolio.append(stock.getInvestmentInfo(), ignore_index=True)
        # also add ROI data of stock to the dataframe containing all roi data points
        if (not stock.stock_data is None):
            self._addStockData(stock.stock_data)

        # set roi_data, if given
        if (not stock.roi_data is None):
            self._addRoiData(stock.name, stock.roi_data.ROI)

    def _addStockData(self, df):
        # loop over columns in given dataframe
        for datacol in df.columns:
            cols = len(self.pf_stock_data.columns)
            self.pf_stock_data.insert(loc=cols,
                                      column=datacol,
                                      value=df[datacol].values
                                     )
        # set index correctly
        self.pf_stock_data.set_index(df.index.values, inplace=True)
        # set index name:
        self.pf_stock_data.index.rename('Date', inplace=True)

    def _addRoiData(self, name, df):
        # get length of columns in pf_roi_data, in order to create a new column
        cols = len(self.pf_roi_data.columns)
        # add roi data to overall dataframe of roi data:
        self.pf_roi_data.insert(loc=cols, column=name, value=df.values)
        # set index correctly
        self.pf_roi_data.set_index(df.index.values, inplace=True)
        # set index name:
        self.pf_roi_data.index.rename('Date', inplace=True)

    # get functions:
    def getPortfolio(self):
        return self.portfolio

    def getPfStockData(self):
        return self.pf_stock_data

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
        res_columns.extend(['ROI','Volatility','Sharpe Ratio'])
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

        # transpose and convert to pandas.DataFrame:
        df_results = pd.DataFrame(results.T,columns=res_columns)
        # adding info of max sharpe ratio and of min volatility
        # to resulting df (with meaningful indices):
        pf_opt = pd.DataFrame([df_results.iloc[df_results['Sharpe Ratio'].idxmax()],
                               df_results.iloc[df_results['Volatility'].idxmin()]],
                              index=['Max Sharpe Ratio', 'Min Volatility'])

        # plot results
        if (plot):
            plt.figure()
            # create scatter plot coloured by Sharpe Ratio
            ax = df_results.plot.scatter(x='Volatility', y='ROI', c='Sharpe Ratio', colormap='RdYlBu', label=None)
            # mark in red the highest sharpe ratio
            pf_opt.loc['Max Sharpe Ratio'].to_frame().T.plot.scatter(x='Volatility', y='ROI', marker="o", color='r', s=100, label='max Sharpe Ratio', ax=ax)
            # mark in green the minimum volatility
            pf_opt.loc['Min Volatility'].to_frame().T.plot.scatter(x='Volatility', y='ROI', marker="o", color='g', s=100, label='min Volatility', ax=ax)
            plt.title('Monte Carlo simulation to optimise the investments')
            plt.xlabel('Volatility')
            plt.ylabel('ROI')
            plt.legend()
            plt.show()

        return (pf_opt)

    def __str__(self):
        return str(self.getPortfolio())



def correctQuandlRequestStockName(names):
    # make sure names is a list of names:
    if (isinstance(names, str)):
        names = [names]
    reqnames = []
    # correct stock names if necessary:
    for name in names:
        if (not name.startswith('WIKI/')):
            name = 'WIKI/'+name
        reqnames.append(name)
    return reqnames

def getStockFromQuandl(name, start=None, end=None):
    try:
        import quandl
        import datetime
    except ImportError:
        print("The following packages are required:\n - quandl\n - datetime\nPlease ensure that they are installed.")
    reqname = correctQuandlRequestStockName(name)
    return quandl.get(reqname, start_date=start, end_date=end)

def getStocksFromQuandl(names, start=None, end=None):
    try:
        import quandl
        import datetime
    except ImportError:
        print("The following packages are required:\n - quandl\n - datetime\nPlease ensure that they are installed.")
    # get correct stock names that quandl get request
    reqnames = correctQuandlRequestStockName(names)
    # get stocks:
    stocks = quandl.get(reqnames, start_date=start, end_date=end)
    return stocks

def getQuandlDataColumnLabel(stock_name, data_label):
    return stock_name+' - '+data_label

def getStocksDataColumns(stocks, names, cols):
    # get correct stock names that quandl get request
    reqnames = correctQuandlRequestStockName(names)
    # get current column labels and replacement labels
    reqcolnames = []
    for name in reqnames:
        for col in cols:
            reqcolnames.append(getQuandlDataColumnLabel(name, col))
    stocks = stocks.loc[:, reqcolnames]
    # now rename the columns:
    newcolnames = {}
    for i in reqcolnames:
        newcolnames.update({i: i.replace('WIKI/','')})
    stocks.rename(columns=newcolnames, inplace=True)
    return stocks

def buildPortfolioFromQuandl(pf_information, names, start=None, end=None,
                             datacolumns=["Adj. Close"]):
    # create an empty portfolio
    pf = Portfolio()
    stocksdata = getStocksFromQuandl(names, start, end)
    # get certain columns:
    stocksdata = getStocksDataColumns(stocksdata, names, datacolumns)
    # add stocks to portfolio
    # better to use stocks function here than the below
    # build portfolio at once:
    
    # build portfolio stock by stock:
    for i in range(len(pf_information)):
        name = pf_information.loc[i].Name
        pf.addStock(Stock(pf_information.loc[i],
                          stock_data=stocksdata.filter(regex=name))
                   )
    return pf

def buildPortfolioFromDf(pf_information, stock_data, roi_data=None):
    
    return None

def buildPortfolio(pf_information=None, roi_data=None, names=None, start=None, end=None, datacolumns=None):
    try:
        import quandl
        import datetime
    except ImportError:
        print("The following packages are required:\n - quandl\n - datetime\nPlease ensure that they are installed.")

    # create an empty portfolio
    pf = Portfolio()
    # Get data from quandl
    for name in names:
        try:
            roi = quandl.get(name, start_date=start, end_date=end)
            self.asset[symbol] = DataReader(
                symbol, "yahoo", start=start, end=end)
        except:
            print("Asset " + str(symbol) + " not found!")


    #except:
        #print("Asset " + str(symbol) + " not found!")
    #None
