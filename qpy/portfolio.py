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
    It also requires daily return of investments (ROI) as a pandas.DataFrame
    or pandas.Series. If it is a DataFrame, the "roi_data" data structure is
    required to contain the following label
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
        To fill the portfolio with investment information and daily
        return of investments (ROI) data, the function addStock(stock)
        should be used, in which `stock` is a `Stock` object, a
        pandas.DataFrame of the portfolio investment information.
        The corresponding daily return of investments (ROI) are stored
        in the Stock object.
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

#    def addStocks(self, stocks, data):
#        for i in range(len(df_pf)):
#            self.addStock(Stock(stocks.loc[i], roi))
#
#        for i in range(len(df_pf)):
#            #print(df_pf.loc[i])
#            age = df_pf.loc[i].Age
#            strategy = df_pf.loc[i].Strategy
#            #data = extractRoiData()
#            pf.addStock(Stock(df_pf.loc[i],
#                extractRoiData(df_data, age, strategy)))

    def addStock(self, stock):
        # adding stock to dictionary containing all stocks provided
        self.stocks.update({stock.name: stock})
        # adding information of stock to the portfolio
        self.portfolio = self.portfolio.append(
            stock.getInvestmentInfo(),
            ignore_index=True)
        # also add ROI data of stock to the dataframe containing
        # all roi data points
        if (stock.stock_data is not None):
            self._addStockData(stock.stock_data)

        # set roi_data, if given
        if (stock.roi_data is not None):
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
        # trying num_trials different weights of the investment in the
        # portfolio.
        # return values are:
        # (portfolio with highest sharpe ratio,
        #  portfolio with minimum volatility)
        # both are returned as a pandas.Series
        import numpy as np
        if (plot):
            import matplotlib.pyplot as plt
        # set number of stocks in the portfolio
        num_stocks = len(self.getStocks())
        # set up array to hold results
        res_columns = list(self.getStocks().keys())
        res_columns.extend(['ROI', 'Volatility', 'Sharpe Ratio'])
        results = np.zeros((len(res_columns), num_trials))
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
            # store results in results array
            results[num_stocks, i] = pf_roi
            results[num_stocks+1, i] = pf_volatility
            # store Sharpe Ratio
            results[num_stocks+2, i] = SharpeRatio(pf_roi,
                                                   riskfreerate,
                                                   pf_volatility)

        # other example for optimisation
        # for i in range(n_portfolios):
        #     w = np.random.rand(len(assets))        # random weights
        #     w = w / sum(w)                         # weights sum to 1
        #     port_return = np.dot(w.T, hist_mean.as_matrix()) * 250         # annualize; 250 business days
        #     port_stdev = np.sqrt(np.dot(w.T, np.dot(hist_cov, w))) * np.sqrt(250)  # annualize; 250 business days
        #     port_returns[i] = port_return
        #     port_stdevs[i] = port_stdev
        # transpose and convert to pandas.DataFrame:
        df_results = pd.DataFrame(results.T, columns=res_columns)
        # adding info of max sharpe ratio and of min volatility
        # to resulting df (with meaningful indices):
        pf_opt = pd.DataFrame([df_results.iloc[
            df_results['Sharpe Ratio'].idxmax()],
            df_results.iloc[df_results['Volatility'].idxmin()]],
            index=['Max Sharpe Ratio', 'Min Volatility'])

        # plot results
        if (plot):
            plt.figure()
            # create scatter plot coloured by Sharpe Ratio
            ax = df_results.plot.scatter(x='Volatility',
                                         y='ROI',
                                         c='Sharpe Ratio',
                                         colormap='RdYlBu',
                                         label=None)
            # mark in red the highest sharpe ratio
            pf_opt.loc['Max Sharpe Ratio'].to_frame().T.plot.scatter(
                x='Volatility',
                y='ROI',
                marker="o",
                color='r',
                s=100,
                label='max Sharpe Ratio',
                ax=ax)
            # mark in green the minimum volatility
            pf_opt.loc['Min Volatility'].to_frame().T.plot.scatter(
                x='Volatility',
                y='ROI',
                marker="o",
                color='g',
                s=100,
                label='min Volatility',
                ax=ax)
            plt.title('Monte Carlo simulation to optimise the investments')
            plt.xlabel('Volatility')
            plt.ylabel('ROI')
            plt.legend()
            plt.show()

        return (pf_opt)

    def __str__(self):
        return str(self.getPortfolio())


def _correctQuandlRequestStockName(names):
    ''' This function makes sure that all strings in the given list of
        stock names are leading with "WIKI/" as required by quandl to
        request data.

        Example: If an element of names is "GOOG" (which stands for
        Google), this function modifies the element of names to
        "WIKI/GOOG".
    '''
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


def _quandlRequest(names, start_date=None, end_date=None):
    ''' This function performs a simple request from quandl.

        Input:
         * names: List of strings of stock names to be requested
         * start_date (optional): String/datetime of the start date
             of relevant stock data
         * end_date (optional): String/datetime of the end date of
             relevant stock data
    '''
    try:
        import quandl
    except ImportError:
        print("The following package is required:\n - quandl\n"
              + "Please make sure that it is installed.")
    # get correct stock names that quandl.get can request,
    # e.g. "WIKI/GOOG" for Google
    reqnames = _correctQuandlRequestStockName(names)
    return quandl.get(reqnames, start_date=start_date, end_date=end_date)


def _getQuandlDataColumnLabel(stock_name, data_label):
    ''' Given stock name and label of a data column, this function returns
        the string <stock_name> - <data_label> as it can be found in a
        DataFrame returned by quandl.
    '''
    return stock_name+' - '+data_label


def _getStocksDataColumns(stock_data, names, cols):
    ''' This function returns a subset of the given DataFrame stock_data,
        which contains only the data columns as specified in the input cols.

        Input:
         * stock_data: A DataFrame which contains quantities of the stocks
             listed in pf_information
         * names: A string or list of strings, containing the names of the
             stocks, e.g. 'GOOG' for Google.
         * cols: A list of strings of column labels of stock_data to be
             extracted.
        Output:
         * stock_data: A DataFrame which contains only the data columns of
             stock_data as specified in cols.
    '''
    # get correct stock names that quandl get request
    reqnames = _correctQuandlRequestStockName(names)
    # get current column labels and replacement labels
    reqcolnames = []
    for name in reqnames:
        for col in cols:
            reqcolnames.append(_getQuandlDataColumnLabel(name, col))
    stock_data = stock_data.loc[:, reqcolnames]
    # now rename the columns:
    newcolnames = {}
    for i in reqcolnames:
        newcolnames.update({i: i.replace('WIKI/', '')})
    stock_data.rename(columns=newcolnames, inplace=True)
    return stock_data


def _buildPortfolioFromQuandl(pf_information,
                              names,
                              start_date=None,
                              end_date=None,
                              datacolumns=["Close"]):
    ''' Returns a portfolio based on input in form of a list of
        strings/names of stocks.

        Input:
         * pf_information: DataFrame with the required data column
             labels "Name" and "FMV" of the stocks.
         * names: A string or list of strings, containing the names of
             the stocks, e.g. 'GOOG' for Google.
         * start_date (optional): String/datetime start date of stock data
             to be requested through quandl (default: None)
         * end_date (optional): String/datetime end date of stock data to
             be requested through quandl (default: None)
         * datacolumns (optional): A list of strings of data column labels
             to be extracted and returned (default: ["Close"]).
        Output:
         * pf: Instance of Portfolio which contains all the information
             requested by the user.
    '''
    # create an empty portfolio
    pf = Portfolio()
    # request data from quandl:
    stock_data = _quandlRequest(names, start_date, end_date)
    # extract only certain columns:
    stock_data = _getStocksDataColumns(stock_data, names, datacolumns)
    # build portfolio:
    pf = _buildPortfolioFromDf(pf_information, stock_data=stock_data)
    return pf


def _stocknamesInDataColumns(names, df):
    ''' Returns True if at least one element of names was found as a
        column label in the dataframe df.
    '''
    return any((name in label for name in names for label in df.columns))


def _buildPortfolioFromDf(pf_information, stock_data=None, roi_data=None):
    ''' Returns a portfolio based on input in form of pandas.DataFrame.

        Input:
         * pf_information: DataFrame with the required data column labels
             "Name" and "FMV" of the stocks.
         * stock_data (optional): A DataFrame which contains quantities of
             the stocks listed in pf_information
         * roi_data (optional): A DataFrame which contains the return of
             investment (ROI) data of the stocks listed in pf_information
        Output:
         * pf: Instance of Portfolio which contains all the information
             requested by the user.
    '''
    if ((stock_data is None and roi_data is None) or
       ((stock_data is not None) and (roi_data is not None))):
        raise ValueError("One of the two inpurt arguments stock_data,"
                         + "roi_data must be set.")
    # make sure stock names are in data dataframe
    if (stock_data is None):
        data = roi_data
    elif (roi_data is None):
        data = stock_data
    if (not _stocknamesInDataColumns(pf_information.Name.values, data)):
        raise ValueError("Error: None of the provided stock names were"
                         + "found in the provided dataframe.")
    # building portfolio:
    # better to use stocks function here than the below
    # build portfolio at once:
    # build portfolio stock by stock:
    pf = Portfolio()
    for i in range(len(pf_information)):
        name = pf_information.loc[i].Name
        if (roi_data is None):
            pf.addStock(Stock(pf_information.loc[i],
                              stock_data=stock_data.filter(regex=name))
                        )
        else:
            pf.addStock(Stock(pf_information.loc[i],
                              roi_data=roi_data.filter(regex=name))
                        )
    return pf


def _allListEleInOther(l1, l2):
    ''' Returns True if all elements of list l1 are found in list l2.
    '''
    return all(ele in l2 for ele in l1)


def _anyListEleInOther(l1, l2):
    ''' Returns True if any element of list l1 is found in list l2.
    '''
    return any(ele in l2 for ele in l1)


def _listComplement(A, B):
    ''' Returns the relative complement of A in B (also denoted as A\\B)
    '''
    return list(set(B) - set(A))


def buildPortfolio(pf_information, **kwargs):
    ''' This function builds and returns a portfolio given a set of
        input arguments.

        Input:
         * pf_information: This input is always required. DataFrame
             with the required data column labels "Name" and "FMV"
             of the stocks.
         * names: A string or list of strings, containing the names of
             the stocks, e.g. 'GOOG' for Google.
         * start (optional): String/datetime start date of stock data
             to be requested through quandl (default: None)
         * end (optional): String/datetime end date of stock data to be
             requested through quandl (default: None)
         * stock_data (optional): A DataFrame which contains quantities
             of the stocks listed in pf_information
         * roi_data (optional): A DataFrame which contains the return of
             investment (ROI) data of the stocks listed in pf_information
         * datacolumns (optional): A list of strings of data column labels
             to be extracted and returned.
        Output:
         * pf: Instance of Portfolio which contains all the information
             requested by the user.

        Only the following combinations of inputs are allowed:
         * pf_information, names, start_date (optional), end_date
             (optional), datacolumns (optional)
         * pf_information, stock_data
         * pf_information, roi_data
        In the latter case, stock data (e.g. prices) are not present in
        the resulting portfolio, as the roi_data was given by user.
    '''
    docstringMsg = "Please read through the docstring, " \
                   "'buildPortfolio.__doc__'."
    inputError = "Error: None of the input arguments {} are allowed " \
                 "in combination with {}. "+docstringMsg
    if (kwargs is None):
        raise ValueError("Error: "+docstringMsg)

    # create an empty portfolio
    pf = Portfolio()

    # list of all valid optional input arguments
    allInputArgs = ['names',
                    'start_date',
                    'end_date',
                    'datacolumns',
                    'stock_data',
                    'roi_data']

    # 1. names, start_date, end_date
    allowedInputArgs = ['names',
                        'start_date',
                        'end_date',
                        'datacolumns']
    complementInputArgs = _listComplement(allowedInputArgs, allInputArgs)
    if (_allListEleInOther(['names'], kwargs.keys())):
        # check that no input argument conflict arises:
        if (_anyListEleInOther(complementInputArgs, kwargs.keys())):
            raise ValueError(inputError.format(
                complementInputArgs, allowedInputArgs))
        # get portfolio:
        pf = _buildPortfolioFromQuandl(pf_information, **kwargs)

    # 2. stock_data
    allowedInputArgs = ['stock_data']
    complementInputArgs = _listComplement(allowedInputArgs, allInputArgs)
    if (_allListEleInOther(['stock_data'], kwargs.keys())):
        # check that no input argument conflict arises:
        if (_anyListEleInOther(_listComplement(
             allowedInputArgs, allInputArgs), kwargs.keys())):
            raise ValueError(inputError.format(
                complementInputArgs, allowedInputArgs))
        # get portfolio:
        pf = _buildPortfolioFromDf(pf_information, **kwargs)

    # 3. roi_data
    allowedInputArgs = ['roi_data']
    complementInputArgs = _listComplement(allowedInputArgs, allInputArgs)
    if (_allListEleInOther(['roi_data'], kwargs.keys())):
        # check that no input argument conflict arises:
        if (_anyListEleInOther(_listComplement(
             allowedInputArgs, allInputArgs), kwargs.keys())):
            raise ValueError(inputError.format(
                complementInputArgs, allowedInputArgs))
        # get portfolio:
        pf = _buildPortfolioFromDf(pf_information, **kwargs)

    return pf
