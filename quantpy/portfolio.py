'''
This module is the heart of QuantPy. It provides
 - a class "Stock" that holds and calculates quantities of a single stock,
 - a class "Portfolio" that holds and calculates quantities of a financial
     portfolio, which is a collection of Stock instances.
 - a function "buildPortfolio()" that automatically constructs and returns
     an instance of "Portfolio" and instances of "Stock". The relevant stock
     data is either retrieved through `quandl` or provided by the user as a
     pandas.DataFrame (after loading it manually from disk/reading from file).
     For an example on how to use it, please read the corresponding docstring,
     or have a look at the examples in the sub-directory `example`.

The classes "Stock" and "Portfolio" are designed to easily manage your
financial portfolio, and make the most common quantitative calculations:
 - cumulative returns of the portfolio's stocks
     ( (price_{t} - price_{t=0} + dividend) / price_{t=0} ),
 - daily returns of the portfolio's stocks (daily percentage change),
 - daily log returns of the portfolio's stocks,
 - expected (annualised) return,
 - volatility,
 - Sharpe ratio,
 - skewness of the portfolio's stocks,
 - Kurtosis of the portfolio's stocks,
 - the portfolio's covariance matrix.

"Portfolio" also provides methods to easily compute and visualise
 - simple moving averages of any given time window,
 - exponential moving averages of any given time window,
 - Bollinger Bands of any given time window,

Furthermore, the constructed portfolio can be optimised for
 - minimum volatility,
 - maximum Sharpe ratio
 - minimum volatility for a given expected return
 - maximum Sharpe ratio for a given target volatility
by either performing a numerical computation based on the Efficient
Frontier, or by performing a Monte Carlo simulation of `n` trials.
The former should be the preferred method for reasons of computational effort
and accuracy. The latter is only included for the sake of completeness.

Finally, methods are implemented to generated the following plots:
 - Monte Carlo run to find optimal portfolio(s)
 - Efficient Frontier
 - Portfolio with the minimum volatility based on an Efficient Frontier optimisation
 - Portfolio with the maximum Sharpe ratio based on an Efficient Frontier optimisation
 - Portfolio with the minimum volatility for a given expected return based on an Efficient Frontier optimisation
 - Portfolio with the maximum Sharpe ratio for a given target volatility based on an Efficient Frontier optimisation
 - Individual stocks of the portfolio (expected return over volatility)
'''


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from quantpy.quants import weightedMean, weightedStd, sharpeRatio
from quantpy.optimisation import monte_carlo_optimisation
from quantpy.returns import historicalMeanReturn
from quantpy.returns import dailyReturns, cumulativeReturns, dailyLogReturns
from quantpy.efficient_frontier import EfficientFrontier


class Stock(object):
    '''
    Object that contains information about a stock/fund.
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
    It also requires either data, e.g. daily closing prices as a
    pandas.DataFrame or pandas.Series.
    "data" must be given as a DataFrame, and at least one data column
    is required to containing the closing price, hence it is required to
    contain one column label "<stock_name> - Adj. Close" which is used to
    compute the return of investment. However, "data" can contain more
    data in additional columns.
    '''
    def __init__(self, investmentinfo, data):
        self.name = investmentinfo.Name
        self.investmentinfo = investmentinfo
        self.data = data
        # compute expected return and volatility of stock
        self.expectedReturn = self.compExpectedReturn()
        self.volatility = self.compVolatility()
        self.skew = self._compSkew()
        self.kurtosis = self._compKurtosis()

    # functions to compute quantities
    def compDailyReturns(self):
        '''
        Computes the daily returns (percentage change)
        '''
        return dailyReturns(self.data)

    def compExpectedReturn(self, freq=252):
        '''
        Computes the expected return of the stock.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year
        '''
        return historicalMeanReturn(self.data, freq=freq)

    def compVolatility(self, freq=252):
        '''
        Computes the volatility of the stock.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year
        '''
        return self.compDailyReturns().std() * np.sqrt(freq)

    def _compSkew(self):
        '''
        Computes and returns the skewness of the stock.
        '''
        return self.data.skew().values[0]

    def _compKurtosis(self):
        '''
        Computes and returns the Kurtosis of the stock.
        '''
        return self.data.kurt().values[0]

    def properties(self):
        '''
        Nicely prints out the properties of the stock: expected return,
        volatility, skewness, Kurtosis as well as the FMV (and other
        information provided in investmentinfo.)
        '''
        # nicely printing out information and quantities of the stock
        string = "-"*50
        string += "\nStock: {}".format(self.name)
        string += "\nExpected return:{:0.3f}".format(
            self.expectedReturn.values[0])
        string += "\nVolatility: {:0.3f}".format(
            self.volatility.values[0])
        string += "\nSkewness: {:0.5f}".format(self.skew)
        string += "\nKurtosis: {:0.5f}".format(self.kurtosis)
        string += "\nInformation:"
        string += "\n"+str(self.investmentinfo.to_frame().transpose())
        string += "\n"
        string += "-"*50
        print(string)

    def __str__(self):
        # print short description
        string = "Contains information about "+str(self.name)+"."
        return string


class Portfolio(object):
    '''
    Object that contains information about a investment portfolio.
    To initialise the object, it does not require any input.
    To fill the portfolio with investment information, the
    function addStock(stock) should be used, in which `stock` is
    a `Stock` object, a pandas.DataFrame of the portfolio investment
    information.
    '''
    def __init__(self):
        # initilisating instance variables
        self.portfolio = pd.DataFrame()
        self.stocks = {}
        self.data = pd.DataFrame()
        self.expectedReturn = None
        self.volatility = None
        self.sharpe = None
        self.skew = None
        self.kurtosis = None
        self.totalinvestment = None
        self.riskFreeRate = 0.005
        self.freq = 252
        # instance variable for efficient frontier optimisations
        self.ef = None

    @property
    def totalinvestment(self):
        return self.__totalinvestment

    @totalinvestment.setter
    def totalinvestment(self, val):
        if (val is not None):
            # treat "None" as initialisation
            if (not isinstance(val, (float, int))):
                raise ValueError("Total investment must be a float or "
                                 + "integer.")
            elif (val <= 0):
                raise ValueError("The money to be invested in the "
                                 + "portfolio must be > 0.")
            else:
                self.__totalinvestment = val

    @property
    def freq(self):
        return self.__freq

    @freq.setter
    def freq(self, val):
        if (not isinstance(val, int)):
            raise ValueError("Time window/frequency must be an integer.")
        elif (val <= 0):
            raise ValueError("freq must be > 0.")
        else:
            self.__freq = val
            # now that this changed, update other quantities
            self._update()

    @property
    def riskFreeRate(self):
        return self.__riskFreeRate

    @riskFreeRate.setter
    def riskFreeRate(self, val):
        if (not isinstance(val, (float, int))):
            raise ValueError("Risk free rate must be a float or an integer.")
        else:
            self.__riskFreeRate = val
            # now that this changed, update other quantities
            self._update()

    def addStock(self, stock):
        # adding stock to dictionary containing all stocks provided
        self.stocks.update({stock.name: stock})
        # adding information of stock to the portfolio
        self.portfolio = self.portfolio.append(
            stock.investmentinfo,
            ignore_index=True)
        # setting an appropriate name for the portfolio
        self.portfolio.name = "Allocation of stocks"
        # also add stock data of stock to the dataframe
        self._addStockData(stock.data)

        # update quantities of portfolio
        self._update()

    def _addStockData(self, df):
        # loop over columns in given dataframe
        for datacol in df.columns:
            cols = len(self.data.columns)
            self.data.insert(loc=cols,
                             column=datacol,
                             value=df[datacol].values)
        # set index correctly
        self.data.set_index(df.index.values, inplace=True)
        # set index name:
        self.data.index.rename('Date', inplace=True)

    def _update(self):
        # sanity check (only update values if none of the below is empty):
        if (not (self.portfolio.empty or
                 self.stocks == {} or
                 self.data.empty)):
            self.totalinvestment = self.portfolio.FMV.sum()
            self.expectedReturn = self.compExpectedReturn(freq=self.freq)
            self.volatility = self.compVolatility(freq=self.freq)
            self.sharpe = self.compSharpe()
            self.skew = self._compSkew()
            self.kurtosis = self._compKurtosis()

    def getStock(self, name):
        '''
        Returns the instance of Stock with name <name>.
        '''
        return self.stocks[name]

    def compCumulativeReturns(self):
        '''
        Computes the cumulative returns of all stocks in the
        portfolio.
        (price_{t} - price_{t=0})/ price_{t=0}
        '''
        return cumulativeReturns(self.data)

    def compDailyReturns(self):
        '''
        Computes the daily returns (percentage change) of all
        stocks in the portfolio.
        '''
        return dailyReturns(self.data)

    def compDailyLogReturns(self):
        '''
        Computes the daily log returns of all stocks in the portfolio.
        '''
        return dailyLogReturns(self.data)

    def compMeanReturns(self, freq=252):
        '''
        Computes the mean return based on historical stock price data.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year.
        '''
        return historicalMeanReturn(self.data, freq=freq)

    def compStockVolatility(self, freq=252):
        '''
        Computes the volatilities of all the stocks individually

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year.

        Output:
         * pandas.DataFrame with the individual volatilities of all stocks
             of the portfolio.
        '''
        if (not isinstance(freq, int)):
            raise ValueError("freq is expected to be an integer.")
        return self.compDailyReturns().std() * np.sqrt(freq)

    def compWeights(self):
        '''
        Computes and returns a pandas.Series of the weights of the stocks
        of the portfolio.
        '''
        # computes the weights of the stocks in the given portfolio
        # in respect of the total investment
        return self.portfolio['FMV']/self.totalinvestment

    def compExpectedReturn(self, freq=252):
        '''
        Computes the expected return of the portfolio.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year.
        '''
        if (not isinstance(freq, int)):
            raise ValueError("freq is expected to be an integer.")
        pf_return_means = historicalMeanReturn(self.data,
                                               freq=freq)
        weights = self.compWeights()
        expectedReturn = weightedMean(pf_return_means.values, weights)
        self.expectedReturn = expectedReturn
        return expectedReturn

    def compVolatility(self, freq=252):
        '''
        Computes the volatility of the given portfolio.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year.
        '''
        if (not isinstance(freq, int)):
            raise ValueError("freq is expected to be an integer.")
        # computing the volatility of a portfolio
        volatility = weightedStd(self.compCov(),
                                 self.compWeights()) * np.sqrt(freq)
        self.volatility = volatility
        return volatility

    def compCov(self):
        '''
        Compute and return a pandas.DataFrame of the covariance matrix
        of the portfolio.
        '''
        # get the covariance matrix of the mean returns of the portfolio
        returns = dailyReturns(self.data)
        return returns.cov()

    def compSharpe(self):
        '''
        Compute and return the Sharpe ratio of the portfolio.
        '''
        # compute the Sharpe Ratio of the portfolio
        sharpe = sharpeRatio(self.expectedReturn,
                             self.volatility,
                             self.riskFreeRate)
        self.sharpe = sharpe
        return sharpe

    def _compSkew(self):
        '''
        Computes and returns the skewness of the stocks in the portfolio.
        '''
        return self.data.skew()

    def _compKurtosis(self):
        '''
        Computes and returns the Kurtosis of the stocks in the portfolio.
        '''
        return self.data.kurt()

    # optimising the investments with the efficient frontier class
    def get_EF(self):
        '''
        If self.ef does not exist, create and return an instance of
        quantpy.efficient_frontier.EfficientFrontier, else, return the
        existing instance.
        '''
        if (self.ef is None):
            # create instance of EfficientFrontier
            self.ef = EfficientFrontier(self.compMeanReturns(freq=1),
                                        self.compCov(),
                                        riskFreeRate=self.riskFreeRate,
                                        freq=self.freq)
        return self.ef

    def ef_minimum_volatility(self, verbose=False):
        '''
        Interface to ef.minimum_volatility()
        Finds the portfolio with the minimum volatility.

        Input:
         * verbose: Boolean (default=False), whether to print out properties
             or not.
        '''
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self.get_EF()
        # perform optimisation
        opt_weights = ef.minimum_volatility()
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_maximum_sharpe_ratio(self, verbose=False):
        '''
        Interface to ef.maximum_sharpe_ratio()
        Finds the portfolio with the maximum Sharpe Ratio, also called the
        tangency portfolio.

        Input:
         * verbose: Boolean (default=False), whether to print out properties
             or not.
        '''
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self.get_EF()
        # perform optimisation
        opt_weights = ef.maximum_sharpe_ratio()
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_efficient_return(self, target, verbose=False):
        '''
        Interface to ef.efficient_return()
        Finds the portfolio with the minimum volatility for a given target
        return.

        Input:
         * target: Float, the target return of the optimised portfolio.
         * verbose: Boolean (default=False), whether to print out properties
             or not.
        '''
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self.get_EF()
        # perform optimisation
        opt_weights = ef.efficient_return(target)
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_efficient_volatility(self, target, verbose=False):
        '''
        Interface to ef.efficient_volatility()
        Finds the portfolio with the maximum Sharpe ratio for a given
        target volatility.

        Input:
         * target: Float, the target volatility of the optimised portfolio.
         * verbose: Boolean (default=False), whether to print out properties
             or not.
        '''
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self.get_EF()
        # perform optimisation
        opt_weights = ef.efficient_volatility(target)
        # if verbose==True, print out results
        ef.properties(verbose=verbose)
        return opt_weights

    def ef_efficient_frontier(self, targets=None):
        '''
        Gets portfolios for a range of given target returns.
        If no targets were provided, the algorithm will find the minimum
        and maximum returns of the portfolio's individual stocks, and set
        the target range according to those values.
        Results in the Efficient Frontier.

        Input:
         * targets: list/numpy.ndarray (default: None) of floats,
             range of target returns.

        Output:
         * array of (volatility, return) values.
        '''
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self.get_EF()
        # perform optimisation
        efrontier = ef.efficient_frontier(targets)
        return efrontier

    def ef_plot_efrontier(self):
        '''
        Plots the Efficient Frontier.
        '''
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self.get_EF()
        # plot efficient frontier
        ef.plot_efrontier()

    def ef_plot_optimal_portfolios(self):
        '''
        Plots the optimised portfolios for
         - minimum volatility, and
         - maximum Sharpe ratio.
        '''
        # let EfficientFrontier.efficient_frontier handle input arguments
        # get/create instance of EfficientFrontier
        ef = self.get_EF()
        # plot efficient frontier
        ef.plot_optimal_portfolios()

    def plot_stocks(self, freq=252):
        '''
        Plots the expected annual returns over annual volatility of
        the stocks of the portfolio.

        Input:
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year.
        '''
        # annual mean returns of all stocks
        stock_returns = self.compMeanReturns(freq=freq)
        stock_volatility = self.compStockVolatility(freq=freq)
        # adding stocks of the portfolio to the plot
        # plot stocks individually:
        plt.scatter(stock_volatility,
                    stock_returns,
                    marker='o',
                    s=100,
                    label="Stocks")
        # adding text to stocks in plot:
        for i, txt in enumerate(stock_returns.index):
            plt.annotate(txt,
                         (stock_volatility[i], stock_returns[i]),
                         xytext=(10,0),
                         textcoords='offset points',
                         label=i)
            plt.legend()

    # optimising the investments by performing a Monte Carlo run
    # based on volatility and sharpe ratio
    def mc_optimisation(self,
                      total_investment=None,
                      num_trials=10000,
                      freq=252,
                      verbose=True,
                      plot=True):
        '''
        Optimisation of the portfolio by performing a Monte Carlo simulation.

        Input:
         * total_investment: Float (default: None, which results in the sum of
             FMV of the portfolio information), money to be invested.
         * num_trials: Integer (default: 10000), number of portfolios to be
             computed, each with a random distribution of weights/investments
             in each stock.
         * freq: Integer (default: 252), number of trading days, default
             value corresponds to trading days in a year.
         * verbose: Boolean (default: True), if True, prints out optimised
             portfolio allocations.
         * plot: Boolean (default: True), if True, a plot of the Monte Carlo
             simulation is shown.
        '''
        # if total_investment is not set, use total FMV of given portfolio
        if (total_investment is None):
            total_investment = self.totalinvestment

        return monte_carlo_optimisation(self.data,
                          num_trials=num_trials,
                          total_investment=total_investment,
                          riskFreeRate=self.riskFreeRate,
                          freq=freq,
                          initial_weights=self.compWeights().values,
                          verbose=verbose,
                          plot=plot)

    def properties(self):
        '''
        Nicely prints out the properties of the portfolio: expected return,
        volatility, Sharpe ratio, skewness, Kurtosis as well as the allocation
        of the stocks across the portfolio.
        '''
        # nicely printing out information and quantities of the portfolio
        string = "-"*70
        stocknames = self.portfolio.Name.values.tolist()
        string += "\nStocks: {}".format(", ".join(stocknames))
        string += "\nTime window/frequency: {}".format(self.freq)
        string += "\nRisk free rate: {}".format(self.riskFreeRate)
        string += "\nPortfolio expected return: {:0.3f}".format(
            self.expectedReturn)
        string += "\nPortfolio volatility: {:0.3f}".format(
            self.volatility)
        string += "\nPortfolio Sharpe ratio: {:0.3f}".format(
            self.sharpe)
        string += "\n\nSkewness:"
        string += "\n"+str(self.skew.to_frame().transpose())
        string += "\n\nKurtosis:"
        string += "\n"+str(self.kurtosis.to_frame().transpose())
        string += "\n\nInformation:"
        string += "\n"+str(self.portfolio)
        string += "\n"
        string += "-"*70
        print(string)

    def __str__(self):
        # print short description
        string = "Contains information about a portfolio."
        return string


def _correctQuandlRequestStockName(names):
    '''
    This function makes sure that all strings in the given list of
    stock names are leading with "WIKI/" as required by quandl to
    request data.

    Example: If an element of names is "GOOG" (which stands for
    Google), this function modifies the element of names to "WIKI/GOOG".
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
    '''
    This function performs a simple request from quandl and returns
    a DataFrame containing stock data.

    Input:
     * names: List of strings of stock names to be requested
     * start_date (optional): String/datetime of the start date of
         relevant stock data.
     * end_date (optional): String/datetime of the end date of
         relevant stock data.
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
    '''
    Given stock name and label of a data column, this function returns
    the string "<stock_name> - <data_label>" as it can be found in a
    DataFrame returned by quandl.
    '''
    return stock_name+' - '+data_label


def _getStocksDataColumns(data, names, cols):
    '''
    This function returns a subset of the given DataFrame data, which
    contains only the data columns as specified in the input cols.

        Input:
         * data: A DataFrame which contains quantities of the stocks
             listed in pf_allocation.
         * names: A string or list of strings, containing the names of the
             stocks, e.g. 'GOOG' for Google.
         * cols: A list of strings of column labels of data to be
             extracted.
        Output:
         * data: A DataFrame which contains only the data columns of
             data as specified in cols.
    '''
    # get correct stock names that quandl get request
    reqnames = _correctQuandlRequestStockName(names)
    # get current column labels and replacement labels
    reqcolnames = []
    for i in range(len(names)):
        for col in cols:
            # differ between dataframe directly from quandl and
            # possibly previously processed dataframe, e.g.
            # read in from disk with slightly modified column labels
            # 1. if <stock_name> in column labels
            if (names[i] in data.columns):
                colname = names[i]
            # 2. if "WIKI/<stock_name> - <col>" in column labels
            elif (_getQuandlDataColumnLabel(reqnames[i], col) in
                  data.columns):
                colname = _getQuandlDataColumnLabel(reqnames[i], col)
            # 3. if "<stock_name> - <col>" in column labels
            elif (_getQuandlDataColumnLabel(names[i], col) in
                  data.columns):
                colname = _getQuandlDataColumnLabel(names[i], col)
            # else, error
            else:
                raise ValueError("Could not find column labels in given "
                                 + "dataframe.")
            # append correct name to list of correct names
            reqcolnames.append(colname)

    data = data.loc[:, reqcolnames]
    # now rename the columns (removing "WIKI/" from column labels):
    newcolnames = {}
    for i in reqcolnames:
        newcolnames.update({i: i.replace('WIKI/', '')})
    data.rename(columns=newcolnames, inplace=True)
    # if only one data column per stock exists, rename column labels
    # to the name of the corresponding stock
    newcolnames = {}
    if (len(cols) == 1):
        for i in range(len(names)):
            newcolnames.update({_getQuandlDataColumnLabel(
                names[i], cols[0]): names[i]})
        data.rename(columns=newcolnames, inplace=True)
    return data


def _buildPortfolioFromQuandl(names,
                              pf_allocation,
                              start_date=None,
                              end_date=None):
    '''
    Returns a portfolio based on input in form of a list of strings/names
    of stocks.

    Input:
     * names: A string or list of strings, containing the names of the
         stocks, e.g. 'GOOG' for Google.
     * pf_allocation: DataFrame with the required data column
         labels "Name" and "FMV" of the stocks.
     * start_date (optional): String/datetime start date of stock data to
         be requested through quandl (default: None)
     * end_date (optional): String/datetime end date of stock data to be
         requested through quandl (default: None)
    Output:
     * pf: Instance of Portfolio which contains all the information
         requested by the user.
    '''
    # create an empty portfolio
    pf = Portfolio()
    # request data from quandl:
    data = _quandlRequest(names, start_date, end_date)
    # build portfolio:
    pf = _buildPortfolioFromDf(data, pf_allocation)
    return pf


def _stocknamesInDataColumns(names, df):
    '''
    Returns True if at least one element of names was found as a column
    label in the dataframe df.
    '''
    return any((name in label for name in names for label in df.columns))


def _generatePfAllocation(data):
    '''
    Takes column names of provided DataFrame "data", and generates a
    DataFrame with columns "Name" and "FMV" which contain the names found
    in input "data" and 1./len(data.columns) respectively.

    Input:
     * data: A DataFrame which contains prices of the stocks

    Output:
     * pf_allocation: pandas.DataFrame with columns 'Name' and 'FMV', which
         contain the names and weights of the stocks
    '''
    names = data.columns
    # sanity check: split names at '-' and take the leading part of the
    # split string, and check if this occurs in any of the other names.
    # if so, we treat this as a duplication, and ask the user to provide
    # a DataFrame with one data column per stock.
    splitnames = [name.split('-')[0].strip() for name in names]
    for i in range(len(splitnames)):
            splitname = splitnames[i]
            reducedlist = [elt for num, elt in enumerate(splitnames)
                            if not num == i]
            if (splitname in reducedlist):
                errormsg = "'data' DataFrame contains conflicting "\
                        + "column labels."\
                        + "\nMultiple columns with a substring of "\
                        + "\n "+str(splitname)+"\n"\
                        + "were found. You have two options:"\
                        + "\n 1. call 'buildPortfolio' and pass a "\
                        + "DataFrame 'pf_allocation' that contains the "\
                        + "weights/allocation of stocks within your "\
                        + "portfolio. 'buildPortfolio' will then extract "\
                        + "the columns from 'data' that match the values of "\
                        + "the column 'Name' in the DataFrame 'pf_allocation'."\
                        + "\n 2. call 'buildPortfolio' and pass a DataFrame "\
                        + "'data' that does not have conflicting column labels, "\
                        + "e.g. 'GOOG' and 'GOOG - Adj. Close' are considered "\
                        + "conflicting column headers."
                raise ValueError(errormsg)
    # compute equal weights
    weights = [1./len(names) for i in range(len(names))]
    return pd.DataFrame({'FMV' : weights, 'Name': names})


def _buildPortfolioFromDf(data,
                          pf_allocation=None,
                          datacolumns=["Adj. Close"]):
    '''
    Returns a portfolio based on input in form of pandas.DataFrame.

    Input:
     * data: A DataFrame which contains prices of the stocks listed in
         pf_allocation
     * pf_allocation: DataFrame with the required data column labels
         "Name" and "FMV" of the stocks.
     * datacolumns (optional): A list of strings of data column labels
         to be extracted and returned (default: ["Adj. Close"]).
    Output:
     * pf: Instance of Portfolio which contains all the information
         requested by the user.
    '''
    # if pf_allocation is None, automatically generate it
    if (pf_allocation is None):
        pf_allocation = _generatePfAllocation(data)
    # make sure stock names are in data dataframe
    if (not _stocknamesInDataColumns(pf_allocation.Name.values,
                                     data)):
        raise ValueError("Error: None of the provided stock names were"
                         + "found in the provided dataframe.")
    # extract only 'Adj. Close' column from DataFrame:
    data = _getStocksDataColumns(data,
                                 pf_allocation.Name.values,
                                 datacolumns)
    # building portfolio:
    pf = Portfolio()
    for i in range(len(pf_allocation)):
        # get name of stock
        name = pf_allocation.loc[i].Name
        # extract data column(s) of said stock
        stock_data = data.filter(regex=name).copy(deep=True)
        # if only one data column per stock exists, give dataframe a name
        if (len(datacolumns) == 1):
            stock_data.name = datacolumns[0]
        # create Stock instance and add it to portfolio
        pf.addStock(Stock(pf_allocation.loc[i],
                          data=stock_data))
    return pf


def _allListEleInOther(l1, l2):
    '''
    Returns True if all elements of list l1 are found in list l2.
    '''
    return all(ele in l2 for ele in l1)


def _anyListEleInOther(l1, l2):
    '''
    Returns True if any element of list l1 is found in list l2.
    '''
    return any(ele in l2 for ele in l1)


def _listComplement(A, B):
    '''
    Returns the relative complement of A in B (also denoted as A\\B)
    '''
    return list(set(B) - set(A))


def buildPortfolio(**kwargs):
    '''
    This function builds and returns a portfolio given a set of input
    arguments.

    Input:
     * pf_allocation (optional): DataFrame with the required data column
         labels "Name" and "FMV" of the stocks. If not given, it is
         automatically generated with an equal weights for all stocks
         in the resulting portfolio.
     * names (optional): A string or list of strings, containing the names of the
         stocks, e.g. 'GOOG' for Google.
     * start (optional): String/datetime start date of stock data to be
         requested through quandl (default: None).
     * end (optional): String/datetime end date of stock data to be
         requested through quandl (default: None).
     * data (optional): A DataFrame which contains quantities of
         the stocks listed in pf_allocation.
    Output:
     * pf: Instance of Portfolio which contains all the information
         requested by the user.

    Only the following combinations of inputs are allowed:
     * pf_allocation (optional), names, start_date (optional), end_date (optional)
     * pf_allocation (optional), data

    Moreover, the two different ways this function can be used are useful
    for
     1. building a portfolio by pulling data from quandl,
     2. building a portfolio by providing stock data which was obtained
         otherwise, e.g. from data files.
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
    allInputArgs = ['pf_allocation',
                    'names',
                    'start_date',
                    'end_date',
                    'data']

    # 1. pf_allocation, names, start_date, end_date
    allowedInputArgs = ['pf_allocation',
                        'names',
                        'start_date',
                        'end_date']
    complementInputArgs = _listComplement(allowedInputArgs, allInputArgs)
    if (_allListEleInOther(['names'], kwargs.keys())):
        # check that no input argument conflict arises:
        if (_anyListEleInOther(complementInputArgs, kwargs.keys())):
            raise ValueError(inputError.format(
                complementInputArgs, allowedInputArgs))
        # get portfolio:
        pf = _buildPortfolioFromQuandl(**kwargs)

    # 2. pf_allocation, data
    allowedInputArgs = ['pf_allocation',
                        'data']
    complementInputArgs = _listComplement(allowedInputArgs, allInputArgs)
    if (_allListEleInOther(['data'], kwargs.keys())):
        # check that no input argument conflict arises:
        if (_anyListEleInOther(_listComplement(
             allowedInputArgs, allInputArgs), kwargs.keys())):
            raise ValueError(inputError.format(
                complementInputArgs, allowedInputArgs))
        # get portfolio:
        pf = _buildPortfolioFromDf(**kwargs)

    return pf
