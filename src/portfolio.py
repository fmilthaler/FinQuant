import pandas as pd
class Fund(object):
    ''' Object that contains information about a fund.
    To initialise the object, it requires a name, information about
    the fund given as one of the following data structures:
     - pandas.Series
     - pandas.DataFrame
    The investment information should normally contain labels such as
     - Name
     - Year
     - Strategy
     - CCY
     - FMV
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
    def expectedRoi(self):
        return None
    def __str__(self):
        #result = self.name+":\n"
        #for label, value in self.investmentinfo.iteritems():
        #    result = result+str(label)+": "+str(value)+"\n"
        #return result
        return str(pd.DataFrame([self.investmentinfo]))

class Portfolio(object):
    ''' Object that contains information about a investment portfolio.
    To initialise the object, it requires a name, reference year.
    To fill the portfolio with investment information and daily return of investments
    (ROI) data, the function addFund(fund) should be used, in which `fund` is a `Fund`
    object. a pandas.DataFrame
    of the portfolio investment information. The corresponding daily return of investments
    (ROI) are stored in the Fund object.
    '''
    #def __init__(self, name, ref_year, portfolio):
    def __init__(self, name, ref_year):
        self.name = name
        self.ref_year = ref_year
        self.portfolio = pd.DataFrame()
        #self.portfolio = pd.DataFrame(portfolio)
        self.funds = {}
        # add column "Age" to dataframe:
        #self.addAge()
        #self.pf_roi_data = self.extractPfRoiData()
        #self.funds = self.extractFunds()
        self.pf_roi_data = pd.DataFrame()

    #def addAge(self):
    #    if (not 'Age' in self.portfolio.columns):
    #        idx = self.portfolio.columns.get_loc('Year')
    #        self.portfolio.insert(loc=idx+1, column='Age',
    #                              value=self.getRefYear()-self.portfolio['Year'].values)

    #def extractPfRoiData(self):
    #    # get dictionary that holds relevant information to extract 
    #    # data from the raw dataset:
    #    d = {}
    #    for i in range(len(self.portfolio.index)):
    #        d.update({self.portfolio.Name[i] : {'Age' : self.portfolio.Age[i],
    #                                            'Strategy' : self.portfolio.Strategy[i]}})
    #    # generate a string to query the relevant data:
    #    querystring = ''
    #    for key, value in d.items():
    #        querystring = querystring+'Age=='+str(value['Age'])+' & Strategy=="'+str(value['Strategy'])+'" | '
    #    # remove trailing ' | ' from querystring
    #    querystring = querystring[:-3]
    #    # get data for the given portfolio:
    #    pf_roi_data = self.getRawData().query(querystring).reset_index(drop=True)
    #    return pf_roi_data

    #def extractFunds(self):
    #    funds = {}
    #    for i in range(len(self.portfolio.index)):
    #        name = self.portfolio.Name[i]
    #        age = self.portfolio.Age[i]
    #        strategy = self.portfolio.Strategy[i]
    #        if ('Name' in self.pf_roi_data.columns):
    #            querystring = 'Name=='+str(name)
    #        elif (all(x in self.pf_roi_data.columns for x in ['Age', 'Strategy'])):
    #            None
    #        #elif ('Age' in self.pf_roi_data.columns):
    #        #roi_data = 
    #        roi_data = []
    #        funds.update({name : {'Age' : age,
    #                              'Strategy' : strategy,
    #                              'ROI' : roi_data}})
    #    return funds

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

    def getPortfolio(self):
        return self.portfolio

    def getFunds(self):
        return self.funds

    def getRefYear(self):
        return self.ref_year

    def getPfRoiData(self):
        return self.pf_roi_data

    def getFund(self, name):
        return self.getFunds()[name]

    def getFunds(self):
        return self.funds

    def getCovPf(self):
        # get the covariance matrix of the roi of the portfolio
        return self.pf_roi_data.cov()

    def compPfWeights(self):
        import numpy as np
        # computes the weights of the funds in the given portfolio
        # in respect of the total investment
        total = self.portfolio.FMV.sum()
        weights = self.portfolio.FMV/total
        #weights = []
        #for key in self.funds.keys():
        #    weights.append(self.getFund(key).getInvestmentInfo().FMV/total)
        #return np.array(weights)
        return weights

    def compPfMeans(self):
        return self.getPfRoiData().mean().values

    def compPfExpectedRoi(self):
        import numpy as np
        #calculate portfolio ROI
        pf_means = self.compPfMeans()
        pf_weights = self.compPfWeights()
        expectedRoi = np.sum(pf_means * pf_weights)
        return expectedRoi
    def __str__(self):
        return str(self.getPortfolio())

class PortfolioBKUP(object):
    ''' Object that contains information about a investment portfolio.
    To initialise the object, it requires a name, reference year, a pandas.DataFrame
    of the portfolio investment information. The corresponding daily return of investments
    (ROI) are stored in the Fund object.
    '''
    def __init__(self, name, ref_year, portfolio, raw_data):
        self.name = name
        self.ref_year = ref_year
        self.portfolio = pd.DataFrame(portfolio)
        self.raw_data = pd.DataFrame(raw_data)
        # add column "Age" to dataframe:
        self.addAge()
        self.pf_roi_data = self.extractPfRoiData()
        self.funds = self.extractFunds()

    def addAge(self):
        if (not 'Age' in self.portfolio.columns):
            idx = self.portfolio.columns.get_loc('Year')
            self.portfolio.insert(loc=idx+1, column='Age',
                                  value=self.getRefYear()-self.portfolio['Year'].values)

    def extractPfRoiData(self):
        # get dictionary that holds relevant information to extract 
        # data from the raw dataset:
        d = {}
        for i in range(len(self.portfolio.index)):
            d.update({self.portfolio.Name[i] : {'Age' : self.portfolio.Age[i],
                                                'Strategy' : self.portfolio.Strategy[i]}})
        # generate a string to query the relevant data:
        querystring = ''
        for key, value in d.items():
            querystring = querystring+'Age=='+str(value['Age'])+' & Strategy=="'+str(value['Strategy'])+'" | '
        # remove trailing ' | ' from querystring
        querystring = querystring[:-3]
        # get data for the given portfolio:
        pf_roi_data = self.getRawData().query(querystring).reset_index(drop=True)
        return pf_roi_data

    def extractFunds(self):
        funds = {}
        for i in range(len(self.portfolio.index)):
            name = self.portfolio.Name[i]
            age = self.portfolio.Age[i]
            strategy = self.portfolio.Strategy[i]
            if ('Name' in self.pf_roi_data.columns):
                querystring = 'Name=='+str(name)
            elif (all(x in self.pf_roi_data.columns for x in ['Age', 'Strategy'])):
                None
            #elif ('Age' in self.pf_roi_data.columns):
            #roi_data = 
            roi_data = []
            funds.update({name : {'Age' : age,
                                  'Strategy' : strategy,
                                  'ROI' : roi_data}})
        return funds

    def addFund(self, fund):
        #self.funds.update({fund.name : fund})
        self.portfolio = self.portfolio.append(fund.getInvestmentInfo(), ignore_index=True)
                                              #self.funds[key].getInvestmentInfo(), ignore_index=True)

    #def plotFundsRoi(self):
    #    #for

    #def plotFundRoi(self):
    #    

    #def getPortfolio(self):
    #    df = pd.DataFrame()
    #    for key in self.funds.keys():
    #        df = df.append(self.funds[key].getInvestmentInfo(), ignore_index=True)
    #    return df
    #    #return pd.DataFrame.from_dict(self.funds, orient='index', columns=["ID","Fund","Vintage","Region","Strategy","CCY","FMV"])
    #    #return pd.DataFrame.from_dict(self.funds, orient='index')
    #    #return pd.DataFrame(self.funds)
    #    #pd.DataFrame.from_dict(data, orient='index',
    #    #                  columns=['A', 'B', 'C', 'D'])
    def getRefYear(self):
        return self.ref_year
    def getPortfolio(self):
        return self.portfolio
    def getRawData(self):
        return self.raw_data
    def getPfRoiData(self):
        return self.pf_roi_data
    def getFund(self, name):
        fund = self.getPortfolio().loc[self.getPortfolio()['Name'] == name]
        return fund
    def __str__(self):
        return str(self.getPortfolio())
