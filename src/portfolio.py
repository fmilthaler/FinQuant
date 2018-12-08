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

class Portfolio(object):
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
