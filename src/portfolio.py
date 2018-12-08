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
    def __init__(self, name, ref_year, portfolio, raw_data):
        self.name = name
        self.ref_year = ref_year
        self.portfolio = pd.DataFrame(portfolio)
        self.raw_data = pd.DataFrame(raw_data)
        # add column "Age" to dataframe:
        self.addAge()
        self.pf_roi_data = self.extractPfRoiData()

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
