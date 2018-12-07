import pandas as pd

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
