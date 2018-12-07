import pandas as pd

class Portfolio(object):
    def __init__(self, name, portfolio, raw_data):
        self.name = name
        self.portfolio = pd.DataFrame(portfolio)
        self.raw_data = pd.DataFrame(raw_data)
        self.pf_roi_data = self.extractPfRoiData()
    def extractPfRoiData(self):
        pf_roi_data = None
        return pd.DataFrame([])
    def getPortfolio(self):
        return self.portfolio
    def getRawData(self):
        return self.raw_data
    def getPortfolioRoiData(self):
        return self.pf_roi_data
    def getFund(self, name):
        fund = self.getPortfolio().loc[self.getPortfolio()['Name'] == name]
        return fund
    def __str__(self):
        return str(self.getPortfolio())
