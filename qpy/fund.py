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
    def __str__(self):
        return str(pd.DataFrame([self.investmentinfo]))
