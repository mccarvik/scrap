import pdb, sys, getopt
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt

API_KEY = '2IUT4PK1RKTVJ4MI'

# Might be a better place for this, wasn't sure what made the most sense
PORTFOLIO_VALUE = 1000000

class Position():
    """Position Class - holds simple data about each position
    
        Parameters
        ------------
        symbol : string
            ticker symbol
        weight : float
            proposed portfolio weight
    """
    def __init__(self, symbol, weight):
        self.wgt = weight
        self.sym = symbol
    
    def divs_collected(self):
        return self.hist['dividend_amount'].sum() * self.shares
        
    
    def calc_pos_nav(self, dt):
        return self.shares * float(self.hist[self.hist.timestamp == dt]['adjusted_close'])
    
    def grab_history(self, start, end=dt.datetime.today()):
        # Pulls in the whole history for each ticker so takes a little bit to run, not ideal but didnt see a time parameter option in the api specs
        hist = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol={0}&apikey={1}&datatype=csv'.format(self.sym, API_KEY))
        self.hist = hist[(hist.timestamp > start.strftime('%Y-%m-%d')) & (hist.timestamp < end.strftime('%Y-%m-%d'))]
        self.init_price = self.hist.iloc[-1]['adjusted_close']
        # This will create fractional shares, not realistic but didnt know if you wanted there to be a cash position instead for the remainder
        self.shares = (PORTFOLIO_VALUE * self.wgt) / self.init_price
    

def run(pos, start_dt, end_dt=dt.datetime.today()):
    # Grab Index data
    spy_hist = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol={0}&apikey={1}&datatype=csv'.format('SPY', API_KEY))
    spy_hist = spy_hist[(spy_hist.timestamp > start_dt.strftime('%Y-%m-%d')) & (spy_hist.timestamp < end_dt.strftime('%Y-%m-%d'))]
    
    port_navs = pd.DataFrame(columns=['timestamp','adjusted_close'])
    for d in list(spy_hist['timestamp'].order(ascending=True)):
        nav_temp = 0
        for p in pos:
            nav_temp += p.calc_pos_nav(d)
        port_navs.loc[len(port_navs)] = [d, nav_temp]
    print_stats(pos, port_navs, spy_hist, start_dt, end_dt)


def timeline_chart(pos, port_nav, spy_hist):
    pass

def moving_avg_chart(port_navs, spy_hist):
    pass


def print_stats(pos, port_navs, spy_hist, start_dt, end_dt):
    # vol, corr, beta
    # sortino, treynor
    # Return 
    spy_ret = round(((spy_hist.iloc[0]['adjusted_close'] / spy_hist.iloc[-1]['adjusted_close']) - 1) * 100, 3)
    print("SPY RETURN OVER HISTORY: " + str(spy_ret) + "%")
    port_ret = round(((port_navs.iloc[-1]['adjusted_close'] / port_navs.iloc[0]['adjusted_close']) - 1) * 100, 3)
    print("PORTFOLIO RETURN OVER HISTORY: " + str(port_ret) + "%")
    print()
    
    # Total Return
    # Simply adding the dividends collected to the ending NAV as no reinvestment strategy was specified
    spy_divs = (PORTFOLIO_VALUE / spy_hist.iloc[-1]['adjusted_close']) * spy_hist['dividend_amount'].sum()
    spy_tot_ret = round(((((1 + spy_ret/100) * PORTFOLIO_VALUE + spy_divs) / PORTFOLIO_VALUE) - 1) * 100, 3)
    print("SPY TOTAL RETURN OVER HISTORY: " + str(spy_tot_ret) + "%")
    port_divs = sum([p.divs_collected() for p in pos])
    port_tot_ret = round(((((1 + port_ret/100) * PORTFOLIO_VALUE + port_divs) / PORTFOLIO_VALUE) - 1) * 100, 3)
    print("PORTFOLIO TOTAL RETURN OVER HISTORY: " + str(port_tot_ret) + "%")


if __name__ == '__main__':
    ''' USAGE
    Parameters
    ==========
    start_dt : datetime
        Enter as command line arg as yyyymmdd string using '-d'
    weights : list
        list of weights to use for each of the tickers (if adding a ticker it will be the 6th item and use the 6th weight)
        enter as command line arg as list of comma separated floats using '-w'
    ticker : string
        string representing optional additional ticker, enter as command line arg using '-t'
    
    EXAMPLE COMMAND LINE RUN: python coding_challenge.py -d 20150101 -w 0.5,0.1,0.1,0.1,0.1,0.1 -t MSFT
    '''
    start_dt = dt.datetime(2000,1,1)
    tickers = ['googl', 'ba', 'f', 'vz', 'ge']
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    opts, args = getopt.getopt(sys.argv[1:], 'd:w:t:')
    for opt, arg in opts:
        if opt == '-d':
            start_dt = dt.datetime.strptime(arg, '%Y%m%d')
        elif opt == '-w':
            # No sanity check on weights, can equal < or > than 100. Thought this was logical in case of some cash holding or overlevered to get > 100%
            weights = [float(a) for a in arg.split(",")]
        elif opt == '-t':
            tickers.append(arg)
    
    positions = []
    # For testing purposes
    tickers = ['MSFT']
    weights = [1]
    for t, w in zip(tickers, weights):
        t_pos = Position(t, w)
        t_pos.grab_history(start_dt)
        positions.append(t_pos)
        
    run(positions, start_dt)
    