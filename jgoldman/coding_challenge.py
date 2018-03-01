import warnings
warnings.filterwarnings('ignore')
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
        self.hist = self.hist.sort_values(by=['timestamp'], ascending=True)
        # This will create fractional shares, not realistic but didnt know if you wanted there to be a cash position instead for the remainder
        self.shares = (PORTFOLIO_VALUE * self.wgt) / self.init_price
    

def run(pos, start_dt, end_dt=dt.datetime.today()):
    # Grab Index data
    spy_hist = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&outputsize=full&symbol={0}&apikey={1}&datatype=csv'.format('SPY', API_KEY))
    spy_hist = spy_hist[(spy_hist.timestamp > start_dt.strftime('%Y-%m-%d')) & (spy_hist.timestamp < end_dt.strftime('%Y-%m-%d'))]
    spy_hist = spy_hist.sort_values(by=['timestamp'], ascending=True)
    
    port_navs = pd.DataFrame(columns=['timestamp','adjusted_close'])
    for d in list(spy_hist['timestamp']):
        nav_temp = 0
        for p in pos:
            nav_temp += p.calc_pos_nav(d)
        port_navs.loc[len(port_navs)] = [d, nav_temp]
        
    print_stats(pos, port_navs, spy_hist)
    moving_avg_chart(port_navs, spy_hist)
    timeline_chart(pos, port_navs, spy_hist)
    


def timeline_chart(pos, port_navs, spy_hist):
    port_navs['timestamp'] = pd.to_datetime(port_navs['timestamp'])
    spy_hist['timestamp'] = pd.to_datetime(spy_hist['timestamp'])
    port_navs = port_navs.set_index('timestamp')
    spy_hist = spy_hist.set_index('timestamp')
    
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('index level')
    plt.plot(port_navs['adjusted_close'] / port_navs['adjusted_close'].ix[0] * 100, label='port')
    plt.plot(spy_hist['adjusted_close'] / spy_hist['adjusted_close'].ix[0] * 100, label='spy')
    for p in pos:
        hist = p.hist
        hist['timestamp'] = pd.to_datetime(hist['timestamp'])
        hist = hist.set_index('timestamp')
        plt.plot(hist['adjusted_close'] / hist['adjusted_close'].ix[0] * 100, label=p.sym)
    
    plt.legend(loc='upper left')
    plt.savefig('timeline.png', dpi=300)
    plt.close()

def moving_avg_chart(port_navs, spy_hist):
    port_navs['timestamp'] = pd.to_datetime(port_navs['timestamp'])
    spy_hist['timestamp'] = pd.to_datetime(spy_hist['timestamp'])
    port_navs = port_navs.set_index('timestamp')
    spy_hist = spy_hist.set_index('timestamp')
    spy_hist['50d'] = pd.rolling_mean(spy_hist['adjusted_close'], window=50) / spy_hist['adjusted_close'].ix[0] * 100
    spy_hist['252d'] = pd.rolling_mean(spy_hist['adjusted_close'], window=252) / spy_hist['adjusted_close'].ix[0] * 100
    port_navs['50d'] = pd.rolling_mean(port_navs['adjusted_close'], window=50) / port_navs['adjusted_close'].ix[0] * 100
    port_navs['252d'] = pd.rolling_mean(port_navs['adjusted_close'], window=252) / port_navs['adjusted_close'].ix[0] * 100
    spy_hist['adjusted_close'] = spy_hist['adjusted_close'] / spy_hist['adjusted_close'].ix[0] * 100
    port_navs['adjusted_close'] = port_navs['adjusted_close'] / port_navs['adjusted_close'].ix[0] * 100
    
    plt.figure(figsize=(10, 6))
    plt.xlabel('time')
    plt.ylabel('index level')
    plt.plot(port_navs[['adjusted_close', '50d', '252d']], label=['port', 'port 50d', 'port 252d'])
    plt.plot(spy_hist[['adjusted_close', '50d', '252d']], label=['spy', 'spy 50d', 'spy 252d'])
    plt.legend(['port', 'port 50d', 'port 252d', 'spy', 'spy 50d', 'spy 252d'], loc='upper left')
    plt.savefig('mvg_avg.png', dpi=300)
    plt.close()


def print_stats(pos, port_navs, spy_hist):
    # vol, corr, beta
    # sortino, treynor
    # Return 
    spy_ret = round(((spy_hist.iloc[-1]['adjusted_close'] / spy_hist.iloc[0]['adjusted_close']) - 1) * 100, 3)
    print("SPY RETURN OVER HISTORY: " + str(spy_ret) + "%")
    port_ret = round(((port_navs.iloc[-1]['adjusted_close'] / port_navs.iloc[0]['adjusted_close']) - 1) * 100, 3)
    print("PORTFOLIO RETURN OVER HISTORY: " + str(port_ret) + "%")
    print()
    
    # Total Return
    # Simply adding the dividends collected to the ending NAV as no reinvestment strategy was specified
    spy_divs = (PORTFOLIO_VALUE / spy_hist.iloc[0]['adjusted_close']) * spy_hist['dividend_amount'].sum()
    spy_tot_ret = round(((((1 + spy_ret/100) * PORTFOLIO_VALUE + spy_divs) / PORTFOLIO_VALUE) - 1) * 100, 3)
    print("SPY TOTAL RETURN OVER HISTORY: " + str(spy_tot_ret) + "%")
    port_divs = sum([p.divs_collected() for p in pos])
    port_tot_ret = round(((((1 + port_ret/100) * PORTFOLIO_VALUE + port_divs) / PORTFOLIO_VALUE) - 1) * 100, 3)
    print("PORTFOLIO TOTAL RETURN OVER HISTORY: " + str(port_tot_ret) + "%")
    print()
    
    # Max Drawdown
    max_nav = port_navs.iloc[0]['adjusted_close']
    max_drawdown = 0
    max_drawdown_dt = port_navs.iloc[0]['timestamp']
    for ix, row in port_navs.iterrows():
        if row['adjusted_close'] - max_nav < max_drawdown:
            max_drawdown = row['adjusted_close'] - max_nav
            max_drawdown_dt = row['timestamp']
        if row['adjusted_close'] > max_nav:
            max_nav = row['adjusted_close']
    print("PORTFOLIO MAX DRAWDOWN WAS: " + str(max_drawdown))
    print("PORTFOLIO MAX DRAWDON OCCURRED ON: " + str(max_drawdown_dt))
    print()
    
    # Volatility Calcs
    # calculate the average 1 day volatility over the history than scale it to a 1 year vol
    # thought about calcing for length of the history but decided to stick with one year
    rolling_window = 1
    # vol_term = np.sqrt(len(port_navs))
    vol_term = np.sqrt(252)
    # I know this isnt quite standard deviation but I had problems with the data naturally trending upward skewing the std
    # aka port_navs['adjusted_close'].std() gave exaggerated results
    # usually do a rolling vol like: port_navs['adjusted_close'].rolling(window=vol_window, center=False).std()
    # but that would ignore vol from certain parts of the history
    port_vol = abs(port_navs['adjusted_close'].pct_change(rolling_window)).mean() * vol_term * 100
    print("PORTFOLIO VOLATILITY (avg 1 yr move) OVER HISTORY: " + str(port_vol) + "%")
    
    # A little weird calculating this over a history and not a year, but considering the user can input the timeframe
    # the history may be less than a year so thought safer to do it this way
    sharpe_ratio = (port_ret - spy_ret) / (port_vol)
    print("PORTFOLIO SHARPE RATIO OVER HISTORY: " + str(sharpe_ratio))
    print()
    
    
    # Correlation, beta, sortino ration, treynor ratio
    # Calculating on returns, not values to avoid issue mentioned above with vol
    corr = correlation(port_navs['adjusted_close'].pct_change(rolling_window), spy_hist['adjusted_close'].pct_change(rolling_window))
    bt = beta(port_navs['adjusted_close'].pct_change(rolling_window), spy_hist['adjusted_close'].pct_change(rolling_window))
    print("PORTFOLIO CORRELATION TO SPY OVER HISTORY: " + str(corr))
    print("PORTFOLIO BETA TO SPY OVER HISTORY: " + str(bt))
    print("PORTFOLIO ALPHA TO SPY OVER HISTORY: " + str(port_ret - bt * spy_ret))
    print("PORTFOLIO TREYNOR RATIO TO SPY OVER HISTORY: " + str(((port_ret - spy_ret) / 100) / (bt)))
    print()
    port_navs['daily_chg'] = port_navs['adjusted_close'].pct_change(rolling_window)
    downside_vol = abs(port_navs[port_navs.daily_chg < 0]['daily_chg']).mean() * vol_term * 100
    print("PORTFOLIO DOWNSIDE VOL OVER HISTORY: " + str(downside_vol) + "%")
    print("PORTFOLIO SORTINO RATIO TO SPY OVER HISTORY: " + str((port_ret - spy_ret) / (downside_vol)))
    

def covariance(data1, data2):
    # measure of joint variation --> when both numbers move away from mean simultaneously, creates large values and vice versa
    mx = data1.mean()
    my = data2.mean()
    tot = 0
    for x, y in zip(data1, data2):
        tot += (x - mx) * (y - my)
    return tot / len(data1)


def correlation(data1, data2):
    # Pearson correlation coefficient. Will always between -1 and 1
    data1 = data1.dropna()
    data2 = data2.dropna()
    std1 = data1.std()
    std2 = data2.std()
    return covariance(data1, data2) / (std1 * std2)


def beta(dep_var, indep_var):
    # covariance(dependent var, independent var) / variance(independant var)
    dep_var = dep_var.dropna()
    indep_var = indep_var.dropna()
    cov = covariance(dep_var, indep_var)
    var = indep_var.var()
    return cov / var


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
    # tickers = ['MSFT']
    # weights = [1]
    for t, w in zip(tickers, weights):
        t_pos = Position(t, w)
        t_pos.grab_history(start_dt)
        positions.append(t_pos)
        
    run(positions, start_dt)
    