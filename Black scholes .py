import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import warnings
warnings.filterwarnings("ignore")

# to fetch data
import yfinance as yf
import scipy.stats as ss

symbol = '^SPX'
end = date.today()
start = end.replace(year = 2022)

tickerData = yf.Ticker(symbol)

# Read data
df = tickerData.history(period='1d', start=start, end=end, interval = '1d')

symbol = '^SPX'  #the ticker of the stock whose data is to be taken out
dfo = yf.Ticker(symbol)

dfo.options  #to get the date of different expiration time

dfo_exp = dfo.option_chain(dfo.options[1])  #you can change index to try on different expiry dates


def annual_vol(df):
    
    log_return = np.log(df.Close).diff()
    log_return = log_return.dropna()
    daily_vol = log_return.std()
    vol = daily_vol*(252)**0.5

    return vol

vol = annual_vol(df)


def euro_vanilla(S, K, T, r, sigma, option = 'call'):

    d1=(np.log(S/K)+(r+(sigma**2)/2)*T)/(sigma*T**0.5)
    d2=d1-(sigma*T**0.5)
    premium =S*ss.norm.cdf(d1)-K*np.exp(-1*r*T)*ss.norm.cdf(d2)
    return premium


N = 100
T = 8.0/365      #Calculate the number of days left for expiry from your calculation in years
t = T/N


#r is the risk free rate taken from the 10 years us treasury bond
#call price calculated from black scholes model
call = {}
for K in dfo_exp.calls['strike']:
    call_price = euro_vanilla(S = df['Close'][len(df)-1], K = K, T = T, r=0.0123*t, sigma=vol)
  # print(call_price)
    call[K] = call_price
th_call = pd.DataFrame.from_dict(call, orient='index')
th_call.rename(columns = {0:"th_call"}, inplace = True)


#actual call price
ac_call = dfo_exp.calls.loc[:, ['strike', 'lastPrice']]
ac_call.set_index('strike', inplace = True)
ac_call.rename(columns = {"lastPrice":"ac_call"}, inplace=True)


call = th_call
call["ac_call"] = ac_call


call.plot() #plotting the actual and theoretical call prices

#r is the risk free rate taken from the 10 years us treasury bond
#theoretical put price calculated from black scholes model
put = {}
for K in dfo_exp.puts['strike']:
    put_price = euro_vanilla(S = df['Close'][len(df)-1], K = K, T = T, r=0.0158*t, sigma=vol, option = 'put')
  # print(put_price)
    put[K] = put_price
th_put = pd.DataFrame.from_dict(put, orient='index')
th_put.rename(columns = {0:"th_put"}, inplace = True)


#actual put price
ac_put = dfo_exp.puts.loc[:, ['strike', 'lastPrice']]
ac_put.set_index('strike', inplace = True)
ac_put.rename(columns = {"lastPrice":"ac_put"}, inplace=True)


put = th_put
put["ac_put"] = ac_put


put.plot()