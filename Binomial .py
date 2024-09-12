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


# Read data
df = yf.download(symbol,start,end)

df.tail()

symbol = '^SPX'  #the ticker of the stock whose data is to be taken out
dfo = yf.Ticker(symbol)

#calculation of up move and down move factor
dfo_exp = dfo.option_chain(dfo.options[1])  #you can change index to try on different expiry dates

dfo_exp.calls.head() #Data for call options

#Calculation of daily returns
ret=df.pct_change().mul(100)
ret= ret.dropna()

daily_vol = ret["Close"].std()
vol = daily_vol*252**0.5

N = 100
T = 8.0/365      #Calculate the number of days left for expiry from your calculation in years
t = T/N

u = np.exp(vol*(t**0.5))
d = 1/u

#Binomial Pricing for call options
def first_binomial_call(S, K, T, r, u, d, N):
    """
    S:float stock price
    K:float strike price
    T:float expiry time in years
    r:float risk free rate
    u:float size of upfactor move
    d:float size of downfactor move
    N:int number of steps in binomial model
    C:dict the binary pricing model in the form of dictionary
    """
    C={}
    for i  in range(N+1):
      C[N,i]=max(S*(u**i)*(d**(N-i))-K,0)
    p = (np.exp(r*T)-d)/(u-d)
    for i  in range(N-1,0,-1):
      for j in range(i+1) :
        C[i,j]=np.exp(r*T)*(p*C[i+1,j+1]+(1-p)*C[i+1,j])
    return C

#call price calculated by n step binomial model
call = {}
for K in dfo_exp.calls['strike']:
    call_price = first_binomial_call(S = df['Close'][len(df)-1], K = K, T = T, r=0.01*t, u = u, d =d, N=N)[(1,1)]
  # print(call_price)
    call[K] = call_price
th_call = pd.DataFrame.from_dict(call, orient='index')
th_call.rename(columns = {0:"th_call"}, inplace = True)

#price of actual calls
ac_call = dfo_exp.calls.loc[:, ['strike', 'lastPrice']]
ac_call.set_index('strike', inplace = True)
ac_call.rename(columns = {"lastPrice":"ac_call"}, inplace=True)

call = th_call
call["ac_call"] = ac_call

call.plot() #plotting the actual and theoretical option prices