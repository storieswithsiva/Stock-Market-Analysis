
# coding: utf-8

# In[3]:

# Importing all the essential Python libraries

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
sns.set_style('whitegrid')


# In[4]:

# Importing Stock market data from the Internet

from pandas.io.data import DataReader


# In[5]:

# Importing datetime for setting start and end date of the stock market dataset

from datetime import datetime


# In[6]:

# Setting the Start and End date for Stock Market Analysis

end = datetime.now()
start = datetime(end.year-1,end.month,end.day)


# In[7]:

# Importing Google Stock Prices

GOOG = DataReader('GOOG','yahoo',start,end)


# In[8]:

# Some Basic info about the Google Stock

GOOG.describe()


# In[9]:

# Plotting Adjusted Closing price for Google Stock

GOOG['Adj Close'].plot(legend=True,figsize=(10,4))


# In[10]:

# Total volume of stock being traded each day

GOOG['Volume'].plot(legend=True,figsize=(10,4))


# In[11]:

# Calculating Moving average for 10, 20 and 50 days of the stock price

ma_day = [10,20,50]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    
    GOOG[column_name] = pd.rolling_mean(GOOG['Adj Close'],ma)


# In[12]:

# Plotting the moving averages

GOOG[['Adj Close', 'MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(10,4))


# In[13]:

# Plotting Daily returns as a function of Percent change in Adjusted Close value

GOOG['Daily Return'] = GOOG['Adj Close'].pct_change()

GOOG['Daily Return'].plot(legend=True)


# In[14]:

# Plotting the average daily returns of the stock

sns.distplot(GOOG['Daily Return'].dropna(),bins=100)


# In[15]:

# Risk Analysis --  Comparing the Risk vs Expected returns

rets = GOOG['Daily Return'].dropna()

area = np.pi*15

plt.scatter(rets.mean(),rets.std(),s=area)

plt.xlabel('Expected Returns')
plt.ylabel('Risk')


# In[16]:

# Visualizing the Value at Risk

sns.distplot(GOOG['Daily Return'].dropna(),bins=100)


# In[17]:

# Using Quantiles and the Bootstrap Method to calculate the numerical risk of the stock

GOOG['Daily Return'].quantile(0.05)


# In[18]:

## Monte Carlo Simulation

days = 365

dt = 1/days

mu = rets.mean()

sigma = rets.std()


# In[19]:

# Defining the Monte Carlo Simulation Function

def stock_monte_carlo(start_price,days,mu,sigma):
    
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1,days):
        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        
        drift[x] = mu * dt
        
        price[x] = price[x-1] + (price[x-1]* (drift[x] + shock[x]))
        
    return price


# In[20]:

GOOG.head()


# In[21]:

# Running the Monte Carlo simulation a hundred times

start_price = 532.192446

for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
    
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Simulation for Google')


# In[22]:

# Analysing the Monte Carlo Simulation for 10,000 simulations

runs = 10000

simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]
    
# 1 percent impirical quantile or 99% Confidence Interval

q = np.percentile(simulations,1)


# In[23]:

# Plotting the final Risk Analysis plot using Monte Carlo Simulation

plt.hist(simulations,bins=200)

plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



