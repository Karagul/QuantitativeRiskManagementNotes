#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 17:21:30 2019

@author: jan
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn')
import random
from statsmodels.stats import diagnostic as diag
import statsmodels.api as sm
random.seed(1)



def generate_garch_11_ts(n, sigma_sq_0, mu, alpha, beta, omega):
    """ generate GARCH log returns """
    nu = np.random.normal(0,1,n)
    r = np.zeros(n)
    epsilon = np.zeros(n)
    sigma_sq = np.zeros(n)
    sigma_sq[0] = sigma_sq_0
    
    if min(alpha,beta)<0:
        raise ValueError('alpha, beta need to be non-negative')
    if omega <=0:
        raise ValueError('omega needs to be positive')
        
    if alpha+beta>=1:
        print('alpha+beta>=1, variance not defined --> time series will not be weakly stationary')
        
    for i in range(n):
        
        if i >0:
            sigma_sq[i] = omega + alpha * epsilon[i-1]**2 + beta * sigma_sq[i-1]
            
        epsilon[i] = (sigma_sq[i]**0.5) * nu[i]

        r[i] = mu + epsilon[i]
    return r 
    
"""This simulation shows that the Ljung-Box test, which is an improved version of the Box-Pierce test, tests
 the null of independence and not as often claimed the null of no autocorrelation. E.g., if return volatilities 
 cluster, as has been observed in practice, the Ljung-Box test rejects the null of iid even if returns 
 are not autocorrelated. There is however a modification we can impose on the Box-Pierce test which then tests 
 the null of no autocorrelations. """

## GARCH
garch_returns = generate_garch_11_ts(250*15, 1e-5, 0.0, 0.5, 0.45, 1e-5)
price = np.exp(np.cumsum(garch_returns))


fig, axes = plt.subplots(2,1)
fig.tight_layout()
for ax, y, name in zip(axes, [price,garch_returns], ['GARCH Price Time Series',' GARCH Returns']):
    ax.plot(y)
    ax.set(title=name)
plt.show()

lb = diag.acorr_ljungbox(garch_returns,3)
print('GARCH Ljung-Box p_values:', lb[1])


## IID

iid_returns = generate_garch_11_ts(250*15, 1e-5, 0.0, 0, 0, 1e-5)
price = np.exp(np.cumsum(iid_returns))


fig, axes = plt.subplots(2,1)
fig.tight_layout()
for ax, y, name in zip(axes, [price,iid_returns], ['iid_returns Price Time Series',' iid  Returns']):
    ax.plot(y)
    ax.set(title=name)
plt.show()

iid_lb = diag.acorr_ljungbox(iid_returns,3)
print('iid returns Ljung-Box p_values:', iid_lb[1])


## Testing Null of no autocorrelation

def modified_box_pierce(x, lags):
    """
    Modification of the Box-Pierce test such that a large test statistic rejects 
    the null of no autocorrelations instead of iid"""
    x = pd.Series(x)
    rho_hat = pd.Series(sm.tsa.acf(x, nlags=lags))
    mu_hat = x.mean()
    delta_hat = []
    df = pd.DataFrame()
    df['t'] = x
    for lag in range(1, lags+1):
        df[f't+{lag}'] = df['t'].shift(-lag)
        delta_hat.append( ( (df['t']-mu_hat)**2*(df[f't+{lag}']-mu_hat)**2 ).dropna().sum()/ (x.var()**2) )
        
    q = len(x) * ((rho_hat[1:]/pd.Series(delta_hat))**2).sum()
    return q


mod_bp = modified_box_pierce(x=garch_returns, lags=3)
print('GARCH Modified Box-Pierce test stat:', mod_bp)



