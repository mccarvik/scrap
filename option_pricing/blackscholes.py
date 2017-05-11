import numpy as np
import scipy.stats as ss
import time 

#Black and Scholes
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
 
def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
 
def BlackScholes(type,S0, K, r, sigma, T):
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
       return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))
       

S0 = 100.0
K = 130.0
r=0.1
sigma = 0.30
T = 1
Otype='C'



print("S0\tstock price at time 0:", S0)
print("K\tstrike price:", K)
print("r\tcontinuously compounded risk-free rate:", r)
print("sigma\tvolatility of the stock price per year:", sigma)
print("T\ttime to maturity in trading years:", T)


t=time.time()
c_BS = BlackScholes(Otype,S0, K, r, sigma, T)
elapsed=time.time()-t
print("c_BS\tBlack-Scholes price:", c_BS)
print("\nTime to calc: ", elapsed)