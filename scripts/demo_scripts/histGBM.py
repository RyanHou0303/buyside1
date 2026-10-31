import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

np.random.seed(42)
T,p = 1500,50

eps = np.random.randn(T,p)*0.02
sigma = np.abs(np.rando.randn(T))
g = np.where(sigma>np.median(sigma),1.0,-1.0)

r = eps.copy()
for t in range(1,T):
    r[t,1]+=g[t]*(0.8*r[t-1,0])