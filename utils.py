import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np
from flipside import Flipside

# Function to perform the Augmented Dickey-Fuller test
def check_stationarity(series, title='Your Data'):
    result = adfuller(series.dropna())  
    print(f'Results of Dickey-Fuller Test for {title}:')
    print(f'Test Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    if result[1] < 0.05:
        print("Evidence against the null hypothesis, data is stationary")
    else:
        print("Weak evidence against null hypothesis, time series is non-stationary")