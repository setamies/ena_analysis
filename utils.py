import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import requests
from datetime import datetime
import matplotlib.pyplot as plt

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
        
        
def fetch_futures_funding_rates(symbol, start_date, end_date, limit=1000):
    """Fetches futures funding rates from Binance API."""
    base_url = 'https://fapi.binance.com/fapi/v1/fundingRate'
    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)  # convert to milliseconds
    end_time = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)  # convert to milliseconds

    params = {
        'symbol': symbol,
        'limit': limit,
        'startTime': start_time,
        'endTime': end_time
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()  # Raise an exception for HTTP errors

    data = response.json()
    return pd.DataFrame(data)

def process_funding_data(funding_data):
    """Processes the funding data to calculate daily APY and 7-day moving average."""
    funding_data['fundingTime'] = pd.to_datetime(funding_data['fundingTime'], unit='ms', utc=True)
    funding_data['fundingRate'] = funding_data['fundingRate'].astype(float)

    daily_apy = funding_data.groupby(funding_data['fundingTime'].dt.date)['fundingRate'].mean() * 3 * 100  # Convert rate to percentage
    daily_apy = daily_apy.reset_index()
    daily_apy.columns = ['Datetime', 'Daily Funding Rate']
    daily_apy['Datetime'] = pd.to_datetime(daily_apy['Datetime'], utc=True)
    daily_apy['Funding 7-Day MA'] = daily_apy['Daily Funding Rate'].rolling(window=7).mean()
    
    return daily_apy

def plot_daily_apy(daily_apy):
    """Plots the daily APY and its 7-day moving average."""
    daily_apy.set_index('Datetime')['Daily Funding Rate'].plot(kind='line', figsize=(12, 6), title='Daily Funding Rate')
    daily_apy.set_index('Datetime')['Funding 7-Day MA'].plot(kind='line', figsize=(12, 6), title='7-Day Moving Average of Daily Funding Rate')
    plt.ylabel('Funding Rate (%)')
    plt.xlabel('Date')
    plt.legend(['Daily Funding Rate', '7-Day Moving Average'])
    plt.show()
    
def to_datetime(df, columns):
    for col in columns:
        df[col] = pd.to_datetime(df[col])
    return df

def shift_data(df, columns, periods=-1):
    for col in columns:
        df[f'Target_{col}'] = df[col].shift(periods)
        df[f'{col}_Change'] = df[col].diff()
        df[f'Target_{col}_Change'] = df[f'Target_{col}'].diff()
    return df

def prepare_circulating_supply(supply_dict):
    df = pd.DataFrame.from_dict(supply_dict, orient='index', columns=['Circulating Supply'])
    df.index = pd.to_datetime(df.index)
    df = df.resample('D').bfill()
    df.index.name = 'Datetime'
    df.reset_index(inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    return df

def merge_supply(df, supply_df, stake_column):
    df = df.merge(supply_df, how='left', on='Datetime')
    df['Liquid Circulating Supply'] = df['Circulating Supply'] - df[stake_column]
    df['Liquid Circulating Supply Change'] = df['Liquid Circulating Supply'].diff()
    return df

def mean_directional_accuracy(actual, predicted):
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))

# Define a function to create sequences from the data
def create_sequences(data, sequence_length=1):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, :-1])
        y.append(data[i+sequence_length, -1])
    return np.array(X), np.array(y)