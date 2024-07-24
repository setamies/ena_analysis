import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def check_stationarity(series: pd.Series, title: str = 'Your Data') -> None:
    """Perform the Augmented Dickey-Fuller test to check stationarity."""
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

def fetch_futures_funding_rates(symbol: str, start_date: str, end_date: str, limit: int = 1000) -> pd.DataFrame:
    """Fetch futures funding rates from Binance API."""
    base_url = 'https://fapi.binance.com/fapi/v1/fundingRate'
    start_time = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_time = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    params = {
        'symbol': symbol,
        'limit': limit,
        'startTime': start_time,
        'endTime': end_time
    }
    
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    data = response.json()
    return pd.DataFrame(data)

def process_funding_data(funding_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily APY and 7-day moving average from funding data."""
    funding_data['fundingTime'] = pd.to_datetime(funding_data['fundingTime'], unit='ms', utc=True)
    funding_data['fundingRate'] = funding_data['fundingRate'].astype(float)
    
    daily_apy = (funding_data.groupby(funding_data['fundingTime'].dt.date)['fundingRate']
                 .mean() * 3 * 100).reset_index()
    daily_apy.columns = ['Datetime', 'Daily Funding Rate']
    daily_apy['Datetime'] = pd.to_datetime(daily_apy['Datetime'], utc=True)
    daily_apy['Funding 7-Day MA'] = daily_apy['Daily Funding Rate'].rolling(window=7).mean()
    
    return daily_apy

def plot_daily_apy(daily_apy: pd.DataFrame) -> None:
    """Plot daily APY and its 7-day moving average."""
    plt.figure(figsize=(12, 6))
    plt.plot(daily_apy['Datetime'], daily_apy['Daily Funding Rate'], label='Daily Funding Rate')
    plt.plot(daily_apy['Datetime'], daily_apy['Funding 7-Day MA'], label='7-Day Moving Average')
    plt.title('Daily Funding Rate')
    plt.ylabel('Funding Rate (%)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()

def to_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to datetime."""
    for col in columns:
        df[col] = pd.to_datetime(df[col])
    return df

def shift_data(df: pd.DataFrame, columns: List[str], periods: int = -1) -> pd.DataFrame:
    """Shift specified columns by a given number of periods and calculate changes."""
    for col in columns:
        df[f'Target_{col}'] = df[col].shift(periods)
        df[f'{col}_Change'] = df[col].diff()
        df[f'Target_{col}_Change'] = df[f'Target_{col}'].diff()
    return df

def prepare_circulating_supply(supply_dict: Dict[str, Any]) -> pd.DataFrame:
    """Prepare circulating supply data for merging."""
    df = pd.DataFrame.from_dict(supply_dict, orient='index', columns=['Circulating Supply'])
    df.index = pd.to_datetime(df.index)
    df = df.resample('D').bfill()
    df.index.name = 'Datetime'
    df.reset_index(inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    return df

def merge_supply(df: pd.DataFrame, supply_df: pd.DataFrame, stake_column: str) -> pd.DataFrame:
    """Merge circulating supply data and calculate liquid supply."""
    df = df.merge(supply_df, how='left', on='Datetime')
    df['Liquid Circulating Supply'] = df['Circulating Supply'] - df[stake_column]
    df['Liquid Circulating Supply Change'] = df['Liquid Circulating Supply'].diff()
    return df

def mean_directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate mean directional accuracy."""
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))

def create_sequences(data: np.ndarray, sequence_length: int = 1) -> (np.ndarray, np.ndarray):
    """Create sequences of data for time series analysis."""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :-1])
        y.append(data[i + sequence_length, -1])
    return np.array(X), np.array(y)
