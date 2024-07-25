import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import networkx as nx
from typing import Dict, List
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests


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

def run_granger_causality_tests(df: pd.DataFrame, tokens: list, y_values: list, x_values: list, maturities: list) -> dict:
    # Dictionary to store results
    causality_results = {}

    # Loop through each token
    for token in tokens:
        causality_results[token] = {}
        
        # Loop through each maturity for the token
        for maturity in maturities:
            maturity_data = df[(df['token'] == token) & (df['maturity'] == maturity)].dropna()
            causality_results[token][maturity] = {}

            # Loop through each y_value
            for y in y_values:
                causality_results[token][maturity][y] = {}

                # Check each x variable for Granger causality on y
                for x in x_values:
                    if x != y:  # Avoid testing a variable on itself
                        test_data = maturity_data[[y, x]].dropna()  # Ensure data has no NAs

                        # Check for constant values
                        if test_data[x].nunique() <= 1 or test_data[y].nunique() <= 1:
                            print(f"Skipping {x} -> {y} due to constant values in data for token {token} at maturity {maturity}.")
                            continue

                        try:
                            result = grangercausalitytests(test_data, maxlag=10, verbose=False)  # Run Granger Test
                            p_values = [result[lag][0]['ssr_ftest'][1] for lag in range(1, 11)]
                            causality_results[token][maturity][y][x] = p_values
                        except Exception as e:
                            print(f"Error in testing {x} -> {y} for token {token} at maturity {maturity}: {str(e)}")

    return causality_results

def plot_granger_causality_network(results, significance_level=0.05, layout_type='spring'):
    """
    Visualizes a network of Granger causality relationships based on p-values, including maturities.

    Parameters:
        results (dict): A nested dictionary containing tokens, maturities, variables, and p-values for various lags.
                        Structure: {token: {maturity: {dependent_var: {independent_var: [p-values]}}}}
        significance_level (float): The threshold below which a p-value is considered significant.
        layout_type (str): Type of network layout ('shell', 'spring', 'circular').

    Returns:
        None: This function plots a network graph.
    """
    G = nx.DiGraph()

    for token, maturities in results.items():
        for maturity, maturity_data in maturities.items():
            for y_var, causations in maturity_data.items():
                for x_var, p_values in causations.items():
                    for lag, p_value in enumerate(p_values, start=1):
                        if isinstance(p_value, str):
                            p_value = float(p_value)  # Convert string to float if necessary
                        if p_value < significance_level:
                            source = f"{x_var} (Lag {lag})"
                            target = f"{y_var} ({token}, {maturity})"
                            G.add_edge(source, target, weight=p_value, label=f"{p_value:.4f}")

    if layout_type == 'shell':
        pos = nx.shell_layout(G)
    elif layout_type == 'spring':
        pos = nx.spring_layout(G)
    elif layout_type == 'circular':
        pos = nx.circular_layout(G)
    else:
        raise ValueError("Unsupported layout type. Choose 'shell', 'spring', or 'circular'.")

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=5000, alpha=0.6)
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', node_size=5000)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green')

    plt.title('Network of Granger Causality by Token, Maturity, and Variable')
    plt.axis('off')
    plt.show()
    
# Function to create correlation matrix and heatmap
def create_correlation_heatmap(df: pd.DataFrame, token: str) -> None:
    corr_matrix = df.corr()
    
    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title(f'Correlation Heatmap for Token: {token}')
    plt.show()

def get_maturity(month_year: pd.Series) -> str:
    # Define a mapping from month abbreviations to numbers
    month_map = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
    }
    
    # Extract the month and year parts from the input
    month = month_year.str[:3].str.lower().map(month_map)
    year = '20' + month_year.str[-2:]
    
    # Combine month and year into MM-YYYY format
    return month + '-' + year

# Function to format the data based on the token type
def format_data(df: pd.DataFrame, token_type: str) -> pd.DataFrame:
    if token_type in ['pt', 'yt']:
        df['token'] = df['source_file'].str[3:-7]
        df = df.loc[:, ['time', 'volume', 'value', 'maturity', 'token']].copy()
        df.rename(columns={'value': f'{token_type}_price'}, inplace=True)
    elif token_type == 'imp_apy':
        df['token'] = df['source_file'].str[8:-7]
        df = df.loc[:, ['time', 'underlyingApy', 'impliedApy', 'maturity', 'token']].copy()
    return df
