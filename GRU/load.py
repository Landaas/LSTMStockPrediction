import pandas as pd
from glob import glob
from sklearn.preprocessing import MinMaxScaler

import numpy as np

features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = 'Target'
pre_days = 50

# Suppose your CSV files are in a directory called 'data'
file_paths = glob("subset/*.csv")  # Get all csv files in data directory
dataframes = []
for fp in file_paths:
    df = pd.read_csv(fp, parse_dates=['Date'], index_col='Date')
    df = df[features]
    df['Target'] = df['Close'].shift(-1)
    dataframes.append(df)


combined_data = []
for fp, df in zip(file_paths, dataframes):
    ticker = fp.split('/')[-1].replace('.csv', '')
    df['Ticker'] = ticker
    df.dropna(inplace=True)
    combined_data.append(df)

for i in range(len(dataframes)):
    dataframes[i] = dataframes[i].ffill().bfill()  # Simple fill strategy

full_dataset = pd.concat(combined_data, axis=0)
# Now full_dataset has a 'Ticker' column indicating the stock.

scaler = MinMaxScaler(feature_range=(0, 1))
# For separate stocks:
# dataframes[i][['Open','High','Low','Close','Volume']] = scaler.fit_transform(dataframes[i][['Open','High','Low','Close','Volume']]]

# For combined:
full_dataset[features] = scaler.fit_transform(full_dataset[features])

def create_sequences(data, features, target, window_size=30):
    X, y = [], []
    for i in range(len(data)-window_size):
        X.append(data[features].iloc[i:i+window_size].values)
        y.append(data[target].iloc[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(full_dataset, features, target, pre_days)
