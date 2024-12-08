import pandas as pd
from glob import glob
from sklearn.preprocessing import MinMaxScaler

import numpy as np

features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = 'Target'
pre_days = 50

file_paths = glob("subset/*.csv")
dataframes = []
for fp in file_paths:
    df = pd.read_csv(fp, parse_dates=['Date'], index_col='Date')
    df.dropna(inplace=True)
    df = df[features]
    df['Target'] = df['Close'].shift(-1)
    dataframes.append(df)


combined_data = []
for fp, df in zip(file_paths, dataframes):
    split = fp.split('/')
    if len(split) < 2:
        split = fp.split('\\')
    ticker = split[-1].replace('.csv', '')
    df['Ticker'] = ticker
    df.dropna(inplace=True)
    combined_data.append(df)

full_dataset = pd.concat(combined_data, axis=0)

scaler = MinMaxScaler(feature_range=(0, 1))

full_dataset[features + ['Target']] = scaler.fit_transform(full_dataset[features + ['Target']])

def create_sequences(data, features=features, target='Target', window_size=50):
    X, y = [], []
    for i in range(len(data)-window_size):
        X.append(data[features].iloc[i:i+window_size].values)
        y.append(data[target].iloc[i+window_size])
    return np.array(X), np.array(y)

X, y = create_sequences(full_dataset, features, target, pre_days)
