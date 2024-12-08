import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from load import create_sequences
import matplotlib.pyplot as plt
from glob import glob
import os

def mape(y_true, y_pred):

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('TOT.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data.reset_index(drop=True, inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])
scaled_data = pd.DataFrame(scaled_data, columns=features)

X_test, y_test = create_sequences(scaled_data, target='Close')
X_test_tensor = torch.from_numpy(np.array(X_test)).float().to(device)

num_features = len(['Open','High','Low','Close','Volume'])
close_col_idx = 3

model_files = glob("models/*.pth")

for model_path in model_files:
    model = torch.load(model_path, map_location=device)
    model.eval()

    with torch.no_grad():
        outputs = model(X_test_tensor)

    predictions = outputs.cpu().numpy()
    predictions = np.concatenate(predictions)
    true_values = y_test

    temp = np.zeros((len(predictions), num_features))
    temp[:, close_col_idx] = predictions
    temp_inverted = scaler.inverse_transform(temp)
    predictions_inverted = temp_inverted[:, close_col_idx]

    temp_true = np.zeros((len(true_values), num_features))
    temp_true[:, close_col_idx] = true_values
    temp_true_inverted = scaler.inverse_transform(temp_true)
    true_values_inverted = temp_true_inverted[:, close_col_idx]

    mape_metric = mape(true_values_inverted, predictions_inverted)

    print(f"Model: {os.path.basename(model_path)}, MAPE: {mape_metric:.5f}%")
