import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from load import create_sequences, scaler
from torchmetrics.regression import MeanAbsolutePercentageError

def mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) between two lists.
    
    Parameters:
        y_true (list or array-like): Ground truth (correct) values
        y_pred (list or array-like): Predicted values
        
    Returns:
        float: MAPE value in percentage
    """
    # Convert inputs to numpy arrays for vectorized operations
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    epsilon = 1e-8  # Small constant to prevent division by zero
    # Compute MAPE
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100



import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = torch.load('models/test_epochs=50 (1).pth', map_location=device)
model.eval()

data = pd.read_csv('data/TOT.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data.reset_index(drop=True, inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])
scaled_data = pd.DataFrame(scaled_data, columns=features)

# Create sequences
X_test, y_test = create_sequences(scaled_data, target='Close')

train_size = int(len(X_test) * 0.8)
X_test = X_test[train_size:]
y_test = y_test[train_size:]

# Convert X_test to a PyTorch tensor
X_test_tensor = torch.from_numpy(np.array(X_test)).float().to(device)

# Make predictions on the testing data
with torch.no_grad():
    outputs = model(X_test_tensor)


# Convert predictions to NumPy array
predictions = outputs.cpu().numpy()

predictions = np.concatenate(predictions)
true_values = y_test



num_features = len(['Open','High','Low','Close','Volume'])
close_col_idx = 3  

temp = np.zeros((len(predictions), num_features))
temp[:, close_col_idx] = predictions

# Invert transform
temp_inverted = scaler.inverse_transform(temp)
predictions_inverted = temp_inverted[:, close_col_idx]

# For true_values, do the same:
temp_true = np.zeros((len(true_values), num_features))
temp_true[:, close_col_idx] = true_values
temp_true_inverted = scaler.inverse_transform(temp_true)
true_values_inverted = temp_true_inverted[:, close_col_idx]

mape_metric = mape(true_values_inverted, predictions_inverted)

print('MAPE:', mape_metric)

plt.figure(figsize=(12,6))
plt.plot(true_values_inverted, label='Actual Close Price')
plt.plot(predictions_inverted, label='Predicted Close Price')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('actual_vs_predicted_prices.png')
plt.show()