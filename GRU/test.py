import torch
from dataset import test_loader
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from load import scaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('models/test.pth')

model.eval()
predictions = []
true_values = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)  
        y_batch = y_batch.to(device)
        
        outputs = model(X_batch) 
        
        predictions.append(outputs.squeeze().cpu().numpy())
        true_values.append(y_batch.cpu().numpy())

predictions = np.concatenate(predictions)
true_values = np.concatenate(true_values)

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

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(true_values_inverted, label='True Values')
plt.plot(predictions_inverted, label='Predictions')
plt.title('Model Predictions vs True Values on Test Set')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.show()