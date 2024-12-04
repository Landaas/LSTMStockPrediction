import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('SingelCompany\TOT.csv')

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data = data.sort_values('Date')

# Reset the index
data.reset_index(drop=True, inplace=True)

# Select features
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Initialize the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the data
scaled_data = scaler.fit_transform(data[features])

# Convert scaled data back to a DataFrame
scaled_data = pd.DataFrame(scaled_data, columns=features)

# Define the look-back period (number of previous time steps to use)
look_back = 50  

# Function to create sequences
def create_sequences(dataset, look_back):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset.iloc[i:(i + look_back)].values)
        y.append(dataset.iloc[i + look_back]['Close'])
    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(scaled_data, look_back)

# Check the shape of the data
print(f'Input shape: {X.shape}')
print(f'Output shape: {y.shape}')

# Define the training data size
train_size = int(len(X) * 0.8)

# Split the data
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Check the shapes
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

# Initialize the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the model's architecture
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Make predictions on the testing data
predictions = model.predict(X_test)

# Create a new scaler for the 'Close' column
close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]

# Inverse transform the predicted and actual values
y_test_actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))
predictions_actual = close_scaler.inverse_transform(predictions)

# Plot the actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Close Price')
plt.plot(predictions_actual, label='Predicted Close Price')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('actual_vs_predicted_prices.png')

# Save the model
model.save('stock_price_prediction_model.keras')