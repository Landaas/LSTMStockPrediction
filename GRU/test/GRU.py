import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import matplotlib.pyplot as plt

# Function to create sequences for time-series data
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, 3])  # Use the 'Close' column as the target
    return np.array(X), np.array(y)

# GRU-based stock prediction model
def build_gru_model(look_back, input_dim):
    model = Sequential()
    model.add(GRU(units=64, return_sequences=True, input_shape=(look_back, input_dim)))
    model.add(Dropout(0.2))
    model.add(GRU(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for single prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main function for stock prediction
def stock_prediction(file_path, look_back, epochs, batch_size, future_days):
    # Load and preprocess data
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)

    # Select features and normalize
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    
    # Create sequences
    X, y = create_sequences(scaled_data, look_back)
    
    # Split data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train the GRU model
    model = build_gru_model(look_back, X.shape[2])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    
    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss}")
    
    # Predict future prices
    predictions = model.predict(X_test)
    future_features = scaled_data[-look_back:]
    future_features = np.expand_dims(future_features, axis=0)
    future_prediction = model.predict(future_features).flatten()
    
    # Inverse scale predictions and actual values
    y_test_actual = scaler.inverse_transform(np.concatenate((X_test[:, 0, :-1], y_test.reshape(-1, 1)), axis=1))[:, 3]
    predictions_actual = scaler.inverse_transform(np.concatenate((X_test[:, 0, :-1], predictions), axis=1))[:, 3]
    future_prediction_actual = scaler.inverse_transform(
        np.concatenate((np.zeros((1, 5)), future_prediction.reshape(-1, 1)), axis=1)
    )[:, 3]
    
    # Print future prediction
    print(f"Predicted price {future_days} days ahead: {future_prediction_actual[0]}")

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label='Actual Prices', color='blue')
    plt.plot(predictions_actual, label='Predicted Prices', color='red', linestyle='dashed')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# Example usage
path = 'MultiCompany/data'
look_back = 200  # Number of days to look back
epochs = 1
batch_size = 32
future_days = 15

for filename in os.listdir(path):
    if filename.endswith('.csv'):
        file_path = os.path.join(path, filename)
        print(f"\nProcessing file: {filename}")
        stock_prediction(file_path, look_back, epochs, batch_size, future_days)
