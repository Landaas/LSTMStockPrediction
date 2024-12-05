import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

look_back = 50 
batch_size=32
epochs=50

def create_sequences(dataset, look_back):
                X, y = [], []
                for i in range(len(dataset) - look_back):
                    X.append(dataset.iloc[i:(i + look_back)].values)
                    y.append(dataset.iloc[i + look_back]['Close'])
                return np.array(X), np.array(y)

def load_data_from_folder_and_train_model(folder_path, look_back, epochs, batch_size):
    
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)

            # Drop rows with any null values
            data.dropna(inplace=True)
           
            # Convert 'Date' to datetime and sort data
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date')
            data.reset_index(drop=True, inplace=True)

            features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

            # Scale only the remaining features
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data[features])
            scaled_data = pd.DataFrame(scaled_data, columns=features)
            
            # Create sequences for training
            X, y = create_sequences(scaled_data, look_back)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Train the model
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

            # Evaluate the model
            test_loss = model.evaluate(X_test, y_test)
            print(f'Test Loss: {test_loss}')
            

# Load all data from the data folder
data_folder_path = 'data'

model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(look_back, 6)),
    LSTM(50),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the model's architecture
model.summary()

load_data_from_folder_and_train_model(data_folder_path, look_back, epochs, batch_size)

#test the model
data = pd.read_csv('data/TOT.csv')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data.reset_index(drop=True, inplace=True)
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[features])
scaled_data = pd.DataFrame(scaled_data, columns=features)

X, y = create_sequences(scaled_data, look_back)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

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