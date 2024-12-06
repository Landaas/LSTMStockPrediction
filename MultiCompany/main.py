import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from itertools import product


look_back = [10, 50, 200]
batch_size = [16, 32]
epochs = [5, 10]
lstm_values = [50, 100]
lstm_models = [0, 1]
dropout_values = [0,0.2]

# Load all data from the data folder
data_folder_path = 'data'

def create_sequences(dataset, look_back):
                X, y = [], []
                for i in range(len(dataset) - look_back):
                    X.append(dataset.iloc[i:(i + look_back)].values)
                    y.append(dataset.iloc[i + look_back]['Close'])
                return np.array(X), np.array(y)

def load_data_from_folder_and_train_model(folder_path, look_back, epochs, batch_size):  
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv') and filename:
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
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

            # Evaluate the model
            test_loss = model.evaluate(X_test, y_test)
            print(f'Test Loss: {test_loss}')
            
for look_back, batch_size, epochs, lstm_values, lstm_models, dropout_values in product(look_back, batch_size, epochs, lstm_values, lstm_models, dropout_values):
    # Create model
    model = Sequential()
    # First LSTM layer needs input_shape
    model.add(LSTM(lstm_values, return_sequences=True, input_shape=(look_back, 6)))
    model.add(Dropout(dropout_values))

    # Additional LSTM layers from the loop
    for _ in range(lstm_models):  # Start from 1 since the first layer is added above
        model.add(LSTM(lstm_values, return_sequences=True))
        model.add(Dropout(dropout_values))

    model.add(LSTM(50))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Display the model's architecture
    model.summary()

    load_data_from_folder_and_train_model(data_folder_path, look_back, epochs, batch_size)

    #test the model
    data = pd.read_csv('TOT.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    data.reset_index(drop=True, inplace=True)
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])
    scaled_data = pd.DataFrame(scaled_data, columns=features)

    X_test, y_test = create_sequences(scaled_data, look_back)

    # Make predictions on the testing data
    predictions = model.predict(X_test)

    # Create a new scaler for the 'Close' column
    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]

    # Inverse transform the predicted and actual values
    y_test_actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_actual = close_scaler.inverse_transform(predictions)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    mse = mean_squared_error(y_test_actual, predictions_actual)
    mape = np.mean(np.abs((y_test_actual - predictions_actual) / y_test_actual)) * 100
    r2 = r2_score(y_test_actual, predictions_actual)


    # Plot the actual vs predicted prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Close Price')
    plt.plot(predictions_actual, label='Predicted Close Price')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plot_filename = f'actual_vs_predicted_prices,{look_back},{batch_size},{epochs},{lstm_values},{lstm_models},{dropout_values}.png'
    plt.savefig(os.path.join('graphs', plot_filename))

    # Save the evaluation results to a CSV file
    results_file = 'evaluation_results.csv'
    # Check if the results file exists; if not, create it
    if not os.path.isfile(results_file):
        with open(results_file, 'w') as f:
            f.write('Mean squared error,Mean Absolute Error,Root Mean Squared Error,Mean Absolute Percentage Error,R2,Look back,Batch size,Epochs,LSTM values,LSTM models,Dropout values\n')

    # Append the evaluation results to the CSV file
    with open(results_file, 'a') as f:
        f.write(f'{mse},{mae},{rmse},{mape},{r2},{look_back},{batch_size},{epochs},{lstm_values},{lstm_models},{dropout_values}\n')

    # Save the model
    model_filename = f'stock_price_prediction_model,{look_back},{batch_size},{epochs},{lstm_values},{lstm_models},{dropout_values}.keras'
    model.save(os.path.join('models', model_filename))