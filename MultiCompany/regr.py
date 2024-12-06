import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Data path
path = 'MultiCompany/data'

def run_model(path):
    data = pd.read_csv(path)
    
    # Drop rows with missing values
    data.dropna(inplace=True)
    
    # Sort by date
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Features and traget
    features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
    target = 'Close'
    
    X = data[features]
    y = data[target]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    
    # Model
    model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    
    print(f"Model: Linear regression")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2 Score: {r2}")
    
    # Plotting actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual prices', color='blue')
    plt.plot(predictions, label='Predicted prices', color='red')
    plt.title('Actual vs predicted stock prices')
    plt.xlabel('Time')
    plt.ylabel('Close price')
    plt.legend()
    plt.show()

# Loop through all the stocks
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        file = os.path.join(path, filename)
        print(f"\nProcessing file: {filename}")
        run_model(file)
