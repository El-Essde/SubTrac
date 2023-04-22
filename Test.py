import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load traffic data
traffic_data = pd.read_csv('traffic_data.csv')

# Split data into training and testing sets
train_data = traffic_data.iloc[:1000, :]
test_data = traffic_data.iloc[1000:, :]

# Prepare training data
X_train = train_data[['traffic_volume', 'weather', 'time_of_day']].values
y_train = train_data['travel_time'].values

# Train a linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Use the model to predict travel time
X_test = test_data[['traffic_volume', 'weather', 'time_of_day']].values
y_pred = regressor.predict(X_test)

# Evaluate the model performance
mse = np.mean((y_pred - test_data['travel_time'].values)**2)
rmse = np.sqrt(mse)
print('Root mean squared error:', rmse)
