import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import math
import datetime as dt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Input stock ticker symbol
stock = input("Enter the stock name (e.g. AAPL): ")
df = yf.download(stock, start="2010-04-01", end=dt.datetime.today())

# Debug prints to check data load and columns
print("Columns in dataframe:", df.columns)
print("First 5 rows of data:\n", df.head())

# Use safer column selection
if 'Close' not in df.columns:
    raise ValueError(f"'Close' column not found in data for stock {stock}")

data = df[['Close']]

print("Filtered 'Close' data shape:", data.shape)

if data.empty:
    raise ValueError(f"Filtered data is empty for stock {stock}")

dataset = data.values
training_data_len = math.ceil(len(dataset) * 0.8)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Prepare training data
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape input for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Prepare test data
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

# Reshape test data for LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict stock price
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print(f"Root Mean Square Error: {rmse}")

# Plot training data, validation data, and predictions
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Stock Price Prediction Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# Predict next day closing price
last_60_days = data[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

X_input = np.array([last_60_days_scaled])
X_input = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))

pred_price = model.predict(X_input)
pred_price = scaler.inverse_transform(pred_price)
print(f"Predicted next day Close price: {pred_price[0][0]}")
