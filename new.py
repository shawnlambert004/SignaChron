from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:*****@localhost/signachronodata')
data = pd.read_sql_table("S&P500_price_history",con=engine)

aapl = data[data['Symbol'] == 'AAPL']
#initialise Data Visualisation
plt.figure(figsize=(12,6))
plt.plot(aapl['Date'], aapl['Open'], label="Open", color="blue")
plt.plot(aapl['Date'], aapl['Close'], label="Close", color="green")
plt.title("Open-Close Price overTime")
plt.legend()
plt.show()

#Trading Vol
plt.figure(figsize=(12,6))
plt.plot(aapl['Date'], aapl['Volume'], label="Volume", color="red")
plt.title("Volume overTime")
plt.show()

numeric_data = aapl.select_dtypes(include=['Int64', 'Float64'])

#LSTM model
stock_close_price = aapl['Close']
dataset = stock_close_price.values
dataset_len = int(np.ceil(len(dataset)*0.8))


#preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset.reshape(-1, 1))

training_data = scaled_data[:dataset_len]

x_train, y_train = [], []

for i in range(5, dataset_len):
    x_train.append(training_data[i-5:i, 0])
    y_train.append(training_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = keras.models.Sequential()

model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1],1)))

model.add(keras.layers.LSTM(64, return_sequences=False))

model.add(keras.layers.Dense(128, activation="relu"))

model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer="adam",
              loss="mae",
              metrics=[keras.metrics.RootMeanSquaredError()])

training = model.fit(x_train, y_train, epochs=20, batch_size=32)

#test data prep
test_data = scaled_data[dataset_len-60:]
x_test, y_test = [], dataset[dataset_len:]

print("test_data shape:", test_data.shape)
print("len(test_data):", len(test_data))
for i in range(5, len(test_data)):
    x_test.append(test_data[i-5:i, 0]) 

print("sample num",x_test)
print("Shape of One Sample:", np.array(x_test[0]).shape)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

train = aapl[:dataset_len]
test = aapl[dataset_len:]
test= test.copy()
test = test.iloc[-len(predictions):]
test['Predictions'] = predictions

plt.figure(figsize=(12,8))
plt.plot(train['Date'], train['Close'], label="Train (Actual)", color='blue')
plt.plot(test['Date'], test['Close'], label="Test (Actual)", color='orange')
plt.plot(test['Date'], test['Predictions'], label="Predictions", color='red')
plt.title("Stock Predictions")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()