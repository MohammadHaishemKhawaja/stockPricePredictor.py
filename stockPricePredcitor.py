import numpy
import matplotlib.pyplot
import pandas
import pandas_datareader
import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

company = 'AAPL'

start = datetime.datetime(2020,7,1)
end = datetime.datetime.now()

data = pandas_datareader.DataReader(company, 'yahoo', start, end)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1,1))

days = 60

x_train = []
y_train = []

for x in range(days, len(scaled_data)):
    x_train.append(scaled_data[x-days: x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = numpy.array(x_train), numpy.array(y_train)
x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)


test_start = datetime.datetime(2020,1,1)
test_end = datetime.datetime.now()

test_data = pandas_datareader.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pandas.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)


x_test = []

for x in range(days, len(model_inputs)):
    x_test.append(model_inputs[x-days:x, 0])

x_test = numpy.array(x_test)
x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

matplotlib.pyplot.plot(actual_prices, color="black", label=f"Actual {company} Price")
matplotlib.pyplot.plot(predicted_prices, color='green', label=f"Preicted {company} Price")
matplotlib.pyplot.title(f"{company} Share Price")
matplotlib.pyplot.xlabel('Time')
matplotlib.pyplot.ylabel(f'{company} Share Price')
matplotlib.pyplot.legend()
matplotlib.pyplot.show()

real_data = [model_inputs[len(model_inputs) + 1 - days:len(model_inputs), 0]]
real_data = numpy.array(real_data)
real_data = numpy.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print (f"Prediction: {prediction}")
