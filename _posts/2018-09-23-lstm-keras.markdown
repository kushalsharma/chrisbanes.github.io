---
layout: post
title: A simple LSTM network using Keras for sequence prediction!
date: '2018-09-23'
cover_image: /content/images/LSTM3-chain.png
---

Long short-term memory (LSTM) units are units of a recurrent neural network (RNN). An RNN composed of LSTM units is often called an LSTM network. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.

LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series.

A typical LSTM network is comprised of different memory blocks called cells. There are two states that are being transferred to the next cell; the cell state and the hidden state. The memory blocks are responsible for remembering things and manipulations to this memory is done through three major mechanisms, called gates.

![png](/content/images/LSTM2-notation.png)

### Forget gate
Forget gate is responsible for removing information from the cell state. The information that is no longer required for the LSTM to understand things or the information that is of less importance is removed via multiplication of a filter. This is required for optimizing the performance of the LSTM network.

![png](/content/images/LSTM3-focus-f.png)

h_t-1 is the hidden state from the previous cell or the output of the previous cell and x_t is the input at that particular time step. The given inputs are multiplied by the weight matrices and a bias is added. After this, the sigmoid function is applied to this value. The sigmoid function outputs a vector, with values ranging from 0 to 1, corresponding to each number in the cell state. If a ‘0’ is output for a particular value in the cell state, it means that the forget gate wants the cell state to forget that piece of information completely. Similarly, a ‘1’ means that the forget gate wants to remember that entire piece of information. This vector output from the sigmoid function is multiplied to the cell state.

### Input gate
The input gate is responsible for the addition of information to the cell state. First it regulates what values need to be added to the cell state by involving a sigmoid function.

![png](/content/images/LSTM3-focus-i.png)

This is similar to the forget gate and acts as a filter for all the information from h_t-1 and x_t. Then it creates a vector containing all possible values that can be added (as perceived from h_t-1 and x_t) to the cell state. This is done using the tanh function, which outputs values from -1 to +1. Lastly, the value of the regulatory filter (the sigmoid gate) is multiplied to the created vector (the tanh function) and then this information is added to the cell state via addition operation.

### Output gate
The output gate selects useful information from the current cell state and show it as an output. It creates a vector after applying tanh function to the cell state, thereby scaling the values to the range -1 to +1.

![png](/content/images/LSTM3-focus-o.png)

Then it makes a filter using the values of h_t-1 and x_t, such that it can regulate the values that need to be output from the vector created above. This filter again employs a sigmoid function. Lastly it multiplies the value of this regulatory filter to the vector created using the tanh function, and sending it out as a output along with to the hidden state of the next cell.



```python
import sys
!{sys.executable} -m pip install -r requirements.txt
```


```python
import math
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
```


```python
# convert an array of values into a data_set matrix
def create_data_set(_data_set, _look_back=1):
    data_x, data_y = [], []
    for i in range(len(_data_set) - _look_back - 1):
        a = _data_set[i:(i + _look_back), 0]
        data_x.append(a)
        data_y.append(_data_set[i + _look_back, 0])
    return numpy.array(data_x), numpy.array(data_y)
```


```python
# load the data_set
data_frame = read_csv('monthly-milk-production-pounds-p.csv')
data_set = data_frame.values
data_set = data_set.astype('float32')
```


```python
# normalize the data_set
scaler = MinMaxScaler(feature_range=(0, 1))
data_set = scaler.fit_transform(data_set)
```


```python
# split into train and test sets
train_size = int(len(data_set) * 0.67)
test_size = len(data_set) - train_size
train, test = data_set[0:train_size, :], data_set[train_size:len(data_set), :]
```


```python
# reshape into X=t and Y=t+1
look_back = 1
train_x, train_y = create_data_set(train, look_back)
test_x, test_y = create_data_set(test, look_back)
```


```python
# reshape input to be [samples, time steps, features]
train_x = numpy.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = numpy.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
```


```python
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=2)
```

    Epoch 1/100
     - 1s - loss: 0.0912
    Epoch 2/100
     - 0s - loss: 0.0372
    Epoch 3/100
     - 0s - loss: 0.0228
    Epoch 4/100
     - 0s - loss: 0.0203
    Epoch 95/100
     - 0s - loss: 0.0108
    Epoch 96/100
     - 0s - loss: 0.0108
    Epoch 97/100
     - 0s - loss: 0.0108
    Epoch 98/100
     - 0s - loss: 0.0110
    Epoch 99/100
     - 0s - loss: 0.0108
    Epoch 100/100
     - 0s - loss: 0.0109


```python
# make predictions
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)
```


```python
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
train_y = scaler.inverse_transform([train_y])
test_predict = scaler.inverse_transform(test_predict)
test_y = scaler.inverse_transform([test_y])
```


```python
# calculate root mean squared error
train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
print('Train Score: %.2f RMSE' % train_score)
test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
print('Test Score: %.2f RMSE' % test_score)
```

    Train Score: 43.06 RMSE
    Test Score: 47.65 RMSE



```python
# shift train predictions for plotting
train_predict_plot = numpy.empty_like(data_set)
train_predict_plot[:, :] = numpy.nan
train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict
```


```python
# shift test predictions for plotting
test_predict_plot = numpy.empty_like(data_set)
test_predict_plot[:, :] = numpy.nan
test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(data_set) - 1, :] = test_predict
```


```python
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data_set))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()
```


![png](/content/images/output_14_0.png)

