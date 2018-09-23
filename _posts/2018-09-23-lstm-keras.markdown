---
layout: post
title: A simple LSTM using Keras for sequence prediction
date: '2018-09-23'
---

```python
import sys
!{sys.executable} -m pip install -r requirements.txt
```

    Requirement already satisfied: numpy==1.14.5 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from -r requirements.txt (line 1)) (1.14.5)
    Requirement already satisfied: pandas==0.21.1 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from -r requirements.txt (line 2)) (0.21.1)
    Requirement already satisfied: keras==2.2.2 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from -r requirements.txt (line 3)) (2.2.2)
    Requirement already satisfied: requests==2.18.4 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from -r requirements.txt (line 4)) (2.18.4)
    Requirement already satisfied: scipy==1.0.0 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from -r requirements.txt (line 5)) (1.0.0)
    Requirement already satisfied: scikit-learn==0.19.1 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from -r requirements.txt (line 6)) (0.19.1)
    Requirement already satisfied: matplotlib==2.2.3 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from -r requirements.txt (line 7)) (2.2.3)
    Requirement already satisfied: pytz>=2011k in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from pandas==0.21.1->-r requirements.txt (line 2)) (2018.4)
    Requirement already satisfied: python-dateutil>=2 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from pandas==0.21.1->-r requirements.txt (line 2)) (2.7.3)
    Requirement already satisfied: six>=1.9.0 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from keras==2.2.2->-r requirements.txt (line 3)) (1.11.0)
    Requirement already satisfied: pyyaml in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from keras==2.2.2->-r requirements.txt (line 3)) (3.12)
    Requirement already satisfied: h5py in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from keras==2.2.2->-r requirements.txt (line 3)) (2.8.0)
    Requirement already satisfied: keras_applications==1.0.4 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from keras==2.2.2->-r requirements.txt (line 3)) (1.0.4)
    Requirement already satisfied: keras_preprocessing==1.0.2 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from keras==2.2.2->-r requirements.txt (line 3)) (1.0.2)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 4)) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 4)) (2018.8.13)
    Requirement already satisfied: idna<2.7,>=2.5 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 4)) (2.6)
    Requirement already satisfied: urllib3<1.23,>=1.21.1 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from requests==2.18.4->-r requirements.txt (line 4)) (1.22)
    Requirement already satisfied: kiwisolver>=1.0.1 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from matplotlib==2.2.3->-r requirements.txt (line 7)) (1.0.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from matplotlib==2.2.3->-r requirements.txt (line 7)) (2.2.0)
    Requirement already satisfied: cycler>=0.10 in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from matplotlib==2.2.3->-r requirements.txt (line 7)) (0.10.0)
    Requirement already satisfied: setuptools in /anaconda2/envs/tensorflow3.6/lib/python3.6/site-packages (from kiwisolver>=1.0.1->matplotlib==2.2.3->-r requirements.txt (line 7)) (39.2.0)
    [31mtwisted 18.7.0 requires PyHamcrest>=1.9.0, which is not installed.[0m
    [31mtensorflow 1.10.0 has requirement setuptools<=39.1.0, but you'll have setuptools 39.2.0 which is incompatible.[0m
    [33mYou are using pip version 10.0.1, however version 18.0 is available.
    You should consider upgrading via the 'pip install --upgrade pip' command.[0m



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
    Epoch 5/100
     - 0s - loss: 0.0192
    Epoch 6/100
     - 0s - loss: 0.0182
    Epoch 7/100
     - 0s - loss: 0.0171
    Epoch 8/100
     - 0s - loss: 0.0163
    Epoch 9/100
     - 0s - loss: 0.0153
    Epoch 10/100
     - 0s - loss: 0.0148
    Epoch 11/100
     - 0s - loss: 0.0140
    Epoch 12/100
     - 0s - loss: 0.0132
    Epoch 13/100
     - 0s - loss: 0.0127
    Epoch 14/100
     - 0s - loss: 0.0123
    Epoch 15/100
     - 0s - loss: 0.0121
    Epoch 16/100
     - 0s - loss: 0.0116
    Epoch 17/100
     - 0s - loss: 0.0115
    Epoch 18/100
     - 0s - loss: 0.0114
    Epoch 19/100
     - 0s - loss: 0.0114
    Epoch 20/100
     - 0s - loss: 0.0112
    Epoch 21/100
     - 0s - loss: 0.0111
    Epoch 22/100
     - 0s - loss: 0.0111
    Epoch 23/100
     - 0s - loss: 0.0110
    Epoch 24/100
     - 0s - loss: 0.0110
    Epoch 25/100
     - 0s - loss: 0.0110
    Epoch 26/100
     - 0s - loss: 0.0109
    Epoch 27/100
     - 0s - loss: 0.0109
    Epoch 28/100
     - 0s - loss: 0.0109
    Epoch 29/100
     - 0s - loss: 0.0110
    Epoch 30/100
     - 0s - loss: 0.0108
    Epoch 31/100
     - 0s - loss: 0.0112
    Epoch 32/100
     - 0s - loss: 0.0112
    Epoch 33/100
     - 0s - loss: 0.0111
    Epoch 34/100
     - 0s - loss: 0.0111
    Epoch 35/100
     - 0s - loss: 0.0110
    Epoch 36/100
     - 0s - loss: 0.0109
    Epoch 37/100
     - 0s - loss: 0.0110
    Epoch 38/100
     - 0s - loss: 0.0108
    Epoch 39/100
     - 0s - loss: 0.0109
    Epoch 40/100
     - 0s - loss: 0.0109
    Epoch 41/100
     - 0s - loss: 0.0110
    Epoch 42/100
     - 0s - loss: 0.0109
    Epoch 43/100
     - 0s - loss: 0.0109
    Epoch 44/100
     - 0s - loss: 0.0110
    Epoch 45/100
     - 0s - loss: 0.0109
    Epoch 46/100
     - 0s - loss: 0.0109
    Epoch 47/100
     - 0s - loss: 0.0110
    Epoch 48/100
     - 0s - loss: 0.0111
    Epoch 49/100
     - 0s - loss: 0.0110
    Epoch 50/100
     - 0s - loss: 0.0109
    Epoch 51/100
     - 0s - loss: 0.0109
    Epoch 52/100
     - 0s - loss: 0.0109
    Epoch 53/100
     - 0s - loss: 0.0109
    Epoch 54/100
     - 0s - loss: 0.0109
    Epoch 55/100
     - 0s - loss: 0.0110
    Epoch 56/100
     - 0s - loss: 0.0108
    Epoch 57/100
     - 0s - loss: 0.0109
    Epoch 58/100
     - 0s - loss: 0.0109
    Epoch 59/100
     - 0s - loss: 0.0107
    Epoch 60/100
     - 0s - loss: 0.0111
    Epoch 61/100
     - 0s - loss: 0.0109
    Epoch 62/100
     - 0s - loss: 0.0109
    Epoch 63/100
     - 0s - loss: 0.0108
    Epoch 64/100
     - 0s - loss: 0.0108
    Epoch 65/100
     - 0s - loss: 0.0110
    Epoch 66/100
     - 0s - loss: 0.0109
    Epoch 67/100
     - 0s - loss: 0.0111
    Epoch 68/100
     - 0s - loss: 0.0109
    Epoch 69/100
     - 0s - loss: 0.0109
    Epoch 70/100
     - 0s - loss: 0.0110
    Epoch 71/100
     - 0s - loss: 0.0110
    Epoch 72/100
     - 0s - loss: 0.0110
    Epoch 73/100
     - 0s - loss: 0.0110
    Epoch 74/100
     - 0s - loss: 0.0110
    Epoch 75/100
     - 0s - loss: 0.0110
    Epoch 76/100
     - 0s - loss: 0.0111
    Epoch 77/100
     - 0s - loss: 0.0109
    Epoch 78/100
     - 0s - loss: 0.0109
    Epoch 79/100
     - 0s - loss: 0.0110
    Epoch 80/100
     - 0s - loss: 0.0110
    Epoch 81/100
     - 0s - loss: 0.0107
    Epoch 82/100
     - 0s - loss: 0.0112
    Epoch 83/100
     - 0s - loss: 0.0109
    Epoch 84/100
     - 0s - loss: 0.0112
    Epoch 85/100
     - 0s - loss: 0.0110
    Epoch 86/100
     - 0s - loss: 0.0110
    Epoch 87/100
     - 0s - loss: 0.0109
    Epoch 88/100
     - 0s - loss: 0.0108
    Epoch 89/100
     - 0s - loss: 0.0110
    Epoch 90/100
     - 0s - loss: 0.0108
    Epoch 91/100
     - 0s - loss: 0.0109
    Epoch 92/100
     - 0s - loss: 0.0111
    Epoch 93/100
     - 0s - loss: 0.0109
    Epoch 94/100
     - 0s - loss: 0.0109
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





    <keras.callbacks.History at 0x11f6e14a8>




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

