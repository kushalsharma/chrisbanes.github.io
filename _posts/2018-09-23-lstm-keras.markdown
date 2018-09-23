---
layout: post
title: A simple LSTM network using Keras for sequence prediction!
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

