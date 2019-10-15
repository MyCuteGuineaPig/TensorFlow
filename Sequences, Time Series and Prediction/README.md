
## Pandas Plot Autocorrelation

Note: ```series``` need to be list either built-in list or ```np.array()```

```python
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
```

## Evaluation Metrics

- **Errors**: difference from model and actual values over evalution period
  - ```errors = forecasts - actual``` 
- **Mean Square Error**: square to get rid of negative values
  - ```np.square(forecast - actual).mean()```
  - ```keras.metrics.mean_squared_error(forecast, actual).numpy()``` Generate the same result as above
- **Root Mean Square Error**: Make the mean of error calculation to be the same scale as original errors
  - ```rmse = np.sqrt(mse)```
- **Mean Absolute Error**: mean absolute error, also called mean absolute derivation or mad, this is not penalize large errors as mush as mse does
  - ```np.abs(forecast - actual).mean()```
  - ```keras.metrics.mean_absolute_error(forecast, actual).numpy()``` Generate the same result as above
- **Mean Absolute Percentage Error**: mean ratio between absolute error and absolute value. this gives the idea of size of errors compared to the values
  - ```mape = np.abs(errors / x_valid).mean()```:


## Generate Data From TimeSeries DataSet

- ```dataset.shuffle(shuffle_buffer)```:  For example. If have 100,000 items in dataset, but set shuffle_buffer 1000, it will just fill the buffer with first 1000 elements, and pick them random. Then it will replace that with the 1001 before randomly pickly again. This way with super large datasets, the random element choosing can choose from a smaller number which speed up effectively.

```python
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  # batch_size is the size for training 
  # shuffle_buffer: determine how data will be shuffled
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True) # + 1 bc 1 as label
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1) # batched into selected batch size
  return dataset
```

## Train in DNN

- ```tf.keras.callbacks.LearningRateScheduler```: update learning rate after each epoch by callbacks


```python
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(10, activation="relu"), 
    tf.keras.layers.Dense(1)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
    
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule], verbose=0)

tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()
```


## Plot Prediction

#### Plot validation vs prediction

```python
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
forecast = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
  ## np. newaxis then just reshape to input dimension that used by the model

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]


plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)
```

#### Plot Learning Rate VS Loss

```python
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])
```

#### Plot loss vs epoch
```python
loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.show()
```
