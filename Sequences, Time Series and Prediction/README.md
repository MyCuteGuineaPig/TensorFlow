
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
