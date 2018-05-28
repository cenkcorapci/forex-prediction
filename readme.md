# Forex Prediction

Testing some models on this [kaggle data set](https://www.kaggle.com/kimy07/eurusd-15-minute-interval-price-prediction)
## Results
| Model               | Mean Squared Error    | Mean Absolute error  | Details                                                                                           |
|---------------------|-----------------------|----------------------|---------------------------------------------------------------------------------------------------|
| Multilayer-GRU      | 0.0007788221697455181 | 0.026117904067719574 | [Notebook](https://github.com/cenkcorapci/forex-prediction/blob/master/baseline-experiment.ipynb) |
| Bi-directional LSTM | 6.159634463135478e-05 | 0.007467821835500006 | [Notebook](https://github.com/cenkcorapci/forex-prediction/blob/master/bi-rnn-experiment.ipynb)   |
