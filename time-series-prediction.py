import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.layers.wrappers import *
from config import *

df = pd.read_csv(eur_usd_loc)
print(df.count())

# Rename bid OHLC columns
df.rename(columns={'Time': 'timestamp', 'Open': 'open', 'Close': 'close',
                   'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
df.set_index('timestamp', inplace=True)
df = df.astype(float)

# Add additional features
df['momentum'] = df['volume'] * (df['open'] - df['close'])
df['avg_price'] = (df['low'] + df['high']) / 2
# df['range'] = df['high'] - df['low']
df['ohlc_price'] = (df['low'] + df['high'] + df['open'] + df['close']) / 4
df['oc_diff'] = df['open'] - df['close']

print(df.head())


def create_dataset(dataset, look_back=20):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# Scale and create datasets
target_index = df.columns.tolist().index('close')
high_index = df.columns.tolist().index('high')
low_index = df.columns.tolist().index('low')
dataset = df.values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create y_scaler to inverse it later
y_scaler = MinMaxScaler(feature_range=(0, 1))
t_y = df['close'].values.astype('float32')
t_y = np.reshape(t_y, (-1, 1))
y_scaler = y_scaler.fit(t_y)

X, y = create_dataset(dataset, look_back=30)
y = y[:, target_index]

train_size = int(len(X) * 0.99)
trainX = X[:train_size]
trainY = y[:train_size]
testX = X[train_size:]
testY = y[train_size:]

# create a small LSTM network
model = Sequential()
model.add(
    Bidirectional(LSTM(30, input_shape=(X.shape[1], X.shape[2]),
                       return_sequences=True),
                  merge_mode='sum',
                  weights=None,
                  input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(10, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(4, return_sequences=False))
model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
print(model.summary())


def train():
    # Save the best weight during training.
    checkpoint = ModelCheckpoint(bi_rnn_weights,
                                 monitor='val_mean_squared_error',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')

    # Fit
    callbacks_list = [checkpoint]
    history = model.fit(trainX, trainY, epochs=200, batch_size=1024, verbose=1, callbacks=callbacks_list,
                        validation_split=0.1)

    epoch = len(history.history['loss'])

    for k in list(history.history.keys()):
        if 'val' not in k:
            plt.figure(figsize=(40, 10))
            plt.plot(history.history[k])
            plt.plot(history.history['val_' + k])
            plt.title(k)
            plt.ylabel(k)
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
    # Baby the model a bit
    # Load the weight that worked the best
    model.load_weights(bi_rnn_weights)

    # Train again with decaying learning rate

    def scheduler(epoch):
        if epoch % 2 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * .9)
            print("lr changed to {}".format(lr * .9))
        return K.get_value(model.optimizer.lr)

    lr_decay = LearningRateScheduler(scheduler)

    callbacks_list = [checkpoint, lr_decay]
    history = model.fit(trainX, trainY, epochs=int(min(epoch / 3, 5)), batch_size=500, verbose=1,
                        callbacks=callbacks_list,
                        validation_split=0.1)


def benchmark():
    # Benchmark
    model.load_weights(bi_rnn_weights)

    pred = model.predict(testX)
    pred = y_scaler.inverse_transform(pred)
    close = y_scaler.inverse_transform(np.reshape(testY, (testY.shape[0], 1)))
    predictions = pd.DataFrame()
    predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
    predictions['close'] = pd.Series(np.reshape(close, (close.shape[0])))

    p = df[-pred.shape[0]:].copy()
    predictions.index = p.index
    predictions = predictions.astype(float)
    predictions = predictions.merge(p[['low', 'high']], right_index=True, left_index=True)

    ax = predictions.plot(x=predictions.index, y='close', c='red', figsize=(40, 10))
    ax = predictions.plot(x=predictions.index, y='predicted', c='blue', figsize=(40, 10), ax=ax)
    index = [str(item) for item in predictions.index]
    plt.fill_between(x=index, y1='low', y2='high', data=p, alpha=0.4)
    plt.title('Prediction vs Actual (low and high as blue region)')
    plt.show()

    predictions['diff'] = predictions['predicted'] - predictions['close']
    plt.figure(figsize=(10, 10))
    sns.distplot(predictions['diff']);
    plt.title('Distribution of differences between actual and prediction ')
    plt.show()

    g = sns.jointplot("diff", "predicted", data=predictions, kind="kde", space=0)
    plt.title('Distributtion of error and price')
    plt.show()

    # predictions['correct'] = (predictions['predicted'] <= predictions['high']) & (predictions['predicted'] >= predictions['low'])
    # sns.factorplot(data=predictions, x='correct', kind='count')

    print("MSE : ", mean_squared_error(predictions['predicted'].values, predictions['close'].values))
    print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['close'].values))
    predictions['diff'].describe()


train()
benchmark()
