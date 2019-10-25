import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import Utilities as util

# read the data from data file

milk = pd.read_csv("./data/monthly-milk-production.csv", index_col='Month')

milk.index = pd.to_datetime(milk.index)
# milk.plot()
# plt.show()

# split the data
train_data = milk.head(156)
test_data = milk.tail(12)

# scale the train data and test data
scaler = MinMaxScaler()

# Do not fit the test data to make sure that the same data distribution between training data and test data
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

x, y = util.generate_sample_data(scaled_train_data, 12, 1000)

# exit()
# print(scaled_train_data)
# print('\n')
# print(scaled_test_data)

# create the sequence model with keras
num_neurons = 200
num_input = 12
num_output = 1
batch_size = 1
time_steps = 12
mode = 'Inference'


# mode = 'train'


def build_mode():
    model = keras.Sequential()

    # lstm_layer = keras.layers.LSTM(units=num_neurons, return_sequences=True)
    rnn_layer = keras.layers.SimpleRNN(units=num_neurons, input_dim=1, return_sequences=True)

    dense_layer = keras.layers.Dense(units=1, activation='sigmoid')
    model.add(rnn_layer)

    model.add(dense_layer)

    model.compile(optimizer='Adam', loss='mse')
    return model


if mode == 'train':
    model = build_mode()
    model.fit(x, y, batch_size=64, epochs=100)
    model.save("SequenceModel.h5", overwrite=True)
    model.save_weights("SequenceModelWeights_RNN.h5", overwrite=True)

model_2 = build_mode()
model_2.load_weights("SequenceModelWeights_RNN.h5")
# model_2 = keras.models.load_model("SequenceModel.h5")

scaled_test_data = np.array(scaled_test_data).reshape(-1, 12, 1)
result = model_2.predict(scaled_test_data)

final_result = np.array(result).reshape(12, 1)

final_result = scaler.inverse_transform(final_result)

test_data['Generate'] = final_result

test_data.plot()

plt.show()
