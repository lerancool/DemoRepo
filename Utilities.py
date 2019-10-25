import numpy as np


def generate_sample_data(train_data, time_steps, num_sample):
    x = []
    y = []
    for i in range(num_sample):
        start_point = np.random.randint(0, len(train_data) - time_steps)
        batch_data = train_data[start_point: start_point + time_steps + 1]
        # x_batch = np.array(batch_data[:-1, :]).reshape(-1, 12, 1)
        # y_batch = np.array(batch_data[1:, :]).reshape(-1, 12, 1)
        x_batch = batch_data[:-1, :]
        y_batch = batch_data[1:, :]

        x.append(x_batch)
        y.append(y_batch)

    x = np.array(x).reshape(num_sample, time_steps, 1)
    y = np.array(y).reshape(num_sample, time_steps, 1)
    return x, y
