import numpy as np


def load_data(data_path):
    # Load and transform the "emissions" data
    raw_data = np.load(data_path)["data"][:, :, 0]
    data = np.expand_dims(raw_data, axis=-1)  # (17856, 170, 1)
    return data
