import numpy as np


def Add_Window_Horizon(data, pos_w, pos_d, window=3, horizon=1, single=False):
    """
    :param data: shape [B, ...]
    :param pos_w: shape [B, ...] - position within the week
    :param pos_d: shape [B, ...] - position within the day
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    """
    length = len(data)
    end_index = length - horizon - window + 1
    X = []  # windows
    Y = []  # horizon
    pos_w_windows = []
    pos_d_windows = []
    original_indices = []  # To store the original indices

    index = 0

    if single:
        while index < end_index:
            X.append(data[index : index + window])
            Y.append(data[index + window + horizon - 1 : index + window + horizon])
            pos_w_windows.append(pos_w[index : index + window])
            pos_d_windows.append(pos_d[index : index + window])
            original_indices.append(
                np.arange(index, index + window + horizon)
            )  # Store the starting index
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index : index + window])
            Y.append(data[index + window : index + window + horizon])
            pos_w_windows.append(pos_w[index : index + window])
            pos_d_windows.append(pos_d[index : index + window])
            original_indices.append(
                np.arange(index, index + window + horizon)
            )  # Store the starting index
            index = index + 1

    X = np.array(X)
    Y = np.array(Y)
    pos_w_windows = np.array(pos_w_windows)
    pos_d_windows = np.array(pos_d_windows)
    original_indices = np.array(original_indices)  # Convert indices to array

    return X, Y, pos_w_windows, pos_d_windows, original_indices
