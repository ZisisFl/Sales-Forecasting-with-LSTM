import numpy as np
x = np.array([[20, 10, 5, 44],
              [34, 25, 22, 4],
              [16, 22, 3, 27]])

y = np.array([[76, 43, 9, 4],
              [30, 33, 23, 32],
              [22, 31, 4, 57]])


def scaler(target_data):
    minmax = []
    n_columns = target_data.shape[1]
    n_rows = target_data.shape[0]
    print(n_rows, n_columns)
    for column in range(n_columns):
        value_min = min(target_data[:, column])
        value_max = max(target_data[:, column])
        minmax.append([value_min, value_max])
    return minmax


def scale_data(target_data, minmax):
    target_data = target_data.astype(float)
    n_columns = target_data.shape[1]
    n_rows = target_data.shape[0]
    for column in range(n_columns):
        for row in range(n_rows):
            if minmax[column][1] > 0:  # if max = 0 there is division error and column is for sure full of zero
                target_data[row, column] = (target_data[row, column] - minmax[column][0]) / (minmax[column][1] - minmax[column][0])
    return target_data


def inverse_scaling(target_data, minmax_matrix):
    n_columns = target_data.shape[1]
    n_rows = target_data.shape[0]
    for column in range(n_columns):
        for row in range(n_rows):
            target_data[row, column] = (target_data[row, column] * (minmax_matrix[column][1] - minmax_matrix[column][0])) + minmax_matrix[column][0]
    return target_data


print(x)
scaler_x = scaler(x)
scaler_y = scaler(y)

x_scaled = scale_data(x, scaler_x)
y_scaled = scale_data(y, scaler_y)
print(x_scaled)

x_inv = inverse_scaling(x_scaled, scaler_x)
y_inv = inverse_scaling(y_scaled, scaler_y)
print(x_inv)
