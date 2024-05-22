import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)


def build_and_train_model(X_train, y_train, X_test, y_test, neurons):
    model = Sequential([
        Dense(neurons, input_dim=X_train.shape[1], activation='sigmoid'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, verbose=0)
    mse_train = model.evaluate(X_train, y_train, verbose=0)
    mse_test = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neurons: {neurons}, MSE Train: {mse_train}, MSE Test: {mse_test}")

    return mse_train, mse_test
