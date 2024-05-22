from models import build_and_train_model


def train_models(X_train, y_train, X_test, y_test):
    neurons_range = range(5, 51, 5)
    results = []
    for neurons in neurons_range:
        mse_train, mse_test = build_and_train_model(X_train, y_train, X_test, y_test, neurons)
        results.append((neurons, mse_train, mse_test))
    return results
