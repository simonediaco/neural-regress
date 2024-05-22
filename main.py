import os
from train import train_models
from models import load_data, split_data
import matplotlib.pyplot as plt


def process_dataset(file_path, dataset_name):
    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    results = train_models(X_train, y_train, X_test, y_test)
    for n, mse_train, mse_test in results:
        print(f"Dataset: {dataset_name}, Neurons: {n}, MSE Train: {mse_train}, MSE Test: {mse_test}")

    return results


def main():
    file_path_wizmir = './data/wizmir.txt'
    file_path_ele2 = './data/ele-2.txt'

    results_wizmir = process_dataset(file_path_wizmir, "wizmir")
    results_ele2 = process_dataset(file_path_ele2, "ele2")

    neurons = [n for n, _, _ in results_wizmir]

    mse_train_wizmir = [mse_train for _, mse_train, _ in results_wizmir]
    mse_test_wizmir = [mse_test for _, _, mse_test in results_wizmir]

    mse_train_ele2 = [mse_train for _, mse_train, _ in results_ele2]
    mse_test_ele2 = [mse_test for _, _, mse_test in results_ele2]

    plt.figure(figsize=(10, 5))

    # Plot per wizmir
    plt.plot(neurons, mse_train_wizmir, label='MSE Training Wizmir', marker='o')
    plt.plot(neurons, mse_test_wizmir, label='MSE Test Wizmir', marker='o')

    # Plot per ele2
    plt.plot(neurons, mse_train_ele2, label='MSE Training Ele2', marker='o')
    plt.plot(neurons, mse_test_ele2, label='MSE Test Ele2', marker='o')

    plt.title('MSE vs. Neurons in Hidden Layer (Confronto dei Dataset)')
    plt.xlabel('Number of Neurons')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)

    plt.ylim(bottom=0)

    plots_dir = './plots'
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, 'mse_plot_comparison.png')
    plt.savefig(plot_path)
    print(f"Grafico salvato in: {plot_path}")

    # plt.show()


if __name__ == '__main__':
    main()
