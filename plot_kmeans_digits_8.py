from sklearn.datasets import load_digits

from plot_kmeans_digits import run

X_digits, y_digits = load_digits(return_X_y=True)
data = X_digits

n_samples, n_features = data.shape
n_digits = 10
labels = y_digits

if __name__ == '__main__':
    run(data, n_samples, n_features, n_digits, labels, dataset='digits')
