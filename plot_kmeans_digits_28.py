from keras.datasets import mnist

from plot_kmeans_digits import run

(X_digits, y_digits), (x_test, y_test) = mnist.load_data()
data = X_digits.reshape(60000, 784)

n_samples, n_features = data.shape
n_digits = 10
labels = y_digits

if __name__ == '__main__':
    run(data, n_samples, n_features, n_digits, labels, dataset='full_mnist')
