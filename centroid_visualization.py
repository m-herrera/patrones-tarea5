from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def plot_centroid(data, k, dataset='mnist', init='k-means++', n_init=10):

    if(dataset == 'mnist'):
        shape = 28
        max_value = 255
    else:
        shape = 8
        max_value = 16

    full_dimensions = KMeans(init=init, n_clusters=k, n_init=n_init)
    full_dimensions.fit(data)
    centroids = full_dimensions.cluster_centers_
    images = centroids.reshape(centroids.shape[0], shape, shape).astype('uint8')

    plt.figure()
    cols = k // 2
    if k % 2 != 0:
        cols += 1

    for i in range(k):
        plt.subplot(2, cols, i + 1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=max_value)
    plt.show()

    return centroids
