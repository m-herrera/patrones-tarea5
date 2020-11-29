"""
===========================================================
A demo of K-Means clustering on the handwritten digits data
===========================================================

In this example we compare the various initialization strategies for
K-means in terms of runtime and quality of the results.

As the ground truth is known here, we also apply different cluster
quality metrics to judge the goodness of fit of the cluster labels to the
ground truth.

Cluster quality metrics evaluated (see :ref:`clustering_evaluation` for
definitions and discussions of the metrics):

=========== ========================================================
Shorthand    full name
=========== ========================================================
homo         homogeneity score
compl        completeness score
v-meas       V measure
ARI          adjusted Rand index
AMI          adjusted mutual information
silhouette   silhouette coefficient
=========== ========================================================

"""
from centroid_visualization import plot_centroid
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(__doc__)

np.random.seed(42)


def run(data, n_samples, n_features, n_digits, labels, dataset='full_mnist'):
    sample_size = 300
    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_features))

    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

    def bench_k_means(estimator, name, data):
        t0 = time()
        estimator.fit(data)
        print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
              % (name, (time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(labels, estimator.labels_),
                 metrics.completeness_score(labels, estimator.labels_),
                 metrics.v_measure_score(labels, estimator.labels_),
                 metrics.adjusted_rand_score(labels, estimator.labels_),
                 metrics.adjusted_mutual_info_score(labels, estimator.labels_),
                 metrics.silhouette_score(data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=sample_size)))

    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                  name="k-means++", data=data)

    bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
                  name="random", data=data)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_digits).fit(data)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
                  name="PCA-based",
                  data=data)
    print(82 * '_')

    # #############################################################################
    # plot_centroid(data, 3, dataset='mnist')
    centroids = plot_centroid(data, n_digits, dataset=dataset)
    # plot_centroid(data, 20, dataset='mnist')

    # Graphing subset
    graphing_data = data[np.random.choice(data.shape[0], 500, replace=False), :]

    # Visualize the results on PCA-reduced data
    pca_2d = PCA(n_components=2)
    pca_2d.fit(data)
    reduced_data = pca_2d.transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    print("Reduce data ", reduced_data.shape)

    reduced_data = pca_2d.transform(graphing_data)

    centroids_multipleD = pca_2d.transform(centroids)

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = (x_max - x_min) / 2000  # point in the mesh [x_min, x_max]x[y_min, y_max].

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids_2d = kmeans.cluster_centers_
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    # Plot the centroids as a green X
    plt.scatter(centroids_multipleD[:, 0], centroids_multipleD[:, 1],
                marker='x', s=169, linewidths=3,
                color='g', zorder=10)

    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross\n'
              'Centroids in all dimensions are marked with green cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    # #############################################################################
    # Visualize the results with the centroids principal components
    pca_2d = PCA(n_components=2)
    pca_2d.fit(centroids)
    reduced_data = pca_2d.transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    print("Reduce data ", reduced_data.shape)

    reduced_data = pca_2d.transform(graphing_data)

    centroids_multipleD = pca_2d.transform(centroids)

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = (x_max - x_min) / 2000  # point in the mesh [x_min, x_max]x[y_min, y_max].

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids_2d = kmeans.cluster_centers_
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    # Plot the centroids as a green X
    plt.scatter(centroids_multipleD[:, 0], centroids_multipleD[:, 1],
                marker='x', s=169, linewidths=3,
                color='g', zorder=10)

    plt.title('K-means clustering on the digits dataset (All dimensions principal components)\n'
              'Centroids are marked with white cross\n'
              'Centroids in all dimensions are marked with green cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
