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
print(__doc__)

from time import time
import numpy as np
from numpy import genfromtxt
import collections, numpy
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer

from sklearn.decomposition import NMF

np.random.seed(42)

#digits = load_digits()

#Numpy array
#alttype = np.dtype([('myfloat0', 'f8'),('myfloat1', 'f8'),('myfloat02', 'f8'),('myfloat03', 'f8'),('myfloat4', 'f8'),('myfloat5', 'f8'),('myfloat6', 'f8')
#      ('myfloat7', 'f8'),('myfloat8', 'f8'),('myfloat9', 'f8'),('mystring10', 'S5'),('mystring11', 'S5'),('mystring12', 'S5')])

#Type={'names': ('f0', 'f1', 'f2', 'f3', 'f4', 'f5','f6', 'f7', 'f8', 'f9', 's1','s2','s3'),'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4','f4', 'f4', 'f4','f4','S1','S2','S3')}


gazedataString = genfromtxt('resources/q_full_half_owner.csv', delimiter=',', dtype='str')

gazedata = genfromtxt('resources/q_full_half_owner.csv', delimiter=',', usecols = (0,1,2,3,4,5,6,7,8,9))


#gazedata = genfromtxt('a_full_half_owner.csv', delimiter=',')

#newdata= gazedata[:,:10]

data = scale(gazedata)

# get the ith column in np
#newdata = data[:,:10]


#data = MaxAbsScaler().fit_transform(gazedata)
#data = Normalizer().fit_transform(gazedata)


n_samples, n_features = data.shape
clusters = 5
n_iteration = 10

#n_digits = len(np.unique(digits.target))
#labels = digits.target


sample_size = 300
#n_features = 10

print(" \t n_samples %d, \t n_features %d"
      % ( n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
#             metrics.homogeneity_score(labels, estimator.labels_),
#             metrics.completeness_score(labels, estimator.labels_),
#             metrics.v_measure_score(labels, estimator.labels_),
#             metrics.adjusted_rand_score(labels, estimator.labels_),
#             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=clusters, n_init=n_iteration),
              name="k-means++", data=data)

kmeans=KMeans(init='k-means++', n_clusters=clusters, n_init=20)
kmeans.fit(data)
kmeansCenters=kmeans.cluster_centers_
kmeansLabels = kmeans.labels_


#gazeData+KmeansLabels
gazeDataLabels = np.column_stack((gazedata,kmeansLabels))


#gazeDataString+KmeansLabels
gazeDataStringLabels = np.column_stack((gazedataString,kmeansLabels))


unique, counts = numpy.unique(kmeansLabels, return_counts=True)
Counts = dict(zip(unique, counts))



bench_k_means(KMeans(init='random', n_clusters=clusters, n_init=n_iteration),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=clusters).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=clusters, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=n_iteration)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
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

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=4)


centroids = kmeans.cluster_centers_

print("Centriods: ",centroids)

#labels count of KMeans++
for i in Counts:
    print (i, ": ", Counts[i])
    
    
# Scatter plot PCA
# Plot the centroids as a white X
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the gaze dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
