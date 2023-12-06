import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats

from sklearn import cluster
from sklearn import neighbors
from sklearn import naive_bayes

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib as mpl


SIZE = 200


def gen_2d_norm_dist(ctr: np.ndarray, cov_diag: np.ndarray):
    cov = np.array([[cov_diag[0], 0], [0, cov_diag[1]]])
    return np.random.multivariate_normal(mean=ctr, cov=cov, size=SIZE)


if __name__ == '__main__':
    data1 = gen_2d_norm_dist(np.array([3, 3]), np.array([1.5, 1.5]))
    data2 = gen_2d_norm_dist(np.array([9, 2]), np.array([1, 1]))
    data3 = gen_2d_norm_dist(np.array([9, 6]), np.array([1, 1]))

    plt.figure(num='initial data')
    plt.title("Initial data")
    plt.plot(data1[:, 0], data1[:, 1], 'ro', markersize=3, label='c(3, 3), d(1.5, 1.5)')
    plt.plot(data2[:, 0], data2[:, 1], 'go', markersize=3, label='c(9, 2), d(1, 1)')
    plt.plot(data3[:, 0], data3[:, 1], 'bo', markersize=3, label='c(9, 6), d(1, 1)')
    plt.grid()
    plt.legend()
    # plt.show()

    n_classes = 3

    X = np.concatenate((np.concatenate((data1, data2)), data3))

    cluster_model = cluster.KMeans(n_clusters=n_classes)
    cluster_model.fit(X)
    clust_y = cluster_model.predict(X)

    predict = [[] for _ in range(n_classes)]
    for xi, yi in zip(X, clust_y):
        predict[yi].append(xi)

    means1 = np.array(predict[0])
    means2 = np.array(predict[1])
    means3 = np.array(predict[2])

    plt.figure(num='k-means')
    plt.title('K-means method')
    plt.plot(means1[:, 0], means1[:, 1], 'ro', markersize=3, label='c(3, 3), d(1.5, 1.5)')
    plt.plot(means2[:, 0], means2[:, 1], 'go', markersize=3, label='c(9, 2), d(1, 1)')
    plt.plot(means3[:, 0], means3[:, 1], 'bo', markersize=3, label='c(9, 6), d(1, 1)')
    plt.grid()
    plt.legend()
    # plt.show()

    data = [data1, data2, data3]
    k_means = [means1, means2, means3]

    ress = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            ress_ij = 0
            for x in data[i]:
                for y in k_means[j]:
                    if all(x == y):
                        ress_ij += 1
            ress[i] = max(ress[i], ress_ij)


    def split_predicted_data(X, model):
        res = model.predict(X)

        i_neight_data1 = np.argwhere(res == 0)
        i_neight_data2 = np.argwhere(res == 1)
        i_neight_data3 = np.argwhere(res == 2)

        neight_data1 = X[i_neight_data1[:, 0]]
        neight_data2 = X[i_neight_data2[:, 0]]
        neight_data3 = X[i_neight_data3[:, 0]]
        return neight_data1, neight_data2, neight_data3


    X = np.concatenate((data1, data2, data3), axis=0)
    y = np.array([0] * SIZE + [1] * SIZE + [2] * SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1. / 5.), random_state=42)

    neighbors = KNeighborsClassifier(n_neighbors=3)
    neighbors.fit(X_train, y_train)

    neighbors1, neighbors2, neighbors3 = split_predicted_data(X, neighbors)

    plt.figure(num='k-neighbors')
    plt.title('k-neighbors')
    plt.plot(neighbors1[:, 0], neighbors1[:, 1], 'ro', markersize=3, label='c(3, 3), d(1.5, 1.5)')
    plt.plot(neighbors2[:, 0], neighbors2[:, 1], 'go', markersize=3, label='c(9, 2), d(1, 1)')
    plt.plot(neighbors3[:, 0], neighbors3[:, 1], 'bo', markersize=3, label='c(9, 6), d(1, 1)')
    plt.grid()
    plt.legend()
    # plt.show()

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    gnb1, gnb2, gnb3 = split_predicted_data(X, gnb)

    plt.figure(num='gauss')
    plt.title('gauss')
    plt.plot(gnb1[:, 0], gnb1[:, 1], 'ro', markersize=3, label='c(3, 3), d(1.5, 1.5)')
    plt.plot(gnb2[:, 0], gnb2[:, 1], 'go', markersize=3, label='c(9, 2), d(1, 1)')
    plt.plot(gnb3[:, 0], gnb3[:, 1], 'bo', markersize=3, label='c(9, 6), d(1, 1)')
    plt.grid()
    plt.legend()
    # plt.show()

    print("k neighbors score: {:.5f}".format(sum(ress) / 600))
    print("k neighbors train score: {:.5f}".format(neighbors.score(X_train, y_train)))
    print("k neighbors test score: {:.5f}".format(neighbors.score(X_test, y_test)))
    print("gauss train score: {:.5f}".format(gnb.score(X_train, y_train)))
    print("gauss test score: {:.5f}".format(gnb.score(X_test, y_test)))

    cm = []
    print("K-neigh train conf matrix")
    cm.append(['K-neigh train', confusion_matrix(y_train, neighbors.predict(X_train))])
    print(confusion_matrix(y_train, neighbors.predict(X_train)))
    print("Gaussian Naive Bayes train conf matrix")
    cm.append(['Gaussian Naive Bayes train', confusion_matrix(y_train, gnb.predict(X_train))])
    print(confusion_matrix(y_train, gnb.predict(X_train)))

    print("K-neigh test conf matrix")
    cm.append(['K-neigh test', confusion_matrix(y_test, neighbors.predict(X_test))])
    print(confusion_matrix(y_test, neighbors.predict(X_test)))
    print("Gaussian Naive Bayes test conf matrix")
    cm.append(['Gaussian Naive Bayes test', confusion_matrix(y_test, gnb.predict(X_test))])
    print(confusion_matrix(y_test, gnb.predict(X_test)))

    plt.show()

    # fig, ax = plt.subplots(2, 2)
    # for i in range(2):
    #     for j in range(2):
    #         ax[i, j].set_title(cm[i*2+j][0])
    #         ConfusionMatrixDisplay(confusion_matrix=cm[i*2+j][1],
    #                                display_labels=['1','2','3']).plot(ax=ax[i, j])
    # plt.show()
