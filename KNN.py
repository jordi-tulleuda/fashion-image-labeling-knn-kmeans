__authors__ = ['UAB AI Team 25']
__group__ = 'Team_25'

import numpy as np

from collections import Counter
from scipy.spatial.distance import cdist

from utils import rgb2gray


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        train_data = train_data.astype(float)
        if train_data.ndim == 4:
            train_data = rgb2gray(train_data)
        P = train_data.shape[0]
        self.train_data = train_data.reshape(P, -1)


    def get_k_neighbours(self, test_data, k, distance='euclidean'):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix
        :param k: the number of neighbors to look at
        :param distance: distance metric to use: 'euclidean', 'cityblock' or 'cosine'
        """
        test_data = test_data.astype(float)

        if test_data.ndim == 3 and test_data.shape[-1] == 3 or test_data.ndim == 4:
            test_data = rgb2gray(test_data)

        if test_data.ndim == 2 or test_data.ndim == 3 and test_data.shape[-1] == 3:
            test_data = test_data.reshape(1, -1)

        if test_data.ndim == 3 or test_data.ndim == 4:
            test_data = test_data.reshape(test_data.shape[0], -1)

        distancies = cdist(test_data, self.train_data, metric=distance)
        idx = np.argsort(distancies, axis=1)
        idx = idx[:, :k]
        self.neighbors = self.labels[idx]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        predictions = []
        for i in range(self.neighbors.shape[0]):
            veins = self.neighbors[i]
            recompte = Counter(veins)
            max_count = max(recompte.values())
            candidates = [etiqueta for etiqueta in veins if recompte[etiqueta] == max_count]
            predictions.append(candidates[0])
        return np.array(predictions)

    def predict(self, test_data, k, distance='euclidean'):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """
        self.get_k_neighbours(test_data, k, distance)
        return self.get_class()