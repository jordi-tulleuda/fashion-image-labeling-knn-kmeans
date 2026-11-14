__authors__ = ['UAB AI Team 25']
__group__ = 'Team_25'

import numpy as np
import utils_data
import utils

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        X = np.array(X, dtype=float)
        if len(X.shape) > 2:
            self.X = X.reshape(-1, X.shape[-1])
        else:
            self.X = X

    def _init_options(self, options=None):
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'
        self.options = options

    def _init_centroids(self):
        """
        Inicialitza els centroides segons el mètode seleccionat:
        - first
        - random
        - kmeans++
        - custom
        """
        method = self.options['km_init'].lower()
        # FIRST CENTROIDS
        if method == 'first':
            centroids = []
            for x in self.X:
                if not any(np.array_equal(x, c) for c in centroids):
                    centroids.append(x)

                if len(centroids) == self.K:
                    break
            self.centroids = np.array(centroids)

        # RANDOM CENTROIDS
        elif method == 'random':
            idx = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[idx]

        # KMEANS++
        elif method == 'kmeans++':
            centroids = []
            # Primer centroide aleatori
            idx = np.random.randint(self.X.shape[0])
            centroids.append(self.X[idx])
            # Escollir següents centroides
            for _ in range(1, self.K):
                distances = []
                for x in self.X:
                    d = min(np.linalg.norm(x - c) for c in centroids)
                    distances.append(d**2)
                distances = np.array(distances)
                probabilities = distances / distances.sum()
                next_idx = np.random.choice(len(self.X), p=probabilities)
                centroids.append(self.X[next_idx])
            self.centroids = np.array(centroids)

        # CUSTOM INITIALIZATION
        elif method == 'custom':
            # Agafa punts separats uniformement
            idx = np.linspace(0, len(self.X)-1, self.K, dtype=int)
            self.centroids = self.X[idx]
        # FALLBACK
        else:
            idx = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[idx]
        self.old_centroids = np.zeros_like(self.centroids)

    def get_labels(self):
        # Compute distances: shape (N, K)
        distances = np.linalg.norm(self.X[:, np.newaxis] - self.centroids, axis=2)
        # Assign each point to closest centroid
        self.labels = np.argmin(distances, axis=1)

    def get_centroids(self):
        # Save previous centroids
        self.old_centroids = self.centroids.copy()
        # Compute new centroids
        for k in range(self.K):
            points = self.X[self.labels == k]
            if len(points) > 0:
                self.centroids[k] = np.mean(points, axis=0)

    def converges(self):
        diff = np.linalg.norm(self.centroids - self.old_centroids)
        return diff <= self.options['tolerance']

    def fit(self):
        self._init_centroids()
        self.num_iter = 0
        while self.num_iter < self.options['max_iter']:
            self.get_labels() # Step 1: assign labels
            self.get_centroids() # Step 2: update centroids
            self.num_iter += 1
            if self.converges(): # Step 3: check convergence
                break

    def withinClassDistance(self):
        assigned_centroids = self.centroids[self.labels]
        distances = np.sum((self.X - assigned_centroids) ** 2, axis=1)
        self.WCD = np.mean(distances)
        return self.WCD


    def find_bestK(self, max_K):
        """Choose a K in [2, max_K] using a simple WCD elbow, Fisher or separation heuristic.

        This corrects an early return in the submitted version, which made it
        impossible to inspect more than the first candidate value.
        """
        if max_K < 2:
            raise ValueError('max_K must be at least 2')
        max_K = min(int(max_K), len(self.X))
        heuristic = str(self.options.get('fitting', 'WCD')).upper()
        if heuristic not in ('WCD', 'FISHER', 'INTER'):
            raise ValueError('fitting must be WCD, FISHER or INTER')

        candidate_ks = []
        wcd_values = []
        scores = []
        for k in range(2, max_K + 1):
            self.K = k
            self.fit()
            wcd = self.withinClassDistance()
            candidate_ks.append(k)
            wcd_values.append(wcd)

            if heuristic != 'WCD':
                centroids = self.centroids
                inter = sum(
                    np.linalg.norm(centroids[i] - centroids[j])
                    for i in range(k) for j in range(i + 1, k)
                )
                if heuristic == 'FISHER':
                    scores.append(inter / (wcd + 1e-6))
                else:
                    scores.append(inter / (k * (k - 1) / 2))

        if heuristic == 'WCD':
            # First K where adding the next cluster improves WCD by <20%.
            best_k = candidate_ks[-1]
            for i in range(1, len(wcd_values)):
                previous = wcd_values[i - 1]
                improvement = (previous - wcd_values[i]) / previous if previous > 0 else 0
                if improvement < 0.20:
                    best_k = candidate_ks[i - 1]
                    break
        else:
            best_k = candidate_ks[int(np.argmax(scores))]

        self.K = best_k
        self.fit()
        return self.K



def distance(X, C):
    """Pairwise Euclidean distances between N points and K centroids."""
    X, C = np.asarray(X, dtype=float), np.asarray(C, dtype=float)
    return np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)

def get_colors(centroids):
    #Converteix centroides RGB a etiquetes de color.
    probs = utils.get_color_prob(centroids)
    idx = np.argmax(probs, axis=1)
    return [utils.colors[i] for i in idx]