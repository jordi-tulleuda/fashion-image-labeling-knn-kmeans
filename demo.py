"""Reproducible demonstration on synthetic data; no course datasets needed."""
import numpy as np
from KNN import KNN
from Kmeans import KMeans, get_colors

def main():
    # Small RGB images with different arrangements, representing two classes.
    dark = np.array([30, 30, 30], dtype=np.uint8)
    bright = np.array([230, 230, 230], dtype=np.uint8)
    horizontal = np.tile(np.array([dark, bright, dark, bright], dtype=np.uint8), (4, 1, 1))
    vertical = np.transpose(horizontal, (1, 0, 2))
    images = np.stack([horizontal, vertical])
    knn = KNN(images, ['horizontal', 'vertical'])
    print('KNN predictions for two synthetic patterns:')
    for metric in ('euclidean', 'cityblock', 'cosine'):
        print(f'  {metric}: {knn.predict(images, k=1, distance=metric).tolist()}')

    # Synthetic red/blue image, no third-party images required.
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :4, :] = [230, 25, 20]
    img[:, 4:, :] = [15, 50, 235]
    km = KMeans(img, K=2, options={'km_init': 'first', 'max_iter': 50, 'tolerance': 1e-6})
    km.fit()
    print('K-means centroids:', np.round(km.centroids).astype(int).tolist())
    print('Approximate color labels:', get_colors(km.centroids))

if __name__ == '__main__':
    main()
