import numpy as np
from sklearn.cluster import KMeans
import cv2

def run_kmeans(X, K):
    X_ravel = X.reshape(-1, X.shape[-1])
    if isinstance(K, int):
        K = [K]
    transformed_images = []
    for k in K:
        print(f'Fitting k={k}...')
        model = KMeans(k).fit(X_ravel)
        transformed_images.append(model.cluster_centers_[model.labels_].astype(np.uint8).reshape(X.shape))
    return transformed_images if len(transformed_images) > 1 else transformed_images[0]


image = cv2.imread('bird.jpg')
K = [2, 4, 8, 16, 32]

# (1)
transformed_images = run_kmeans(image, K)
for img, k in zip(transformed_images, K):
    cv2.imwrite(f'1-1_{k}.jpg', img)

# (2)
image_with_XY = np.concatenate([image, np.array([[i, j] for i in range(image.shape[0]) for j in range(image.shape[1])]).reshape(image.shape[:-1] + (-1,))], axis=-1)
transformed_images = run_kmeans(image_with_XY, K)
for img, k in zip(transformed_images, K):
    cv2.imwrite(f'1-2_{k}.jpg', img[...,:3])

