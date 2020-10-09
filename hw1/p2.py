import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import cv2


def load_data(data_dir):
    trainX = np.array([cv2.imread(os.path.join(data_dir, f'{person}_{i}.png'), cv2.IMREAD_GRAYSCALE) for person in range(1, 41) for i in range(1, 10)], np.float64)
    trainY = np.arange(40 * 9) // 9

    testX = np.array([cv2.imread(os.path.join(data_dir, f'{person}_10.png'), cv2.IMREAD_GRAYSCALE) for person in range(1, 41)], np.float64)
    testY = np.arange(40)
    return (trainX, trainY), (testX, testY)


def get_cross_validation_data(X, Y, nfold):
    X = X.reshape(40, 9, -1)
    Y = Y.reshape(40, 9)
    idx = list(range(X.shape[1]))
    for i in range(nfold):
        train_idx = idx[:len(idx) * i // nfold] + idx[len(idx) * (i + 1) // nfold:]
        valid_idx = idx[len(idx) * i // nfold:len(idx) * (i + 1) // nfold]
        yield (X[:, train_idx].reshape(-1, X.shape[-1]), Y[:, train_idx].ravel()), (X[:, valid_idx].reshape(-1, X.shape[-1]), Y[:, valid_idx].ravel())


(trainX, trainY), (testX, testY) = load_data('p2_data')
print(trainX.shape)
img_shape = trainX.shape[1:]
trainX, testX = trainX.reshape(trainX.shape[0], -1), testX.reshape(testX.shape[0], -1)

model = PCA(345).fit(trainX)
mean_face = np.mean(trainX, axis=0)
# (1)
cv2.imwrite(f'2-1_mean.png', mean_face.reshape(img_shape))
for i, eigen_face in enumerate(model.components_[:4]):
    cv2.imwrite(f'2-1_{i}.png', ((eigen_face - np.min(eigen_face)) / (np.max(eigen_face) - np.min(eigen_face)) * 255).astype(np.uint8).reshape(img_shape))

# (2)
coef = model.components_ @ (trainX[9] - mean_face)
imgs = []
for n in [3, 50, 170, 240, 345]:
    img = np.clip(model.components_[:n].T @ coef[:n] + mean_face, 0, 255)
    imgs.append(img)
    cv2.imwrite(f'2-2_{n}.png', img.astype(np.uint8).reshape(img_shape))

# (3)
imgs = np.array(imgs)
print('mse:', ('{:.03f} ' * imgs.shape[0]).format(*np.mean(np.square(trainX[9:10] - imgs), axis=1)))

# (4)
nfold = 3
best_score, best_params = 0., {'k': None, 'n': None}
for k in [1, 3, 5]:
    for n in [3, 50, 170]:
        print(f'k: {k}, n: {n}')
        train_acc, valid_acc = 0., 0.
        for (X, Y), (validX, validY) in get_cross_validation_data(trainX, trainY, nfold):
            m = PCA(n).fit(X)
            imgs = m.inverse_transform(m.transform(X))
            valid_imgs = m.inverse_transform(m.transform(validX))
            
            knn = KNeighborsClassifier(k).fit(imgs, Y)

            train_score = knn.score(imgs, Y)
            valid_score = knn.score(valid_imgs, validY)
            print(f'train acc: {train_score:.05f}, valid acc: {valid_score:.05f}', end='\r')
            train_acc += train_score
            valid_acc += valid_score
        if valid_acc > best_score:
            best_score = valid_acc
            best_params.update({'k': k, 'n': n})
        print(f'train acc: {train_acc / nfold:.05f}, valid acc: {valid_acc / nfold:.05f}')


print(f'best score: {best_score / nfold:.05f}, best k: {best_params["k"]}, best n: {best_params["n"]}')

# (5)
m = PCA(best_params['n']).fit(trainX)
imgs = m.inverse_transform(m.transform(trainX))
knn = KNeighborsClassifier(best_params['k']).fit(imgs, trainY)

test_imgs = m.inverse_transform(m.transform(testX))
print(f'test acc: {knn.score(test_imgs, testY):.05f}')
