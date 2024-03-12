import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target
print(y)
X1 = iris.data[:, :2]
y = iris.target
plt.scatter(X1[y == 0, 0], X1[y == 0, 1], color='r', marker='o')
plt.scatter(X1[y == 1, 0], X1[y == 1, 1], color='g', marker='*')
plt.scatter(X1[y == 2, 0], X1[y == 2, 1], color='b', marker='+')
plt.title('the relationship between sepal and target classes')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

pca = PCA(n_components=3)
de_pca = pca.fit_transform(X)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.25)

plt.scatter(de_pca[y == 0, 0], de_pca[y == 0, 1], color='r', marker='o')
plt.scatter(de_pca[y == 1, 0], de_pca[y == 1, 1], color='g', marker='*')
plt.scatter(de_pca[y == 2, 0], de_pca[y == 2, 1], color='b', marker='+')
plt.xlabel('component 0')
plt.ylabel('component 1')
plt.title('the relationship between components 0 and 1')
plt.show()

ax = plt.subplot(111, projection='3d')
ax.scatter(de_pca[y == 0, 0], de_pca[y == 0, 1], de_pca[y == 0, 2], color='r', marker='o')
ax.scatter(de_pca[y == 1, 0], de_pca[y == 1, 1], de_pca[y == 1, 2], color='g', marker='*')
ax.scatter(de_pca[y == 2, 0], de_pca[y == 2, 1], de_pca[y == 2, 2], color='b', marker='+')
ax.set_title('the relationship between components 0, 1, and 2')
ax.set(xlabel='component 0', ylabel='component 1', zlabel='component 2')
plt.show()

kpca = KernelPCA(n_components=3, kernel='rbf', gamma=0.25)
de_kpca = kpca.fit_transform(X)

ax1 = plt.subplot(111, projection='3d')
ax1.scatter(de_kpca[y == 0, 0], de_kpca[y == 0, 1], de_kpca[y == 0, 2], color='r', marker='o')
ax1.scatter(de_kpca[y == 1, 0], de_kpca[y == 1, 1], de_kpca[y == 1, 2], color='g', marker='*')
ax1.scatter(de_kpca[y == 2, 0], de_kpca[y == 2, 1], de_kpca[y == 2, 2], color='b', marker='+')
ax1.set_title('the relationship between components 0, 1, and 2')
ax1.set(xlabel='component 0', ylabel='component 1', zlabel='component 2')
plt.show()