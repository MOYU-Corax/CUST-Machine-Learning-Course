# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA, KernelPCA

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 使用PCA进行降维
pca = PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

# 使用KPCA进行降维
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1e-3)
kpca.fit(X)
X_kpca = kpca.transform(X)

# 创建一个散点图，将PCA和KPCA的结果可视化
plt.figure(figsize=(8, 4))

# 使用PCA进行降维的鸢尾花数据
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('viridis', 3))
plt.title('PCA')

# 使用KPCA进行降维的鸢尾花数据
plt.subplot(1, 2, 2)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('viridis', 3))
plt.title('KPCA')

plt.tight_layout()
plt.show()
