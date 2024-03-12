import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import pandas as pd

# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 对数据进行标准化处理
transfer = StandardScaler()
X_train = transfer.fit_transform(X_train)
X_test = transfer.transform(X_test)

# 创建KNN分类器
knn = KNeighborsClassifier(n_neighbors=10, p=2)
knn.fit(X_train, y_train)

# 预测测试集
y_pre = knn.predict(X_test)

# 计算准确率
score = knn.score(X_test, y_test)
print('直接对比真实值与预测值：\n', y_test == y_pre)
print('accuracy: \n', score)

# 提取Iris数据集的前两个特征
X1 = iris.data[:, :2]

# 使用KNN分类器对前两个特征进行分类
knn = KNeighborsClassifier(n_neighbors=10, p=2)
knn.fit(X1, y)

# 绘制决策边界
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 'x', 's', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0], X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

# 绘制决策边界
plot_decision_regions(X1, y, classifier=knn)
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
