# 导入所需的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建K近邻分类器实例
knn = KNeighborsClassifier(n_neighbors=3)

# 使用训练数据拟合分类器
knn.fit(X_train, y_train)

# 对测试集进行预测
y_pred = knn.predict(X_test)

# 计算预测的准确性
accuracy = accuracy_score(y_test, y_pred)
print("预测准确性:", accuracy)

# 创建散点图，根据鸢尾花的花萼和花瓣特征绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

# 添加图例
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')

# 添加标题
plt.title('鸢尾花数据集')

# 显示图形
plt.show()
