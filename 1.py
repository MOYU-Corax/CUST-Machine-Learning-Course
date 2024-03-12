import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl


# 加载Iris数据集
iris = datasets.load_iris()

# 特征
iris_feature = iris.data[:,:2]
# 分类标签
iris_label = iris.target


# 划分
X_train, X_test, Y_train, Y_test = train_test_split(iris_feature, iris_label, test_size=0.3, random_state=42)

# 使用SVM进行拟合
svm_classifier = svm.SVC(kernel='rbf')
svm_classifier.fit(X_train, Y_train)

# 对测试集进行预测
Y_pred = svm_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(Y_test, Y_pred)
print("准确率:", accuracy)


# (5)绘制图像
# 1.确定坐标轴范围，x，y轴分别表示两个特征
x1_min, x1_max = iris_feature[:, 0].min(), iris_feature[:, 0].max()  # 第0列的范围  x[:, 0] "："表示所有行，0表示第1列
x2_min, x2_max = iris_feature[:, 1].min(), iris_feature[:, 1].max()  # 第1列的范围  x[:, 0] "："表示所有行，1表示第2列
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点（用meshgrid函数生成两个网格矩阵X1和X2）
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点，再通过stack()函数，axis=1，生成测试点
# .flat 将矩阵转变成一维数组 （与ravel()的区别：flatten：返回的是拷贝
 
print("grid_test = \n", grid_test)
# print("x = \n",x)
grid_hat = svm_classifier.predict(grid_test)       # 预测分类值
 
print("grid_hat = \n", grid_hat)
# print(x1.shape())
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
 
 
# 2.指定默认字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
 
# 3.绘制
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
 
alpha=0.5
 
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light) # 预测值的显示
# plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
plt.plot(iris_feature[:, 0], iris_feature[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
plt.scatter(X_test[:, 0], X_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'花萼长度', fontsize=13)
plt.ylabel(u'花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM二特征分类', fontsize=15)
# plt.grid()
plt.show()