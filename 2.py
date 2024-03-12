# 导入所需的库和模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 加载Minist数据集
mnist = fetch_openml('mnist_784',parser='auto')

# 将数据集划分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(mnist.data, mnist.target, test_size=0.25, random_state=1)

print(X_train.max())

# 创建神经网络模型
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1,learning_rate_init = .1)

# 训练模型
mlp.fit(X_train, Y_train)

# 预测测试集
Y_pred = mlp.predict(X_test)

# 计算准确率
accuracy = mlp.score(X_test, Y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# 获取隐藏层权重
hidden_layer_weights = mlp.coefs_[0]
vmin,vmax = hidden_layer_weights.min(), hidden_layer_weights.max()

# 可视化隐藏层权重图像
fig, axs = plt.subplots(4,4)
for coef,ax in zip(hidden_layer_weights.T, axs.ravel()):
    ax.matshow(coef.reshape(28,28),cmap=plt.cm.gray,vmin=.5*vmin,vmax=.5*vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()