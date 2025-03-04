import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from scipy.io import loadmat

mnist= loadmat('mnist-original.mat')
X=mnist['data'].T
X=X/255
y = mnist['label'].T.flatten()
y = y.astype(np.uint8)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


mlp = MLPClassifier(
hidden_layer_sizes=(50,),
max_iter=100,
alpha=1e-4,
solver='sgd',
verbose=10,
tol=1e-3,
random_state = 1,
learning_rate_init = .1
)

mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef,ax in zip(mlp.coefs_[0].T,axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap = plt.cm.gray, vmin = .5 * vmin, vmax = .5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()