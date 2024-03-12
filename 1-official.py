import numpy as np
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target
plt.scatter(x[y == 0,0],x[y == 0,1],color='r',marker='o')
plt.scatter(x[y == 1,0],x[y == 1,1],color='g',marker='*')
plt.scatter(x[y == 2,0],x[y == 2,1],color='b',marker='+')

plt.title("the relationship between sepal and target classes")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.show()

x_train,x_test,y_train,y_test = (
    train_test_split(iris.data[:,:2],
                     iris.target,test_size=0.2,random_state=42)
)
lin_svc = svm.SVC(kernel='linear').fit(x_train,y_train)
rbf_svc = svm.SVC(kernel='rbf').fit(x_train,y_train)
poly_svc = svm.SVC(kernel='poly').fit(x_train,y_train)

h = 0.02
x_min, x_max = x_train[:, 0].min()-1, x_train[:, 0].max()+1
y_min, y_max = x_train[:, 0].min()-1, x_train[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
titles = ['LinearSVC(linear kernel)',
          'SVC with rbf kernel', 'SVC with polynom1al(degree=3) kernel']
for i, clf in enumerate((lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.1)
    plt.scatter(x[:, 0],x[:, 1],c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.xticks(())
    plt.yticks(())

plt.show()

lin_svc_pre = lin_svc.predict(x_test)
acc_lin_svc = sum(lin_svc_pre==y_test)/len(y_test)
rbf_svc_pre = rbf_svc.predict(x_test)
acc_rbf_svc = sum(rbf_svc_pre==y_test)/len(y_test)
poly_svc_pre = poly_svc.predict(x_test)
acc_poly_svc = sum(poly_svc_pre==y_test)/len(y_test)

print(acc_lin_svc)
print(acc_rbf_svc)
print(acc_poly_svc)