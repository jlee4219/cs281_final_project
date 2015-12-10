import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import my_svm2
import multiclass_svm
import time

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
# y[np.where(y != 0 )] = -1
# y[np.where(y == 0)] = 1
h = .02
C = 1.0
# svc = my_svm2.SVM(1.0, my_svm2.linear(), None, None, None)
svc = multiclass_svm.MultiClassSVM()
begin = time.time()
svc.fit(X, y)
print "Trained:", time.time() - begin

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
print set(Z)
print Z
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()