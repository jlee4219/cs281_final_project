import numpy as np
import my_svm2
import copy

class MultiClassSVM:
  def __init__(self, C = 1.0, K = my_svm2.linear()):
    self.C = C
    self.K = K

  def fit(self, X_train, y_train):
    self.labels = list(set(y_train))
    self.num_classes = len(self.labels)
    self.svms = [my_svm2.SVM(self.C, self.K, None, None, None) for i in range(self.num_classes)]
    for i in range(len(self.labels)):
      label = self.labels[i]
      y_class_train = np.zeros(y_train.shape)
      y_class_train[np.where(y_train == label)] = 1
      y_class_train[np.where(y_train != label)] = -1
      self.svms[i].train(X_train, y_class_train)

  def predict(self, X_test):
    full_preds = np.zeros((self.num_classes, X_test.shape[0]))
    for i in range(self.num_classes):
      full_preds[i, :] = self.svms[i].get_margin(X_test)
    idxs = np.array(np.argmax(full_preds, axis=0), dtype = int)
    return np.array(self.labels)[idxs]

  def score(self, X_test, y_test):
    preds = self.predict(X_test)
    num_correct = len(np.where(y_test==preds)[0])
    return float(num_correct) / len(y_test)
