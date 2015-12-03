import numpy as np
import time
import sklearn
from sklearn.svm import SVC, LinearSVC

print 'Imported Selection.py'

def selectSVM(X_train, Y_train):

	#gives output as ovr shape, but really does ovo
	clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

	# clf.probability = True #probs instead of scores

    # clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, 
    # tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, 
    # intercept_scaling=1, class_weight=None, verbose=0, 
    # random_state=None, max_iter=1000)
	clf.fit(X_train, Y_train)

	coefs = clf.coef_ #n_class - 1 by n_features
	avg_coefs = np.avg(coefs, axis = 0) #1 by n_features
	avg_coefs = enumerate(avg_coefs)
	sorted_coefs = sorted(avg_coefs, key=lambda x: x[0], reverse = True)

	perc = 0.1
	reduced_features = sorted_coefs[:-perc*len(sorted_coefs)] #indices of features

	return reduced_features

def selectNN():
	return
