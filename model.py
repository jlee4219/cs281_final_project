import numpy as np
import time
import sklearn
import sklearn.svm as svm
from sklearn.svm import SVC, LinearSVC

print 'Imported model.py'

def train(X_train, Y_train):

	#gives output as ovr shape, but really does ovo
	# clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
 #    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
 #    max_iter=-1, probability=False, random_state=None, shrinking=True,
 #    tol=0.001, verbose=False)

	# clf.probability = True #probs instead of scores

	clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, 
    tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, 
    intercept_scaling=1, class_weight=None, verbose=0, 
    random_state=None, max_iter=1000)

	clf.fit(X_train, Y_train)
	return clf

def test(clf, X_test, Y_test):
	confidence_cutoff = 1

	scores = np.abs(clf.decision_function(X_test)) #n_samples by n_classes
	# scores = clf.predict_proba(X_test) #probs instead of scores

	preds = clf.predict(X_test)
	acc = clf.score(X_test, Y_test)

	for i in xrange(len(preds)):
		if max(scores[i]) <= confidence_cutoff:
			preds[i] = '-1'
	my_acc = 1.0*sum(preds == Y_test)/len(Y_test)

	return acc, my_acc, preds, scores
