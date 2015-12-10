import numpy as np
import time
import sklearn
import sklearn.svm as svm
import my_svm2
import my_nn2

print 'Imported model.py'

def train(X_train, Y_train, X_test, Y_test, classes):

	#gives output as ovr shape, but really does ovo
	# clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
 #    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
 #    max_iter=-1, probability=False, random_state=None, shrinking=True,
 #    tol=0.001, verbose=False)

	# clf.probability = True #probs instead of scores

	# clf = svm.LinearSVC()
	# clf = my_svm2.SVM()
	clf = my_nn2.NN(classes)
	clf.fit(X_train, Y_train, X_test, Y_test)

	return clf

def test(clf, X_test, Y_test):

	preds = clf.predict(X_test)
	acc, error = clf.score(X_test, Y_test)

	# scores = clf.decision_function(X_test) #n_samples by n_classes
	# # scores = clf.predict_proba(X_test) #probs instead of scores
	# confident_total = 0
	# confident_correct = 0
	# for i in xrange(len(preds)):
	# 	if max(abs(scores[i])) > confidence_cutoff:
	# 		if preds[i] == Y_test[i]:
	# 			confident_correct += 1
	# 		confident_total += 1
	# my_acc = 1.0*confident_correct/confident_total
	# recall = 1.0*confident_total/len(Y_test)
	# print 'accuracy:', acc, 'confident_accuracy:', conf_acc, 'recall:', recall

	return acc, error, preds
