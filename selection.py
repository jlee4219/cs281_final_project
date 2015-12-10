import numpy as np
import time
import sklearn
import sklearn.svm as svm
import sklearn.neural_network as nn
from sklearn.feature_selection import chi2 as chi2
from sklearn.svm import SVC, LinearSVC
import my_svm2
import my_nn2

print 'Imported Selection.py'

def selectSVM_RFE(X_train, Y_train, X_test, selectivity):
	perc = 0.1
	num_to_remove = int(perc * np.shape(X_train)[1])
	iters = (1-selectivity)/perc

	#gives output as ovr shape, but really does ovo
	# clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
	# decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
	# max_iter=-1, probability=False, random_state=None, shrinking=True,
	# tol=0.001, verbose=False)
	
	# clf.probability = True #probs instead of scores
	num_features = np.shape(X_train)[1]
	cutoff = selectivity * np.shape(X_train)[1]
	print 'initial num of features:', num_features
	print 'selectivity:', selectivity, 'cutoff:', cutoff
	while iters > 0:
		clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, 
		tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, 
		intercept_scaling=1, class_weight=None, verbose=0, 
		random_state=None, max_iter=1000)

		clf.fit(X_train, Y_train)
		coefs = clf.coef_ #n_class - 1 by n_features
		# comb_coefs = np.avg(coefs, axis = 0) #1 by n_features
		# comb_coefs = np.max(coefs, axis = 0) #1 by n_features
		comb_coefs = np.sum(coefs**2, axis = 0) #1 by n_features

		comb_coefs = enumerate(comb_coefs)
		sorted_coefs = sorted(comb_coefs, key=lambda x: x[1])
		feature_indices = map(lambda x: x[0], sorted_coefs)

		# print sorted_coefs[:-num_to_remove]
		num_features -= num_to_remove
		print num_features, num_to_remove
		print feature_indices[:10]
		# to recover which features are left, we can pass in a list of the feature names
		# and delete from those too
		X_train = np.delete(X_train, feature_indices[:num_to_remove], 1)
		X_test = np.delete(X_test, feature_indices[:num_to_remove], 1)
		# features_to_remove.extend(feature_indices[:num_to_remove]) #indices of features

		iters -= 1

	return X_train, X_test


def selectNN_RFE(X_train, Y_train, X_test, Y_test, selectivity):
	classes = list(set(Y_train)) #DIFFERENT
	print classes
	perc = 0.1
	num_to_remove = int(perc * np.shape(X_train)[1])
	iters = (1-selectivity)/perc
	
	num_features = np.shape(X_train)[1]
	cutoff = selectivity * np.shape(X_train)[1]
	print 'initial num of features:', num_features
	print 'selectivity:', selectivity, 'cutoff:', cutoff
	while iters > 0:
		clf = my_nn2.NN(classes, num_iter = 10)

		clf.fit(X_train, Y_train, X_test, Y_test)
		w1, w2 = clf.get_weights() #DIFFERENT
		coefs = w1 #n_features by n_hidden nodes
		# comb_coefs = np.avg(coefs, axis = 0)
		# comb_coefs = np.max(coefs, axis = 0)
		comb_coefs = np.sum(coefs**2, axis = 1)
		print 'coefs shape:', coefs.shape

		comb_coefs = enumerate(comb_coefs)
		sorted_coefs = sorted(comb_coefs, key=lambda x: x[1])
		feature_indices = map(lambda x: x[0], sorted_coefs)

		# print sorted_coefs[:-num_to_remove]
		num_features -= num_to_remove
		print num_features, num_to_remove
		print feature_indices[:10]
		# to recover which features are left, we can pass in a list of the feature names
		# and delete from those too
		X_train = np.delete(X_train, feature_indices[:num_to_remove], 1)
		X_test = np.delete(X_test, feature_indices[:num_to_remove], 1)
		# features_to_remove.extend(feature_indices[:num_to_remove]) #indices of features

		iters -= 1

	return X_train, X_test



def selectSVM_SF(X_train, Y_train, selectivity):
	X_train_scaled = copy.deepcopy(X_train)

	#Or is it just one epoch per removal?
	num_epochs = 2
	for i in xrange(num_epochs):
		clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, 
		tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, 
		intercept_scaling=1, class_weight=None, verbose=0, 
		random_state=None, max_iter=1000)

		clf.fit(X_train_scaled, Y_train)
		coefs = clf.coef_ #n_classes by n_features
		opt_sigmas = np.sqrt(np.sum(coefs**2, axis = 0)) #1 by n_features
		X_train_scaled = np.multiply(X_train_scaled, np.sqrt(opt_sigmas))

	opt_sigmas = enumerate(opt_sigmas)
	sorted_coefs = sorted(opt_sigmas, key=lambda x: x[1], reverse = True)

	perc = 0.1
	reduced_features = sorted_coefs[:-perc*len(sorted_coefs)] #indices of features

	return reduced_features