import numpy as np
import time
import copy
import matplotlib.pyplot as plt

class NN():

    def __init__(self, classes, lambd = 3e-1, alpha_0 = 1, alpha_final = 1e-2, mu = 0.1, 
                hidden_nodes = 32, num_iter = 10):
        self.classes = classes
        self.lambd = lambd
        self.alpha_0 = alpha_0
        self.learning_const = np.log(1.0*alpha_0/alpha_final) / num_iter
        self.mu = mu
        self.hidden_nodes = hidden_nodes
        self.num_iter = num_iter

    def get_weights(self):
        return self.w1, self.w2

    def get_weight_updates(self):
        return self.weight_updates

    def get_traintest_erroraccs(self):
        return self.train_errors, self.train_accs, self.test_errors, self.test_accs        

    def plot_weight_updates(self):
        # print self.weight_updates
        x = range(len(self.weight_updates))
        plt.plot(x, self.weight_updates)

    # Returns a k-dimensional vector
    def softmax(self, x):
        #w is a n_classes by n_feats
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims = True)

    # The derivative turns out to be extremely simple when combined with entropy loss
    # def softmax_deriv(w, x):

    def tanh_deriv(self, tanh):
        return (1 - np.power(tanh, 2))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, sig):
        return sig * (1 - sig)

    def perc_correct(self, probs, y_inds):
        preds = np.argmax(probs, axis=1)
        return np.mean([preds == y_inds])

    def calculate_loss(self, probs, y_inds):
        probs_correct = probs[range(len(y_inds)), y_inds]
        logprobs_correct = -np.log(probs_correct)
        total_loss = np.sum(logprobs_correct)
        total_loss += self.lambd/2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        return 1./self.n_data * total_loss

    def fit(self, X, y, X_test, y_test):
        y = np.reshape(y, (y.shape[0], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        begin = time.time()
        self.classes_dict = {}
        for i in xrange(len(self.classes)):
            self.classes_dict[self.classes[i]] = i
        print self.classes_dict
        # y_inds are the indices of the correct categories
        y_inds = map(lambda x: self.classes_dict[x[0]], y)

        self.n_classes = len(self.classes)
        self.n_feats = X.shape[1]
        self.n_data = X.shape[0]
        self.hidden_nodes = self.n_classes

        np.random.seed(1)
        #neural network has 2 hidden layers and X.shape[1] nodes in each layer
        num_nodes = [self.n_feats, self.hidden_nodes, self.n_classes]

        #initialize layer weights
        self.w1 = np.random.randn(num_nodes[0], num_nodes[1]) / np.sqrt(num_nodes[0])
        self.b1 = np.zeros((1, num_nodes[1]))
        self.w2 = np.random.randn(num_nodes[1], num_nodes[2]) / np.sqrt(num_nodes[1])
        self.b2 = np.zeros((1, num_nodes[2]))
        self.weight_updates = []
        self.weight_means = []
        self.weight_vars = []

        self.train_errors = []
        self.train_accs = []
        self.test_errors = []
        self.test_accs = []

        print 'w1:', self.w1[:2,:10]
        
        comb = np.hstack((X, np.reshape(y_inds, (len(y_inds), 1) ) ))
        print 'setup time:', time.time() - begin
        self.minibatch_size = 100
        for i in xrange(self.num_iter):
            begin = time.time()
            alpha = self.alpha_0 * np.exp(-self.learning_const * i)
            X_perm = X
            comb2 = np.random.permutation(comb)
            X_perm = copy.deepcopy(comb2[:, :-1])
            y_perm = copy.deepcopy(comb2[:, -1])
            y_inds = map(int, list(y_perm))
            print 'iter:', i, 'permute time:', time.time() - begin

            begin = time.time()
            for j in xrange(0, self.n_data, self.minibatch_size):
                # Make minibatch
                end_ind = min(self.n_data, j+self.minibatch_size)
                minibatch = X_perm[j:end_ind]
                minibatch_y = y_inds[j:end_ind]
                actual_size = minibatch.shape[0]

                z1 = np.dot(minibatch, self.w1) + self.b1 #n_data by n_hidden
                a1 = np.tanh(z1)
                z2 = np.dot(a1, self.w2) + self.b2 #n_data by n_classes
                probs = self.softmax(z2)

                l2_delta = np.array(probs)
                l2_delta[range(actual_size), minibatch_y] -= 1 #n_data by n_classes

                l1_error = np.dot(l2_delta, self.w2.T)
                l1_delta = l1_error * self.tanh_deriv(a1) #n_data by n_hidden

                dW2 = np.dot(a1.T, l2_delta) #n_hidden by n_classes
                db2 = np.sum(l2_delta, axis = 0, keepdims = True) #1 by n_classes
                dW1 = np.dot(minibatch.T, l1_delta)
                db1 = np.sum(l1_delta, axis = 0, keepdims = True)

                dW2 += self.lambd * self.w2
                dW1 += self.lambd * self.w1

                self.w1 += -alpha / actual_size * dW1
                self.b1 += -alpha / actual_size * db1
                self.w2 += -alpha / actual_size * dW2
                self.b2 += -alpha / actual_size * db2

                # Has momentum
                # v2 = self.mu * v2 - alpha * np.dot(l1.T, l2_delta)
                # v1 = self.mu * v1 - alpha * np.dot(X.T, l1_delta) #sum over all data points
                # self.w2 += v2
                # self.w1 += v1

                weight_update = 0
                weight_update += np.sum(np.power(dW1, 2))
                weight_update += np.sum(np.power(db1, 2))
                weight_update += np.sum(np.power(dW2, 2))
                weight_update += np.sum(np.power(db2, 2))
                self.weight_updates.append(weight_update)
            print 'epoch time:', time.time() - begin

            begin = time.time()
            print 'alpha:', alpha
            # print 'w1:', self.w1[0]
            # print 'Probs:', probs[0]

            probs = self.predict_prob(X_perm)
            a = self.calculate_loss(probs, y_inds)
            self.train_errors.append(a)
            print 'Loss:', a
            a = self.perc_correct(probs, y_inds)
            self.train_accs.append(a)
            print 'Percent correct:', a

            y_test_inds = map(lambda x: self.classes_dict[x[0]], y_test)
            probs = self.predict_prob(X_test)
            a = self.calculate_loss(probs, y_test_inds)
            self.test_errors.append(a)
            print 'Test Loss:', a
            a = self.perc_correct(probs, y_test_inds)
            self.test_accs.append(a)
            print 'Test Percent correct:', a

            print 'score time:', time.time() - begin


    def predict_prob(self, X):
        z1 = np.dot(X, self.w1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        probs = self.softmax(z2)
        return probs

    def predict(self, X):
        probs = self.predict_prob(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        y = np.reshape(y, (y.shape[0], 1))
        probs = self.predict_prob(X)
        y_inds = map(lambda x: self.classes_dict[x[0]], y)
        return self.perc_correct(probs, y_inds), self.calculate_loss(probs, y_inds)



        






