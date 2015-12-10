import numpy as np
import time

class NN():

    def __init__(self, alpha_0 = 0.1, learning_rate = 1e-5, mu = 0.1, hidden_nodes = 32, num_iter = 100000):
        self.alpha_0 = alpha_0
        self.learning_rate = learning_rate
        self.mu = mu
        self.hidden_nodes = hidden_nodes
        self.num_iter = num_iter

    def get_weights(self):
        return self.w1, self.w2

    # Returns a k-dimensional vector
    def softmax(x):
        #w is a n_classes by n_feats
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims = True)

    # The derivative turns out to be extremely simple when combined with entropy loss
    # def softmax_deriv(w, x):

    def tanh_deriv(tanh):
        return (1 - np.power(tanh, 2))

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(sig):
        return sig * (1 - sig)

    def perc_correct(probs, actuals):
        preds = labels[np.argmax(probs, axis = 1)]
        # actuals = np.nonzero(y)[1]
        perc = np.mean([preds == actuals])
        return perc

    def fit(self, X, y):
        # y is some index of the correct category

        # y = np.reshape(y, (y.shape[0], 1))
        self.n_classes = y.shape[1]
        self.n_feats = X.shape[1]
        self.n_data = X.shape[0]

        np.random.seed(1)

        #neural network has 2 hidden layers and X.shape[1] nodes in each layer
        num_nodes = [self.n_feats, self.hidden_nodes, self.n_classes]

        #initialize layer weights
        # self.w1 = 2 * np.random.random((num_nodes[0], num_nodes[1])) - 1
        # self.w2 = 2 * np.random.random((num_nodes[1], num_nodes[2])) - 1
        self.w1 = np.random.randn(num_nodes[0], num_nodes[1]) / np.sqrt(num_nodes[0])
        self.b1 = np.zeros((1, num_nodes[1]))
        self.w2 = np.random.randn(num_nodes[1], num_nodes[2]) / np.sqrt(num_nodes[1])
        self.b2 = np.zeros((1, num_nodes[2]))

        begin = 0
        for i in xrange(self.num_iter):
            lambd = 0.01
            alpha = self.alpha_0 * np.exp(-self.learning_const * i)
            # alpha = 0.5
            if i % (self.num_iter/10) == 0:
                print 'Error:', np.mean(np.abs(l2_error))
                print 'Percent correct:', perc_correct(l2, y)
                print time.time() - begin
                begin = time.time()
            
            z1 = np.dot(X, self.w1) + self.b1 #n_data by n_hidden
            a1 = np.tanh(z1)
            z2 = np.dot(a1, self.w2) + self.b2 #n_data by n_classes
            probs = softmax(z2)

            l2_delta = probs 
            l2_delta[range(self.n_data), y] -= 1 #n_data by n_classes

            l1_error = np.dot(l2_delta, self.w2.T)
            l1_delta = l1_error * tanh_deriv(a1) #n_data by n_hidden

            dW2 = np.dot(a1.T, l2_delta) #n_hidden by n_classes
            db2 = np.sum(l2_delta, axis = 0, keepdims = True) #1 by n_classes
            dW1 = np.dot(X.T, l1_delta)
            db1 = np.sum(l1_delta, axis = 0)

            dW2 += lambd * self.w2
            dW1 += lambd * self.w1

            self.w1 += -alpha * dW1
            self.b1 += -alpha * db1
            self.w2 += -alpha * dW2
            self.b2 += -alpha * db2

            # Has momentum
            # v2 = self.mu * v2 - alpha * np.dot(l1.T, l2_delta)
            # v1 = self.mu * v1 - alpha * np.dot(X.T, l1_delta) #sum over all data points
            # self.w2 += v2
            # self.w1 += v1

    def predict(self, X, y):
        z1 = np.dot(X, self.w1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        probs = softmax(z2)

    def score(self, X, y):
        probs = self.predict(X, y)
        return perc_correct(probs, y)



        






