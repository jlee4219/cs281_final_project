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

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(sig):
        return sig * (1 - sig)

    def perc_correct(preds, actual):
        return np.mean(actual == [preds >= 0.5])

    def train(self, X, y):
        y = np.reshape(y, (y.shape[0], 1))

        # X = np.array([[0, 0, 1],
        #              [0, 1, 1],
        #              [1, 0, 1],
        #              [1, 1, 1]])
        # y = np.array([[0],
        #              [1],
        #              [1],
        #              [0]])

        np.random.seed(1)

        #neural network has 2 hidden layers and X.shape[1] nodes in each layer
        num_nodes = [X.shape[1], self.hidden_nodes, 1]

        #initialize layer weights
        self.w1 = 2 * np.random.random((num_nodes[0], num_nodes[1])) - 1
        self.w2 = 2 * np.random.random((num_nodes[1], num_nodes[2])) - 1
        v1 = 0
        v2 = 0

        start_time = 0
        for i in xrange(self.num_iter):
            alpha = self.alpha_0 * np.exp(-self.learning_const * i)
        #     alpha = 0.5
            if i % (self.num_iter/10) == 0:
                print 'Error:', np.mean(np.abs(l2_error))
                print 'Percent correct:', perc_correct(l2, y)
        #         print time.time() - start_time
                start_time = time.time()
            
            l1 = sigmoid(np.dot(X, w1)) #num_data x num_nodes
            l2 = sigmoid(np.dot(l1, w2)) #num_data x 1
            
            l2_error = l2 - y
            l2_delta = l2_error * sigmoid_deriv(l2) # [4 x 1] dot [4 x 1]
            
            l1_error = np.dot(l2_delta, w2.T) #### [4 x 1] x [1 by 3]
            l1_delta = l1_error * sigmoid_deriv(l1)

            # v2 = - alpha * np.dot(l1.T, l2_delta)
            # v1 = - alpha * np.dot(X.T, l1_delta)
            # Has momentum
            v2 = self.mu * v2 - alpha * np.dot(l1.T, l2_delta)
            v1 = self.mu * v1 - alpha * np.dot(X.T, l1_delta) #sum over all data points
            self.w2 += v2
            self.w1 += v1