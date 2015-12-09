import numpy as np
import time

class NN():

    def __init__(self, lambd = 0.01, alpha_0 = 1e-2, alpha_final = 1e-5, mu = 0.1, 
                hidden_nodes = 32, num_iter = 40):
        self.lambd = lambd
        self.alpha_0 = alpha_0
        self.learning_const = np.log(1.0*alpha_0/alpha_final) / num_iter
        self.mu = mu
        self.hidden_nodes = hidden_nodes
        self.num_iter = num_iter

    def get_weights(self):
        return self.w1, self.w2

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
        # print probs.shape
        preds = np.argmax(probs, axis = 1)
        perc = np.mean([preds == y_inds])
        return perc

    def calculate_loss(self, probs, y_inds):
        probs_correct = probs[range(self.n_data), y_inds]
        # print 'probs:', probs[:10]
        # print 'y_inds:', y_inds[:10]
        # print 'probs_correct:', probs_correct
        logprobs_correct = -np.log(probs_correct)
        total_loss = np.sum(logprobs_correct)
        total_loss += self.lambd/2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        return 1./self.n_data * total_loss

    def fit(self, X, y):
        # y is some index of the correct category
        classes = list(set(y))
        self.classes_dict = {}
        for i in xrange(len(classes)):
            self.classes_dict[classes[i]] = i
        print self.classes_dict
        y_inds = map(lambda x: self.classes_dict[x], y)

        self.n_classes = len(classes)
        self.n_feats = X.shape[1]
        self.n_data = X.shape[0]
        self.hidden_nodes = 2*self.n_classes

        np.random.seed(1)

        #neural network has 2 hidden layers and X.shape[1] nodes in each layer
        num_nodes = [self.n_feats, self.hidden_nodes, self.n_classes]

        #initialize layer weights
        self.w1 = np.random.randn(num_nodes[0], num_nodes[1]) / np.sqrt(num_nodes[0])
        self.b1 = np.zeros((1, num_nodes[1]))
        self.w2 = np.random.randn(num_nodes[1], num_nodes[2]) / np.sqrt(num_nodes[1])
        self.b2 = np.zeros((1, num_nodes[2]))

        print 'w1:', self.w1[:2,:10]
        # print 'b1:', self.b1 
        # print 'w2:', self.w2 
        # print 'b2:', self.b2

        begin = time.time()
        for i in xrange(self.num_iter):
            lambd = 0.01
            alpha = self.alpha_0 * np.exp(-self.learning_const * i)
            alpha = 1e-4
            
            z1 = np.dot(X, self.w1) + self.b1 #n_data by n_hidden
            a1 = np.tanh(z1)
            z2 = np.dot(a1, self.w2) + self.b2 #n_data by n_classes
            probs = self.softmax(z2)

            l2_delta = np.array(probs)
            l2_delta[range(self.n_data), y_inds] -= 1 #n_data by n_classes

            l1_error = np.dot(l2_delta, self.w2.T)
            l1_delta = l1_error * self.tanh_deriv(a1) #n_data by n_hidden

            dW2 = np.dot(a1.T, l2_delta) #n_hidden by n_classes
            db2 = np.sum(l2_delta, axis = 0, keepdims = True) #1 by n_classes
            dW1 = np.dot(X.T, l1_delta)
            db1 = np.sum(l1_delta, axis = 0, keepdims = True)

            dW2 += self.lambd * self.w2
            dW1 += self.lambd * self.w1

            self.w1 += -alpha * dW1
            self.b1 += -alpha * db1
            self.w2 += -alpha * dW2
            self.b2 += -alpha * db2

            if i % (self.num_iter/10) == 0:
                print 'alpha:', alpha
                print 'w1:', self.w1[:2,:10]
                # print 'b1:', self.b1 
                # print 'w2:', self.w2 
                # print 'b2:', self.b2
                print 'Probs:', probs[:2,:10]
                print 'Loss:', self.calculate_loss(probs, y_inds)
                print 'Percent correct:', self.perc_correct(probs, y_inds)
                print time.time() - begin
                begin = time.time()

            # Has momentum
            # v2 = self.mu * v2 - alpha * np.dot(l1.T, l2_delta)
            # v1 = self.mu * v1 - alpha * np.dot(X.T, l1_delta) #sum over all data points
            # self.w2 += v2
            # self.w1 += v1
        print 'Loss:', self.calculate_loss(probs, y_inds)
        print 'Percent correct:', self.perc_correct(probs, y_inds)
        print time.time() - begin
        begin = time.time()

    def predict(self, X):
        z1 = np.dot(X, self.w1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        probs = self.softmax(z2)

    def score(self, X, y):
        probs = self.predict(X)
        y_inds = map(lambda x: self.classes_dict[x], y)
        return self.perc_correct(probs, y_inds)



        






