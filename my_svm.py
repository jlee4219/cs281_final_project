import numpy as np
import time
# import numpy.random

def linear():
    def f(x, z):
        return np.dot(x, z.T)
    return f

class SVM():
    def __init__(self, C = 1, K, b, w, E, alphas, tol = 0.01, max_passes = 1000):
        self.C = C
        self.K = K
        self.b = b
        self.w = w
        self.E = E
        self.alphas = alphas
        self.tol = tol
        self.max_passes = max_passes


    # def kernel_sum(X, Z, K):
    # 	ksum = 0
    # 	for i in xrange(X.shape[0]):
    # 		for j in xrange(Z.shape[0]):
    # 			ksum += K(X[i], Z[j])
    # 	return ksum

    # def W(alphas, X, y):
    # 	alpha_y = np.reshape(np.multiply(alphas, y), (y.shape[0], 1))
    # 	return np.sum(alphas) - 0.5 * np.sum(np.dot(alpha_y, alpha_y.T)) - 0.5 * kernel_sum(X, X)

    def second_choice(self, ind1, E1):
        max_diff = 0
        ind2 = ind1

        # if E1 > 0:
        #     return np.argmin(E)
        # elif E1 < 0:
        #     return np.argmax(E)
        # else:
        #     return ind1
        for j in self.unbound:
            E2 = np.dot(self.X[j], self.w.T) - self.b - self.y[j] #This is perhaps an inefficiency here
            if abs(E1 - E2) > max_diff:
                ind2 = j
                max_diff = abs(E1 - E2)
        return ind2

    def is_unbound(self, x):
        return (x > 0 and x < self.C)

    def roundoff(self, ind2, slack):
        if self.alphas[ind2] < slack:
            return 0
        elif self.alphas[ind2] > self.C - slack:
            return self.C

    def take_step(self, ind1, ind2):
        E1 = np.dot(self.X[ind1], self.w.T) - self.b - self.y[ind1]
        E2 = np.dot(self.X[ind2], self.w.T) - self.b - self.y[ind2]

        if ind1 == ind2:
            return 0
        eps = 1e-3
        eta = 2 * self.K(ind1, ind2) - self.K(ind1, ind1) - self.K(ind2, ind2)
        assert eta <= 0

        L = max(0, self.alphas[ind2] - self.alphas[ind1])
        H = min(self.C, self.C + self.alphas[ind2] - self.alphas[ind1])
        if L == H:
            return 0

        a2_old = self.alphas[ind2]
        if eta < 0:
            self.alphas[ind2] += self.y[ind2] * (E2 - E1)/eta
            self.alphas[ind2] = max(L, min(H, self.alphas[ind2]))
        elif eta == 0:
            obj_L = 0.5 * eta * L**2 + (y[ind2] * (E1 - E2) - eta * a2_old) * L
            obj_H = 0.5 * eta * H**2 + (y[ind2] * (E1 - E2) - eta * a2_old) * H
            if obj_L > obj_H + eps:
                self.alphas[ind2] = L
            elif obj_L < obj_H - eps:
                self.alphas[ind2] = H
        # Round
        self.alphas[ind2] = roundoff(self.alphas, ind2, 1e-8)
        if abs(self.alphas[ind2] - a2_old) < eps * (self.alphas[ind2] + a2_old + eps):
            return 0
        a2_delta = self.alphas[ind2] - a2_old
        a1_delta = -self.y[ind1] * self.y[ind2] * a2_delta
        self.alphas[ind1] += a1_delta

        # Update b
        b1 = self.b - E1 - self.y[ind1] * a1_delta * self.K(self.X[ind1], self.X[ind1]) 
        b1 -= self.y[ind2] * a2_delta * self.K(self.X[ind1], self.X[ind2])
        b2 = self.b - E2 - self.y[ind1] * a1_delta * self.K(self.X[ind1], self.X[ind2]) 
        b2 -= self.y[ind2] * a2_delta * self.K(self.X[ind2], self.X[ind2])
        if self.alphas[ind1] < self.C and self.alphas[ind1] > 0:
            self.b = b1
        elif self.alphas[ind2] < self.C and self.alphas[ind2] > 0:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0
        # print b1, b2

        #For a linear kernel
        self.w += a1_delta * self.y[ind1] * self.X[ind1] + a2_delta * self.y[ind2] * self.X[ind2]

        # # Update errors
        # t1 = a1_delta * y[ind1]
        # t2 = a2_delta * y[ind2]
        # for i in self.unbound:
        #     error = 0
        #     error += t1 * np.dot(X[ind1], X[ind1].T) 
        #     error += t2 * np.dot(X[ind2], X[ind2].T)
        #     error -= b_delta
        #     self.E[i] += error
        # # The error is now zero because we optimized for these points
        # self.E[ind1] = 0
        # self.E[ind2] = 0

        return 1

    def examine(self, ind1):
        # ind1 = np.random.randint(n_data)
        # ind2 = np.random.randint(n_data)
        # for a linear SVM
        E1 = np.dot(self.X[ind1], self.w.T) - self.b - self.y[ind1]
        # E1 = sum of alpha_i y_i <x_i, x> + b

        #Second Choice
        # If the first choice doesn't satisfy the KKT conditions
        if ((self.alphas[ind1] < self.C and E1 < -self.tol) or (self.alphas[ind1] > 0 and E1 > self.tol)):
            if len(self.unbound) > 0:
                ind2 = second_choice(ind1, E1)
                if take_step(ind1, ind2):
                    return 1
            rand_start = np.random.randint(len(self.unbound))
            for j in len(self.unbound):
                ind2 = self.unbound[(rand_start + j) % len(self.unbound)]
                if take_step(ind1, ind2):
                    return 1
            rand_start = np.random.randint(n_data)
            for j in xrange(n_data):
                ind2 = (rand_start + j) % n_data
                if take_step(ind1, ind2):
                    return 1


    def train(self, X, y):
        random.seed(1)
        self.X = X
        self.y = y

    	n_data = X.shape[0]
        n_feat = X.shape[1]
    	self.b = 0
        self.alphas = np.zeros(n_data)
        self.w = np.zeros(n_feat)

        # self.E = np.dot(self.X, self.w.T) - self.b - self.y

    	iters = 0
        numAlphasChanged = 0
        examineAll = True
        pass_num = 0
        while (numAlphasChanged > 0 or examineAll) and pass_num < max_passes:
            numAlphasChanged = 0
            if examineAll:
                start = np.random.randint(n_data)
                new_unbound = set()
                for i in xrange(n_data):
                    numAlphasChanged += examine(i)
                    if is_unbound(alphas[i]):
                        new_unbound.add(i)
                self.unbound = new_unbound #only update self.unbound here, perhaps should do it more often
                examineAll = False
            else:
                for i in self.unbound:
                    numAlphasChanged += examine(i)
                if numAlphasChanged = 0:
                    examineAll = True
            pass_num += 1
	



