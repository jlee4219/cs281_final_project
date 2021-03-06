import numpy as np
import time
import random
import numpy.random
import math

def linear():
    def f(x, z):
        return np.dot(x, z.T)
    return f

class SVM():
    def __init__(self, K, C = 1, tol = 0.01, max_passes = 1000):
        self.K = K
        self.C = C
        self.E = {}
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

    def is_unbound(self, x):
        return (x > 0 and x < self.C)

    def roundoff(self, a2, slack):
        if a2 < slack:
            return 0.0
        elif a2 > self.C - slack:
            return self.C
        else:
            return a2

    def getE(self, j):
        if self.alphas[j] > 0 and self.alphas[j] < self.C and j in self.E:
            return self.E[j]
        else:
            self.E[j] = np.dot(self.X[j], self.w.T) - self.b - self.y[j]
            return self.E[j]

    def second_choice(self, ind1):
        max_diff = 0
        ind2 = ind1

        # if E1 > 0:
        #     return np.argmin(E)
        # elif E1 < 0:
        #     return np.argmax(E)
        # else:
        #     return ind1
        E1 = self.getE(ind1)
        for j in self.unbound:
            E2 = self.getE(j)
            if abs(E1 - E2) > max_diff:
                ind2 = j
                max_diff = abs(E1 - E2)
        return ind2

    def take_step(self, ind1, ind2):
        E1 = self.getE(ind1)
        E2 = self.getE(ind2)
        a2 = 0.0

        if ind1 == ind2:
            return 0
        eps = 1e-3
        K12 = self.K(self.X[ind1,:], self.X[ind2,:])
        K11 = self.K(self.X[ind1,:], self.X[ind1,:])
        K22 = self.K(self.X[ind2,:], self.X[ind2,:])
        eta = 2 * K12 - K11 - K22
        assert eta <= 0

        if self.y[ind1] != self.y[ind2]:
            L = max(0, self.alphas[ind2] - self.alphas[ind1])
            H = min(self.C, self.C + self.alphas[ind2] - self.alphas[ind1])
        else:
            L = max(0, self.alphas[ind1] + self.alphas[ind2] - self.C)
            H = min(self.C, self.alphas[ind1] + self.alphas[ind2])
        if L == H:
            return 0

        a2_old = self.alphas[ind2]
        if eta < 0:
            a2 += self.y[ind2] * (E2 - E1)/eta
            a2 = max(L, min(H, a2))
        elif eta == 0:
            c2 = (self.y[ind2] * (E1 - E2) - eta * a2_old)
            obj_L = 0.5 * eta * L**2 + c2 * L
            obj_H = 0.5 * eta * H**2 + c2 * H
            if obj_L > obj_H + eps:
                a2 = L
            elif obj_L < obj_H - eps:
                a2 = H

        # Round
        # print self.alphas[ind2]
        a2 = self.roundoff(a2, 1e-8)

        # print self.alphas[ind2]
        if abs(a2 - a2_old) < eps * (a2 + a2_old + eps):
            return 0

        s = self.y[ind1] * self.y[ind2]
        a2_delta = a2 - a2_old
        a1_old = self.alphas[ind1]
        a1_delta = -self.y[ind1] * self.y[ind2] * a2_delta
        self.alphas[ind1] += a1_delta

        if (self.alphas[ind1] < 0):
            a2 += s * self.alphas[ind1]
            self.alphas[ind1] = 0
        elif (self.alphas[ind1] > self.C):
            t = float(self.alphas[ind1] - self.C)
            a2 += s * t
            self.alphas[ind1] = self.C

        a1_delta = self.alphas[ind1] - a1_old
        a2_delta = a2 - a2_old

        # Update b
        b1 = self.b + E1 + self.y[ind1] * a1_delta * K11
        b1 += self.y[ind2] * a2_delta * K12
        b2 = self.b + E2 + self.y[ind1] * a1_delta * K12
        b2 += self.y[ind2] * a2_delta * K22
        b_old = self.b
        if self.alphas[ind1] < self.C and self.alphas[ind1] > 0:
            self.b = b1
        elif a2 < self.C and a2 > 0:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2.0

        b_delta = self.b - b_old
        # print b1, b2

        #For a linear kernel
        self.w += a1_delta * self.y[ind1] * self.X[ind1,:] + a2_delta * self.y[ind2] * self.X[ind2,:]

        # Update errors
        t1 = a1_delta * self.y[ind1]
        t2 = a2_delta * self.y[ind2]
        for i in self.unbound:
            part1 = t1 * np.dot(self.X[ind1,:], self.X[i,:].T)
            part2 = t2 * np.dot(self.X[ind2,:], self.X[i,:].T)  
            self.E[i] = part1 + part2 - b_delta
        # The error is now zero because we optimized for these points
        self.E[ind1] = 0
        self.E[ind2] = 0
        self.alphas[ind2] = a2
        if self.is_unbound(a2):
            self.unbound.add(a2)
        else:
            self.unbound.discard(a2)

        return 1

    def examine(self, ind1):
        # ind1 = np.random.randint(n_data)
        # ind2 = np.random.randint(n_data)se
        # for a linear SVM
        alph1 = self.alphas[ind1]
        # E1 = sum of alpha_i y_i <x_i, x> + b
        E1 = self.getE(ind1)
        r1 = E1 * self.y[ind1]

        #Second Choice
        # If the first choice doesn't satisfy the KKT conditions
        if ((alph1 < self.C and r1 < -self.tol) or (alph1 > 0 and r1 > self.tol)):
            if len(self.unbound) > 0:
                ind2 = self.second_choice(ind1)
                if self.take_step(ind1, ind2):
                    return 1
                rand_start = np.random.randint(len(self.unbound))
                ub_list = list(self.unbound)
                for j in range(len(self.unbound)):
                    ind2 = ub_list[(rand_start + j) % len(self.unbound)]
                    if self.take_step(ind1, ind2):
                        return 1
            rand_start = np.random.randint(self.n_data)
            for j in xrange(self.n_data):
                ind2 = (rand_start + j) % self.n_data
                if self.take_step(ind1, ind2):
                    return 1
        return 0


    def fit(self, X, y):
        random.seed(1)
        self.X = X
        self.y = y

    	n_data = X.shape[0]
        self.n_data = n_data
        n_feat = X.shape[1]
    	self.b = 0
        self.alphas = np.zeros(n_data)
        self.w = np.zeros(n_feat)

    	iters = 0
        numAlphasChanged = 0
        examineAll = True
        pass_num = 0
        self.unbound = set()
        while (numAlphasChanged > 0 or examineAll) and pass_num < self.max_passes:
            numAlphasChanged = 0
            if examineAll:
                # start = np.random.randint(n_data)
                new_unbound = set()
                for i in xrange(n_data):
                    numAlphasChanged += self.examine(i)
                    if self.is_unbound(self.alphas[i]):
                        self.unbound.add(i)
                    else:
                        self.unbound.discard(i)
                examineAll = False
            else:
                # print len(self.unbound)
                for i in xrange(n_data):
                    if self.alphas[i] != 0 and self.alphas[i] != self.C:
                        numAlphasChanged += self.examine(i)
                if numAlphasChanged == 0:
                    examineAll = True
            '''new_unbound = set()
            for i in xrange(n_data):
                idx = i
                if self.is_unbound(self.alphas[idx]):
                    new_unbound.add(idx)
                self.unbound = new_unbound''' #only update self.unbound here, perhaps should do it more often
            if pass_num % 100 == 0:
                print numAlphasChanged, pass_num
            pass_num += 1

    def get_margin(self, X_test):
        return np.dot(X_test, self.w) - self.b
	
    def predict(self, X_test):
        vals = self.get_margin(X_test)
        vals[np.where(vals > 0)] = 1
        vals[np.where(vals <= 0)] = -1
        return vals

    def score(self, X_test, y_test):
        preds = self.predict(X_test)
        return len(np.where(y_test == preds)[0]) / float(len(y_test))
