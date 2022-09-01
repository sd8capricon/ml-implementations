import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score


class LogisticRegression():

    def __init__(self, X: np.ndarray, y: np.ndarray, b=0, alpha=0.01, iterations=1500):
        self.X = X
        self.y = y
        self.W = np.zeros(X.shape[1])
        self.b = b
        self.alpha = alpha
        self.iterations = iterations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_cost(self):

        m = self.X.shape[0]
        loss = 0

        for i in range(m):
            f_wb = self.sigmoid(np.dot(self.X[i], self.W) + self.b)
            loss += -self.y[i]*np.log(f_wb)-(1-self.y[i])*np.log(1-f_wb)

        loss /= m

        return loss

    def compute_gradient(self):
        m, n = self.X.shape
        dj_dw = np.zeros(self.W.shape)
        dj_db = 0

        for i in range(m):
            f_wb = self.sigmoid(np.dot(self.X[i], self.W) + self.b)
            # dj_dw += (f_wb-self.y[i])*self.X[i]
            dj_dw = np.add(dj_dw, (f_wb-self.y[i])*self.X[i])
            dj_db += f_wb-self.y[i]

        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db

    def gradient_descent(self):
        for i in range(self.iterations):
            dj_dw, dj_db = self.compute_gradient()
            self.W = self.W - self.alpha*dj_dw
            self.b = self.b - self.alpha*dj_db

    def predict(self, x_test, y_test) -> np.ndarray:
        m, n = x_test.shape
        self.gradient_descent()
        p = np.zeros(m)
        for i in range(m):
            f_wb = self.sigmoid(np.dot(x_test[i], self.W) + self.b)
            p[i] = 1 if f_wb > 0.5 else 0
        return p


df = pd.read_csv("data/log_data.csv")
x = df.iloc[:, 0:2].to_numpy()
y = df.iloc[:, 2].to_numpy()


log_reg = LogisticRegression(x, y)
y_pred = log_reg.predict(x, y)

print('Train Accuracy: %f' % (np.mean(y_pred == y) * 100))
