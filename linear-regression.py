from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x_train = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
y_train = np.array([5, 6, 4, 5, 6, 7], dtype=np.float64)


class LinearRegressionGD():
    def __init__(self, x_in: np.ndarray, y_in: np.ndarray, alpha=0.01, num_iters=1500) -> None:
        self.x_train = x_in
        self.y_train = y_in
        self.w = 0
        self.b = 0
        self.alpha = alpha
        self.num_iters = num_iters

    def compute_cost(self) -> float:
        m = self.x_train.shape[0]

        cost = 0
        for i in range(m):
            f_wb = self.w*self.x_train[i]+self.b
            cost += (f_wb - self.y_train[i])**2

        cost /= 2*m

        return cost

    def compute_gradient(self) -> Tuple[float, float]:

        m = self.x_train.shape[0]

        for i in range(m):
            f_wb = self.w*self.x_train[i]+self.b
            dj_dw = ((f_wb-self.y_train[i])*self.x_train[i])/m
            dj_db = (f_wb-self.y_train[i])/m

        return dj_dw, dj_db

    def gradient_descent(self):

        for i in range(self.num_iters):
            dj_dw, dj_db = self.compute_gradient()
            self.w = self.w - self.alpha*dj_dw
            self.b = self.b - self.alpha*dj_db

    def predict(self):
        m = self.x_train.shape[0]
        predicted = np.zeros(m)
        self.gradient_descent()
        for i in range(m):
            predicted[i] = self.w * self.x_train[i] + self.b
        return predicted


lin_reg = LinearRegressionGD(x_train, y_train)
plt.scatter(x_train, y_train, marker='x')
plt.plot(x_train, lin_reg.predict())
plt.show()
