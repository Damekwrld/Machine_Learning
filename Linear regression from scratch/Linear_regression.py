import numpy as np

# To save our values
np.random.seed(42)

# Generate 100 random numbers between 1 and 10
X =  10 * np.random.rand(100, 1)

# Generate target values with some noise
y = 4 * X + 3 + np.random.randn(100, 1)

# print first five values from x
# print(X[:5])

# print first five values from y
# print(y[:5])

class LinearRegression:
    def __init__(self, epochs=1000, learning_rate=0.01):
        self.w = 0.0
        self.b = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        n = len(X)

        # TRain the model
        for epoch in range(self.epochs):
           y_pred = self.w * X + self.b

           dw = (2 / n) * np.sum((y_pred - y) * X)
           db = (2 / n) * np.sum(y_pred - y)

           self.w -= self.learning_rate * dw
           self.b -= self.learning_rate * db

    def predict(self, X):
        return self.w * X + self.b

    def mse(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)

# model = LinearRegression()
# model.fit(X, y)

# print("Weight", model.w)
# print("bias", model.b)
# print("mse", model.mse(X, y))

# Normal equation

# add bias column
ones = np.ones((len(X), 1))
x_b = np.concatenate((ones, X), axis=1)

# TRanspose of x
x_transposed = x_b.T

# Multiply x transpose by x
xtx = x_transposed @ x_b

# calculate inverse
xtx_inverse = np.linalg.inv(xtx)

# multiply x transpose by y
xty = x_transposed @ y

# calculate parameters
theta = xtx_inverse @ xty
theta2 = np.linalg.pinv(x_b) @ y
print(theta)