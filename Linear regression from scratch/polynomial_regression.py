import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(42)

X = 6 * np.random.rand(100, 1) - 3

y = 0.5 * X ** 2 + X + 2 + np.random.randn(100, 1)

# TRansform the features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train the data
model = LinearRegression()
model.fit(X_poly, y)

# Make the prediction
predictions = model.predict(X_poly)

sorted_indices = np.argsort(X[:, 0])

plt.plot(
    X[sorted_indices],
    predictions[sorted_indices],
    color="green",
)

plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial regression")
plt.show()
