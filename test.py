import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

X = 6 * np.random.rand(100, 3)
y = 4 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, X_test=0.2, random_state=42)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(x_train_scaled, y_train)
ridge_predictions = ridge.predict(x_test_scaled)

lasso = Lasso(alpha=1.0)
lasso.fit(x_train_scaled, y_train)
lasso_predictions = lasso.predict(x_test_scaled)