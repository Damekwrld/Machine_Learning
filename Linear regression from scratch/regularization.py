import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

X = 6 * np.random.rand(100, 3)
y = 4 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_predictions = ridge.predict(X_test_scaled)

lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)
lasso_predictions = lasso.predict(X_test_scaled)

# print("Ridge MSE:", mean_squared_error(y_test, ridge_predictions))
# print("Ridge R **2:", r2_score(y_test, ridge_predictions))

# print("Lasso MSE:", mean_squared_error(y_test, lasso_predictions))
# print("Lasso R **2:", r2_score(y_test, lasso_predictions))

# print("Ridge weights:", ridge.coef_)
# print("Lasso weights:", lasso.coef_)

lasso_1 = Lasso(alpha=0.01)
lasso_1.fit(X_train_scaled, y_train)
lasso_2 = Lasso(alpha=0.1)
lasso_2.fit(X_train_scaled, y_train)
lasso_3 = Lasso(alpha=10)
lasso_3.fit(X_train_scaled, y_train)

print(lasso_1.coef_)
print(lasso_2.coef_)
print(lasso_3.coef_)