import numpy as np

x = np.array([
    [100, 2, 5],
    [150, 3, 3],
    [200, 4, 2],
    [120, 2, 10],
])

y = np.array([
    [20],
    [30],
    [42],
    [10]
])

print(x.shape)
print(y.shape)

# TRain the model with normal equation
ones = np.ones((len(x), 1))
x_b = np.concatenate((ones, x) ,axis=1)
theta = np.linalg.pinv(x_b) @ y

print("Learned parameters:")
print(theta) 

new_house = np.array([1, 180, 3, 4])
prediction = new_house @ theta
print("Predicted price:", prediction[0], "million")
