import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    n = len(y)
    for _ in range(num_iterations):
        y_hat = w * X + b
        w_gradient = -(2/n) * sum(X * (y - y_hat))
        b_gradient = -(2/n) * sum(y - y_hat)
        w -= w_gradient * learning_rate
        b -= b_gradient * learning_rate
        print(f"m: {w}, b: {b}")
    return w, b

def main():
    data = pd.read_csv("../data.csv")

    # Scale the data
    scaler = MinMaxScaler()
    data[["km", "price"]] = scaler.fit_transform(data[["km", "price"]])

    print(data)

    X = data["km"].values
    y = data["price"].values

    w = 0
    b = 0
    learning_rate = 0.01
    epochs = 10000

    w, b = gradient_descent(X, y, w, b, learning_rate, epochs)


    plt.scatter(X, y, color='black')
    plt.plot(X, w * X + b, color='red')
    plt.show()


# STEPS:
## TRaining:
# * Initialize weight as zero
# * Initialize bias as zero

## Given a data point:
# * Predit result by using y-hat = wx + b
# * Calculate error
# * Use gradient descent to figure out new weight and bias values
# * Repeat n times (n = number of iterations)

# class MyLinearRegression:

#     def __init__(self, learning_rate=0.001, n_iters=1000):
#         self.lr = learning_rate
#         self.n_iters = n_iters
#         self.weights = None
#         self.bias = None

#     # used for training
#     def fit(self, X, y):
#         number_of_samples = X.shape[0]
#         self.weights = np.zeros(X.shape[0])
#         self.bias = 0

#         # for _ in range(50):
            
#         y_hat = np.dot(X.T, self.weights) + self.bias
#         print(y_hat)

#         dw = (1 / number_of_samples) * np.dot(X.T, (y_hat - y))
#         db = (1 / number_of_samples) * np.sum(y_hat - y)

#         print(dw, db)

#         self.weights -= self.lr * dw
#         self.bias -= self.lr * db

#     # used for inference
#     def predict(self, X):
#         y_hat = np.dot(X, self.weights) + self.bias
#         return y_hat


# class MyLinearRegression:
#     def __init__(self, learning_rate=0.001, epochs=30):
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.m = None
#         self.b = None

#     def fit(self, data):
#         self.m = 0
#         self.b = 0
#         for _ in range(self.epochs):
#             self.m, self.b = gradient_descent(self.m, self.b, data, self.learning_rate)
#         print(self.m, self.b)
        
# def mse(m, b, points):
#     total_error = 0
#     for i in range(len(points)):
#         x = points.iloc[i].km
#         y = points.iloc[i].price
#         total_error += (y - (m * x + b)) ** 2
#     return total_error / float((len(points)))

# def gradient_descent(current_m, current_b, points, learning_rate):
#     w_gradient = 0
#     b_gradient = 0

#     n = len(points)

#     for i in range(n):
#         x = points.iloc[i].km
#         y = points.iloc[i].price

#         w_gradient = -(2 / n) * sum(x * y - (current_m * x + current_b))
#         b_gradient = -(2 / n) * sum(y - (current_m * x + current_b))

#     m = current_m - w_gradient * learning_rate
#     b = current_b - b_gradient * learning_rate
#     return m, b
    

# def main():
#     data = pd.read_csv("../data.csv")

#     for i in range(len(data)):
#         data.iloc[i].km = data.iloc[i].km / 100
#         data.iloc[i].price = data.iloc[i].price / 100

#     print(data)

#     m = 0
#     b = 0
#     L = 0.0001
#     epochs = 300

#     for i in range(epochs):
#         m, b = gradient_descent(m, b, data, L)
#         print(f"m: {m}, b: {b}")

#     plt.scatter(data.km, data.price, color='black')
#     plt.plot(list(range(10, 3000)), [m * x + b for x in range(10, 3000)], color='red')
#     plt.show()
    




    # # read X, y
    # X = data[1:, 0]
    # y = data[1:, 1]

    # data = np.genfromtxt("data.csv", delimiter=",")
    # X = np.array([[mileage] for mileage in data[1:, 0]])
    # y = np.array([price for price in data[1:, 1]])

    # my_model.fit(X, y)
    # predictions = my_model.predict(X)

    # print(f'predictions: {predictions}')

    # mse = my_model.mse(predictions, y)
    # print(mse)

    # plt.figure()
    # plt.scatter(X, y, color="r", marker="x")
    # plt.xlabel("Km")
    # plt.ylabel("Price")
    # plt.show()




if __name__ == "__main__":
    main()