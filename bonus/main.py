import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cost_function(w, b, X, y):
    m = len(y)
    total_cost = 0
    for i in range(m):
        total_cost += (y[i] - (w * X[i] + b)) ** 2
    return total_cost / (2 * m)


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def train_model(data_file, learning_rate):
    data = pd.read_csv(data_file)

    data["km"] = np.log(data["km"])
    data["price"] = np.log(data["price"])

    X = data["km"].values
    y = data["price"].values

    m = len(y)

    theta0 = 0
    theta1 = 0

    for _ in range(iterations):
        tmp_theta0 = 0
        tmp_theta1 = 0
        for j in range(m):
            tmp_theta0 += estimate_price(X[j], theta0, theta1) - y[j]
            tmp_theta1 += (estimate_price(X[j], theta0, theta1) - y[j]) * X[j]

        theta0 = theta0 - (learning_rate * tmp_theta0) / m
        theta1 = theta1 - (learning_rate * tmp_theta1) / m
    return theta0, theta1


data_file = "../data.csv"
learning_rate = 0.01
iterations = 300000
theta0, theta1 = train_model(data_file, learning_rate)

print("--------------------------------")
print(f"Theta0: {theta0}")
print(f"Theta1: {theta1}")
print("--------------------------------")

mileage = float(input("Enter the mileage of the car: "))
scaled_mileage = np.log(mileage)
estimated_price = np.exp(estimate_price(scaled_mileage, theta0, theta1))

print(f"The estimated price for a car with {mileage} mileage is: {estimated_price}")

data = pd.read_csv(data_file)
error = cost_function(theta1, theta0, np.log(data["km"]), np.log(data["price"]))

data = pd.read_csv(data_file)
data["km"] = np.log(data["km"])
data["price"] = np.log(data["price"])
plt.scatter(data["km"], data["price"], color="black")
plt.plot(data["km"], theta0 + (theta1 * data["km"]), color="red")
plt.title("Cost function: {}".format(error))
plt.show()
