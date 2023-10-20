import numpy as np
import pandas as pd


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

with open("../hyperparameters.txt", "w") as f:
    f.write(f"theta0 = {theta0}\n")
    f.write(f"theta1 = {theta1}\n")
