import numpy as np


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


theta0 = 0
theta1 = 0

with open("../hyperparameters.txt", "r") as f:
    theta0, theta1 = [float(line.split("=")[1].strip()) for line in f.readlines()]


mileage = float(input("Enter the mileage of the car: "))
scaled_mileage = np.log(mileage)
estimated_price = np.exp(estimate_price(scaled_mileage, theta0, theta1))

print(f"The estimated price for a car with {mileage} mileage is: {estimated_price}")
