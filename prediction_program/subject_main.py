import numpy as np

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

theta0 = 0
theta1 = 0

mileage = float(input("Enter the mileage of the car: "))

mileage = np.log(mileage)
estimated_price = estimate_price(mileage, theta0, theta1)
estimated_price = np.exp(estimated_price)

print("The estimated price for a car with {} mileage is: {}".format(mileage, estimated_price))
