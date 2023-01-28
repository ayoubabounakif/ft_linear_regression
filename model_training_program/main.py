import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def gradient_descent(current_w, current_b, points, learning_rate):
    w_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].km
        y = points.iloc[i].price

        w_gradient += -(2 / n) * x * (y - (current_w * x + current_b))
        b_gradient += -(2 / n) * (y - (current_w * x + current_b))

    w = current_w - w_gradient * learning_rate
    b = current_b - b_gradient * learning_rate
    return w, b

def main():
    data = pd.read_csv("../data.csv")

    scaler = MinMaxScaler()
    data[["km", "price"]] = scaler.fit_transform(data[["km", "price"]])

    print(data)

    # θ0, θ1 == w, b
    w = 0
    b = 0
    lr = 0.01
    epochs = 5000

    for _ in range(epochs):
        w, b = gradient_descent(w, b, data, lr)
        print(f"w: {w}, b: {b}")

    plt.scatter(data.km, data.price, color='black')
    plt.plot(data.km, w * data.km + b, color='red')
    plt.show()

if __name__ == "__main__":
    main()