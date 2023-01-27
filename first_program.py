import numpy as np
import matplotlib.pyplot as plt

# STEPS:
## TRaining:
# * Initialize weight as zero
# * Initialize bias as zero

## Given a data point:
# * Predit result by using y-hat = wx + b
# * Calculate error
# * Use gradient descent to figure out new weight and bias values
# * Repeat n times (n = number of iterations)

class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # used for training
    def fit(self, X, y):
        number_of_samples = X.shape[0]
        self.weights = np.zeros(X.shape[0])
        self.bias = 0

        # for _ in range(50):
            
        y_hat = np.dot(X, self.weights) + self.bias
        print(y_hat)

        dw = (1 / number_of_samples) * np.dot(X, (y_hat - y))
        db = (1 / number_of_samples) * np.sum(y_hat - y)

        print(dw, db)

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    # used for inference
    def predict(self, X):
        y_hat = np.dot(X, self.weights) + self.bias
        return y_hat

    def mse(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)



def main():
    my_model = LinearRegression()
    # read_data = np.genfromtxt("data.csv", delimiter=",")

    # # read X, y
    # X = read_data[:, 0]
    # y = read_data[:, 1]

    data = np.genfromtxt("data.csv", delimiter=",")
    X = data[1:, 0]
    y = data[1:, 1]

    print(X)


    my_model.fit(X, y)
    # predictions = my_model.predict(X)

    # print(f'predictions: {predictions}')

    # mse = my_model.mse(predictions, y)
    # print(mse)

    # plt.figure()
    # plt.scatter(X[: ,0], y, color="b", marker="o", s=30)
    # plt.xlabel("Km")
    # plt.ylabel("Price")
    # plt.show()




if __name__ == "__main__":
    main()