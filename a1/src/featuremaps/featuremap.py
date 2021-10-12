import math
from operator import itemgetter
import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

factor = 2.0
outputPath = "../../output/"


class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.matmul(np.linalg.pinv(
            np.matmul(X.transpose(), X)), np.matmul(X.transpose(), y))
        # *** END CODE HERE ***

    def fit_GD(self, X, y, alpha, iterations):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.array([0.0000000001 for i in range(len(X[0]))])
        # For each iteration
        for i in range(iterations):
            # For each feature
            for j in range(len(self.theta)):
                sum = 0
                # For each example
                for k in range(len(X)):
                    sum += (self.predict(X[k]) - y[k]) * X[k][j]
                self.theta[j] -= alpha * sum
        # *** END CODE HERE ***

    def fit_SGD(self, X, y, alpha, iterations):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the stochastic gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.array([0.0000000001 for i in range(len(X[0]))])
        # For each iteration
        for i in range(iterations):
            # For each example
            for k in range(len(X)):
                # For each feature
                for j in range(len(self.theta)):
                    self.theta[j] -= alpha * \
                        (self.predict(X[k]) - y[k]) * X[k][j]
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        arr = []
        # For row in X
        for x in X:
            # For feature in x
            for i in range(1, len(x)):
                # For each power 0-k
                arr += [[x[i] ** j for j in range(k + 1)]]
        return np.array(arr)
        # *** END CODE HERE ***

    def create_cosine(self, k, X):
        """
        Generates a cosine with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        arr = []
        # For row in X
        for x in X:
            # For feature in x
            for i in range(1, len(x)):
                # For each power 0-k
                arr += [[x[i] ** j for j in range(k + 1)] + [math.cos(x[i])]]
        return np.array(arr)
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.matmul(X, self.theta)
        # *** END CODE HERE ***


def run_exp(train_path, cosine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.pdf', f_type='normal', alpha=0.01, iterations=100):

    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-0.1, 1.1, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)
    plt.title(filename[0:-4])
    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        model = LinearModel()
        if cosine:
            poly = model.create_cosine(k, train_x)
            predictX = model.create_cosine(k, plot_x)
        else:
            poly = model.create_poly(k, train_x)
            predictX = model.create_poly(k, plot_x)

        if f_type == "normal":
            model.fit(poly, train_y)
        if f_type == "gd":
            model.fit_GD(poly, train_y, alpha, iterations)
        if f_type == "sgd":
            model.fit_SGD(poly, train_y, alpha, iterations)

        plot_y = model.predict(predictX)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2.5, 2.5)
        plt.plot(plot_x[:, 1], plot_y,
                 label='k={:d}, fit={:s}'.format(k, f_type))

    plt.legend()
    plt.tight_layout()
    plt.savefig(outputPath + filename)
    plt.clf()
    return (plot_x[:, 1], plot_y)


def main(medium_path, small_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # Section 1 Part 2
    x1, y1 = run_exp(medium_path, False, [3], "1-2.png", "normal")
    # Section 1 Part 3
    train_x, train_y = util.load_dataset(medium_path, add_intercept=True)
    run_exp(medium_path, False, [3], "1-3-gd-100.png", "gd", 0.01, 100)
    run_exp(medium_path, False, [3], "1-3-gd-1000.png", "gd", 0.01, 1000)
    x2, y2 = run_exp(medium_path, False, [3], "1-3-gd-10000.png", "gd", 0.01, 10000)
    run_exp(medium_path, False, [3], "1-3-sgd-100.png", "sgd", 0.01, 100)
    run_exp(medium_path, False, [3], "1-3-sgd-1000.png", "sgd", 0.01, 1000)
    x3, y3 = run_exp(medium_path, False, [3], "1-3-sgd-10000.png", "sgd", 0.01, 10000)
    plt.figure()
    plt.title("1-3")
    plt.scatter(train_x[:, 1], train_y)
    plt.plot(x1, y1, label="normal")
    plt.plot(x2, y2, label="gd")
    plt.plot(x3, y3, label="sgd")
    plt.legend()
    plt.savefig(outputPath + "1-3.png")
    plt.clf()
    # Section 1 Part 4
    run_exp(medium_path, False,  [3, 5, 10, 20], "1-4.png", "normal")
    # Section 1 Part 5
    run_exp(medium_path, True,  [3, 5, 10, 20], "1-5.png", "normal")
    # Section 1 Part 6
    run_exp(small_path, False,  [1, 3, 5, 10, 20], "1-6.png", "normal")
    run_exp(small_path, True,  [1, 3, 5, 10, 20], "1-6-cosine.png", "normal")
    run_exp(small_path, False,  [1, 3, 5, 10, 20], "1-6-gd.png", "gd", iterations=10000)
    run_exp(small_path, True,  [1, 3, 5, 10, 20], "1-6-gd-cosine.png", "gd", iterations=10000)
    run_exp(small_path, False,  [1, 3, 5, 10, 20], "1-6-sgd.png", "sgd", iterations=10000)
    run_exp(small_path, True,  [1, 3, 5, 10, 20], "1-6-sgd-cosine.png", "sgd", iterations=10000)
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(medium_path='medium.csv',
         small_path='small.csv')
