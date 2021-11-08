import numpy as np
import util
import matplotlib.pyplot as plt


def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    model = PoissonRegression()
    model.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    p_eval = model.predict(x_eval)
    np.savetxt("../../output/" + save_path, p_eval)
    # *** END CODE HERE ***
    plt.figure()
    # here `y_eval` is the true label for the validation set and `p_eval` is the predicted label.
    plt.scatter(y_eval, p_eval, alpha=0.4, c='red',
                label='Ground Truth vs Predicted')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.legend()
    plt.savefig('../../output/poisson_valid.png')


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = [0 for i in range(len(x[0]))]
        # For each iteration
        for i in range(self.max_iter):
            # For each feature
            response = []
            for j in range(len(self.theta)):
                sum = 0
                # For each example
                for k in range(len(x)):
                    sum += np.dot(np.exp(np.dot(np.transpose(self.theta), x[k])) - y[k], x[k][j])
                change = self.step_size * sum
                self.theta[j] -= change
                response += [change]
            if np.linalg.norm(response) < self.eps:
                break
        print(self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        return [np.dot(self.theta, row) for row in x]
        # *** END CODE HERE ***


if __name__ == '__main__':
    main(lr=1e-5,
         train_path='train.csv',
         eval_path='valid.csv',
         save_path='poisson_pred.txt')
