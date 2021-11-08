import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    model = GDA()
    model.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    util.plot(x_val, y_val, model.theta,
              "../../output/" + save_path[0:-4] + '.png')

    # Use np.savetxt to save outputs from validation set to save_path
    prediction = model.predict(x_val)
    np.savetxt("../../output/" + save_path, prediction)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        y0 = sum([1 if i == 0 else 0 for i in y])
        y1 = sum([1 if i == 1 else 0 for i in y])
        phi = (y1 / len(x))
        mu_0 = (sum([x[i] if y[i] == 0 else 0 for i in range(len(y))])) / y0
        mu_1 = (sum([x[i] if y[i] == 1 else 0 for i in range(len(y))])) / y1
        sigma = []
        for i in range(len(y)):
            if y[i]:
                sigma += [sum([np.matmul(x[i] - mu_1, np.transpose(x[i] - mu_1))])]
            else:
                sigma += [sum([np.matmul(x[i] - mu_0, np.transpose(x[i] - mu_0))])]
        sigma = sum(sigma) / len(x)

        print(f'phi = {phi}')
        print(f'mu_0 = {mu_0}')
        print(f'mu_1 = {mu_1}')
        print(f'simga = {sigma}')

        # Write theta in terms of the parameters
        theta_0 = (0.5 * ((np.matmul(np.transpose(mu_0) * (sigma ** -1), mu_0)) -
                          (np.matmul(np.transpose(mu_1) * (sigma ** -1), mu_1)))) - (np.log((1 - phi) / phi))
        theta = -(sigma ** -1) * (mu_0 - mu_1)
        self.theta = np.concatenate(([theta_0], theta))
        print(f'theta = {self.theta}')
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return [np.dot(self.theta, row) for row in x]
        # *** END CODE HERE


if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
