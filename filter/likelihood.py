import numpy as np


class LikelihoodGaussian:
    """Gaussian likelihood class."""
    def __init__(self, cov):

        self.det_covariance = np.linalg.det(cov)
        self.inv_covariance = np.linalg.inv(cov)

    def compute(self, y, y_hat):
        """Compute the likelihood."""

        d = y.shape[0]
        c = 1 / np.sqrt(((2 * np.pi)**d) * self.det_covariance)

        # y_hat.shape : (d, number of samples)
        # y.shape : (d, ) -> (d, 1) -> (d, number of samples)
        error = y[:, np.newaxis] - y_hat

        # error.T * inv(covariance) * error
        cov_error = np.dot(self.inv_covariance, error)
        expo = np.exp(-0.5 * np.sum(error * cov_error, axis=0))

        # likelihood
        L = c * expo
        if np.sum(L == 0) == L.shape[0]:
            L += 1e-9

        return L
