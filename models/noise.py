import numpy as np


class GaussianNoise:
    """Gaussian noise."""

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def __call__(self, num):
        # v.shape = (mean.shape[0], num)
        v = np.random.multivariate_normal(self.mean, self.cov, size=num).T
        if num == 1:
            v = v.reshape(v.shape[0])

        return v
