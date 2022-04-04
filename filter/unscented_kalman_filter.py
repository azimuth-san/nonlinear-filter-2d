import numpy as np


class UnscentedendKalmanFilter:
    """Unscented kalman filter class."""

    def __init__(self, model, mu_x_init, cov_x_init,
                 cov_w, cov_v,
                 kappa=0, decompose_method='cholesky'):
        """
        model
        x[t+1] = f(t, x[t], u[t]) + w[t]
        y[t] = h(t, x[t]) + v[t]

        f: state equation
        h: observation equation
        x: state
        y: output
        w: system noise (additive)
        v: observation noise (additive)
        """

        self.model = model

        # covariance
        self.cov_w = cov_w
        self.cov_v = cov_v

        # expectaion to estimate.
        self.x_prior = mu_x_init
        self.x_posterior = None

        # covariance to estimate.
        self.P_prior = cov_x_init
        self.P_posterior = None

        # kalman gain
        self.Gain = None

        # weights of sigma points
        n = mu_x_init.shape[0]
        self.weights = np.zeros(2 * n + 1)
        self.weights[0] = kappa / (n + kappa)
        self.weights[1:] = 1 / (2 * (n + kappa))
        self.kappa = kappa

        self.decompose_method = decompose_method.lower()

    def _compute_sigma_points(self, x_center, P):
        """Compute sigma points."""

        if self.decompose_method == 'cholesky':
            P_sqr = np.linalg.cholesky(P)
        elif self.decompose_method == 'svd':
            U, S, Vh = np.linalg.svd(P)
            P_sqr = U @ np.diag(np.sqrt(S))

        n = P_sqr.shape[0]
        num_points = 2 * n + 1

        x_sigmas = np.zeros((x_center.shape[0], num_points))
        x_sigmas[:, 0] = x_center
        # add the new axis for broadcast.
        # x_center.shape: (d,) -> (d, 1) -> (d, n)
        x_sigmas[:, 1:n+1] = x_center[:, np.newaxis] + np.sqrt(n + self.kappa) * P_sqr
        x_sigmas[:, n+1:] = x_center[:, np.newaxis] - np.sqrt(n + self.kappa) * P_sqr

        return x_sigmas

    def _filtering(self, t, y):
        """Filtering step."""

        x_prior, P_prior = self.x_prior, self.P_prior
        x_sigmas = self._compute_sigma_points(x_prior, P_prior)
        y_sigmas = self.model.observation_equation(t, x_sigmas)
        y_hat = np.sum(self.weights * y_sigmas, axis=1)

        # add the new axis for broadcast.
        # y.shape: (d,) -> (d, 1) -> (d, num_sigma_points)
        y_error = y_sigmas - y_hat[:, np.newaxis]
        Py = (self.weights * y_error) @ y_error.T + self.cov_v

        x_error = x_sigmas - x_prior[:, np.newaxis]
        x_error = x_sigmas - x_prior[:, np.newaxis]
        Pxy = (self.weights * x_error) @ y_error.T

        # update the kalman gain.
        self.Gain = Pxy @ np.linalg.inv(Py)
        K = self.Gain

        # update the posterior.
        self.x_posterior = x_prior + K @ (y - y_hat)
        self.P_posterior = P_prior - K @ Py @ K.T

    def _predict(self, t, u):
        """Prediction step."""

        x_posterior, P_posterior = self.x_posterior, self.P_posterior
        x_sigmas = self._compute_sigma_points(x_posterior, P_posterior)
        x_sigmas_next = self.model.state_equation(t, x_sigmas, u)

        # update the prior.
        self.x_prior = np.sum(self.weights * x_sigmas_next, axis=1)

        x_error = self.x_prior[:, np.newaxis] - x_sigmas_next
        self.P_prior = (self.weights * x_error) @ x_error.T + self.cov_w

    def estimate(self, t, y, u=0):
        """Estimate the state variable."""

        self._filtering(t, y)
        self._predict(t, u)

        return self.x_posterior
