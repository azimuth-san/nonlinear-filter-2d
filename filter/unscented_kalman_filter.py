import numpy as np
from .bayes_filter import BayesFilter


class UnscentedendKalmanFilter(BayesFilter):
    """Unscented kalman filter class.

    The UKF estimates the state variable of a plant.
    The state space model of the plant is below.

    x[t+1] = f(t, x[t], u[t]) + L[t] * w[t]
    y[t] = h(t, x[t]) + M[t] * v[t]

    f: state equation
    h: observation equation
    x: state
    y: output
    w: system noise (additive)
    v: observation noise (additive)
    """

    def __init__(self, model, cov_w, cov_v,
                 kappa=0, decompose_method='cholesky'):

        self.model = model

        # covariance
        self.cov_w = cov_w
        self.cov_v = cov_v

        # posterior
        self.x_post = None
        self.P_post = None

        # kalman gain
        self.Gain = None

        # weights of sigma points
        n = self.model.NDIM['x']
        self.weights = np.zeros(2 * n + 1)
        self.weights[0] = kappa / (n + kappa)
        self.weights[1:] = 1 / (2 * (n + kappa))
        self.kappa = kappa

        self.decompose_method = decompose_method.lower()

    def init_state_variable(self, x, P):
        """Initialize state variables."""
        self.x_post = x
        self.P_post = P

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

    def filtering(self, t, x_prior, P_prior, y):
        """Compute the posterior, x[t|t]."""

        x_sigmas = self._compute_sigma_points(x_prior, P_prior)
        y_sigmas = self.model.observation_equation(t, x_sigmas)
        y_hat = np.sum(self.weights * y_sigmas, axis=1)

        M = self.model.Mt(t)
        # add the new axis for broadcast.
        # y.shape: (d,) -> (d, 1) -> (d, num_sigma_points)
        y_error = y_sigmas - y_hat[:, np.newaxis]
        Py = (self.weights * y_error) @ y_error.T + M @ self.cov_v @ M.T

        x_error = x_sigmas - x_prior[:, np.newaxis]
        x_error = x_sigmas - x_prior[:, np.newaxis]
        Pxy = (self.weights * x_error) @ y_error.T

        # update the kalman gain.
        self.Gain = Pxy @ np.linalg.inv(Py)

        # update the posterior.
        x_post = x_prior + self.Gain @ (y - y_hat)
        P_post = P_prior - self.Gain @ Py @ self.Gain.T

        return x_post, P_post

    def predict(self, t, x_post, P_post, u):
        """Compute the prior, x[t+1|t]."""

        x_sigmas = self._compute_sigma_points(x_post, P_post)
        x_sigmas_next = self.model.state_equation(t, x_sigmas, u)

        # update the prior.
        x_prior = np.sum(self.weights * x_sigmas_next, axis=1)

        L = self.model.Lt(t)
        x_error = x_prior[:, np.newaxis] - x_sigmas_next
        P_prior = (self.weights * x_error) @ x_error.T + L @ self.cov_w @ L.T

        return x_prior, P_prior

    def update_state_variable(self, t, y, u_prev):
        """Update the state variable."""

        # compute x[t|t-1]. need u[t-1], previous input.
        x_prior, P_prior = self.predict(
                            t-1, self.x_post, self.P_post, u_prev)

        # compute x[t|t]
        self.x_post, self.P_post = self.filtering(t, x_prior, P_prior, y)

        return self.x_post, x_prior
