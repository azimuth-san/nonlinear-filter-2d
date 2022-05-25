import numpy as np


class ExtendedKalmanFilter:
    """Extended kalman filter class."""

    def __init__(self, model, mu_x_init, cov_x_init,
                 cov_w, cov_v):
        """
        model
        x[t+1] = f(t, x[t], u[t], w[t])
        y[t] = h(t, x[t], v[t])

        f: state equation
        h: observation equation
        x: state
        y: output
        w: system noise
        v: observation noise
        """

        self.model = model

        # covariance
        self.cov_w = cov_w
        self.cov_v = cov_v

        # expectaion to estimate.
        self.x_prior = None
        self.x_posterior = mu_x_init

        # covariance to estimate.
        self.P_prior = None
        self.P_posterior = cov_x_init

        # kalman gain
        self.Gain = None

    def _filtering(self, t, y):
        """Filtering step."""

        x, P = self.x_prior, self.P_prior
        H = self.model.Jh_x(x)
        M = self.model.Jh_v(x)
        R = self.cov_v

        # update the kalman gain.
        PH = P @ H.T
        self.Gain = PH @ np.linalg.inv(H @ PH + M @ R @ M.T)

        K = self.Gain
        # update the posterior.
        # observation_equation : h(t, x[t], v[t])
        error = y - self.model.observation_equation(t, x)
        self.x_posterior = x + K @ error
        self.P_posterior = (np.eye(K.shape[0]) - K @ H) @ P

    def _predict(self, t, u):
        """Prediction step."""

        x, P = self.x_posterior, self.P_posterior
        F = self.model.Jf_x(x)
        L = self.model.Jf_w(x)
        Q = self.cov_w

        # update the prior.
        # state_equation : f(t, x[t], u[t], w[t])
        self.x_prior = self.model.state_equation(t, x, u)
        self.P_prior = F @ P @ F.T + L @ Q @ L.T

    def estimate(self, t, y, u_prev=0):
        """Estimate the state variable."""

        # predict the x(t|t-1), need a previous input u(t-1)
        self._predict(t-1, u_prev)

        # estimate the x(t|t)
        self._filtering(t, y)

        return self.x_posterior
