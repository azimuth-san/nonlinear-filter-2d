import numpy as np
from .bayes_filter import BayesFilter


class ExtendedKalmanFilter(BayesFilter):
    """Extended kalman filter class.

    The EKF estimates the state variable of a plant.
    The state space model of the plant is below.

    x[t+1] = f(t, x[t], u[t], w[t])
    y[t] = h(t, x[t], v[t])

    f: state equation
    h: observation equation
    x: state
    y: output
    w: system noise
    v: observation noise
    """

    def __init__(self, model, cov_w, cov_v):

        self.model = model

        # covariance
        self.cov_w = cov_w
        self.cov_v = cov_v

        # posterior
        self.x_post = None
        self.P_post = None

        # kalman gain
        self.Gain = None

    def init_state_variable(self, x, P):
        """Initialize the state variable."""
        self.x_post = x
        self.P_post = P

    def filtering(self, t, x_prior, P_prior, y):
        """Compute the posterior, x[t|t]."""

        # x, P = x_prior, P_prior
        H = self.model.Jh_x(t, x_prior)
        M = self.model.Jh_v(t, x_prior)
        R = self.cov_v

        # update the kalman gain.
        PH = P_prior @ H.T
        self.Gain = PH @ np.linalg.inv(H @ PH + M @ R @ M.T)

        # update the posterior.
        # observation_equation : h(t, x[t], v[t])
        error = y - self.model.observation_equation(t, x_prior)
        x_post = x_prior + self.Gain @ error
        P_post = (np.eye(self.Gain.shape[0]) - self.Gain @ H) @ P_prior

        return x_post, P_post

    def predict(self, t, x_post, P_post, u):
        """Compute the prior, x[t+1|t]."""

        F = self.model.Jf_x(t, x_post)
        L = self.model.Jf_w(t, x_post)
        Q = self.cov_w

        # update the prior.
        # state_equation : f(t, x[t], u[t], w[t])
        x_prior = self.model.state_equation(t, x_post, u)
        P_prior = F @ P_post @ F.T + L @ Q @ L.T

        return x_prior, P_prior

    def update_state_variable(self, t, y, u_prev):
        """Update the state variable."""

        # compute x[t|t-1]. need u[t-1], previous input.
        x_prior, P_prior = self.predict(
                            t-1, self.x_post, self.P_post, u_prev)

        # compute x[t|t]
        self.x_post, self.P_post = self.filtering(t, x_prior, P_prior, y)

        return self.x_post, x_prior
