import numpy as np
from .state_space_model import StateSpaceModel


class ConstantVelocity2d(StateSpaceModel):
    """Moving object with constant velocity."""

    # dimensions
    NDIM = {'x': 4,  # state
            'y': 2,  # output
            'u': 0,  # control input
            'w': 2,  # system noise
            'v': 2   # observation noise
            }

    def __init__(self, dt=0.1):

        self.dt = dt  # sampling time

    def state_equation(self, t, x, u=0, w=np.zeros(2)):
        """Calculate the state equation.

        x[t+1] = f(t, x[t], u[t], w[t])
        f: state equation.
        x: state
        u: control input
        w: system noise
        t: time
        """

        dt = self.dt
        x_next = np.zeros_like(x)
        x_next[0] = x[0] + dt * x[1] + 0.5 * (dt**2) * w[0]
        x_next[1] = x[1] + dt * w[0]
        x_next[2] = x[2] + dt * x[3] + 0.5 * (dt**2) * w[1]
        x_next[3] = x[3] + dt * w[1]

        return x_next

    def observation_equation(self, t, x, v=np.zeros(2)):
        """Calculate the observation equation.

        y[t] = h(t, x[t], v[t])
        h: observation equation
        y: output
        v: observation noise
        t: time
        """

        if x.ndim == 1:
            y = np.zeros(2)
        else:
            y = np.zeros((2, x.shape[1]))

        d = np.sqrt(x[0] ** 2 + x[2] ** 2)
        y[0] = d + v[0]

        eps = 1e-9
        # y[1] = np.arctan(x[2] / (x[0] + eps)) + v[1]
        y[1] = np.arctan2(x[2], x[0] + eps) + v[1]

        return y

    def Jf_x(self, t, x):
        """The Jacobian of the system model.

        (df/dx)(x), x[t+1] = f(x[t], u[t], w[t], t).
        """
        F = np.array([[1, self.dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, self.dt],
                     [0, 0, 0, 1]])
        return F

    def Jf_w(self, t, x):
        """The Jacobian of the state equation.

        (df/dw)(x), x[t+1] = f(x[t], u[t], w[t], t).
        """
        L = np.array([[0.5 * (self.dt**2), 0],
                     [self.dt, 0],
                     [0, 0.5 * (self.dt**2)],
                     [0, self.dt]])
        return L

    def Jh_x(self, t, x):
        """The Jacobian of the observation equation.

        (dh/dx)(x), y[t] = h(x[t], v[t], t)
        """
        d_square = np.sum(x[0]**2 + x[2]**2)
        H = np.array([[x[0]/np.sqrt(d_square), 0, x[2]/np.sqrt(d_square), 0],
                     [-x[2]/d_square, 0, x[0]/d_square, 0]])
        return H

    def Jh_v(self, t, x):
        """The Jacobian of the observation equation.

        (dh/dv)(x), y[t] = h(x[t], v[t], t)
        """
        return np.eye(2)
