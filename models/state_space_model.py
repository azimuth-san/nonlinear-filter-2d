from abc import abstractclassmethod, ABC


class StateSpaceModel(ABC):

    # dimensions
    NDIM = {'x': 1,  # state
            'y': 1,  # output
            'u': 0,  # control input
            'w': 1,  # system noise
            'v': 1   # observation noise
            }

    @abstractclassmethod
    def state_equation(self, t, x, u=0, w=0):
        """Calculate the state equation.

        x[t+1] = f(t, x[t], u[t], w[t])
        f: state equation.
        x: state
        u: control input
        w: system noise
        t: time
        """
        pass

    @abstractclassmethod
    def observation_equation(self, t, x, v=0):
        """Calculate the observation equation.

        y[t] = h(t, x[t], v[t])
        h: observation equation
        y: output
        v: observation noise
        t: time
        """
        pass

    def Jf_x(self, t, x):
        """The Jacobian of the system model.

        (df/dx)(x), x[t+1] = f(x[t], u[t], w[t], t).

        This method is needed in the extended kalman filter.
        """
        raise NotImplementedError

    def Jf_w(self, t, x):
        """The Jacobian of the state equation.

        (df/dw)(x), x[t+1] = f(x[t], u[t], w[t], t).

        This method is needed in the extended kalman filter.
        """
        raise NotImplementedError

    def Jh_x(self, t, x):
        """The Jacobian of the observation equation.

        (dh/dx)(x), y[t] = h(x[t], v[t], t)

        This method is needed in the extended kalman filter.
        """
        raise NotImplementedError

    def Jh_v(self, t, x):
        """The Jacobian of the observation equation.

        (dh/dv)(x), y[t] = h(x[t], v[t], t)

        This method is needed in the extended kalman filter.
        """
        raise NotImplementedError

    def Lt(self, t):
        """x[t+1] = f(x[t], u[t], t) + L[t] * w[t]

        In case of system noise is additive, return L[t].
        """
        raise NotImplementedError

    def Mt(self, t):
        """y[t] = h(x[t], t) + M[t] * v[t]

        In case of observatoin noise is additive, return M[t].
        """
        raise NotImplementedError
