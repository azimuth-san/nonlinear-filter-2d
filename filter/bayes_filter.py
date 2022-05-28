from abc import abstractmethod, ABC


class BayesFilter(ABC):
    """Bayes filter class."""

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
        self.x_prior = None  # x[t|t-1]
        self.x_posterior = None  # x[t|t]

    @abstractmethod
    def filtering(self, t, y):
        """Compute the posterior, x[t|t]."""
        pass

    @abstractmethod
    def predict(self, t, u):
        """Compute the prior, x[t+1|t]."""
        pass

    def estimate(self, t, y, u_prev=0):
        """Estimate the state variable."""

        # compute x[t|t-1]. need u[t-1], previous input.
        self.predict(t-1, u_prev)

        # compute x[t|t]
        self.filtering(t, y)

        return self.x_posterior
