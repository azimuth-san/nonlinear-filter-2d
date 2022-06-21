from abc import abstractmethod, ABC


class BayesFilter(ABC):
    """Bayes filter class."""

    @abstractmethod
    def init_state_variable(self, x, P):
        """Initialize state variables."""

    @abstractmethod
    def update_state_variable(self, t, y, u_prev=0):
        """Update the state variables."""

    def estimate(self, t, y, u_prev=0):
        """Estimate the state variables."""

        # update the posterior and prior
        x_post, x_pred = self.update_state_variable(t, y, u_prev)

        return x_post, x_pred
