import numpy as np
from .bayes_filter import BayesFilter


class ParticleFilter(BayesFilter):
    """Particle filter class."""

    def __init__(self, model, system_noise, likelihood, num_particles):
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
        self.system_noise = system_noise  # w()

        self.num_particles = num_particles
        self.likelihood = likelihood

        self.filter_ensemble = None

    def init_state_variable(self, x, P):
        """Initialize state variables."""
        # initialize the ensembles.
        self.filter_ensemble = self._initialize_ensemble(
                                    x, P, self.num_particles)

    def _initialize_ensemble(self, mu, cov, num):
        """Initialze a ensemble."""

        # ensemble.shape : (dimension of x[t], number of particle)
        x_ensemble_init = np.random.multivariate_normal(mu, cov, size=num).T

        return x_ensemble_init

    def filtering(self, t, predict_ensemble, y):
        """Compute the posterior, filter ensemble."""

        # observation_equation : h(t, x[t], v[t])
        # y_hat.shape : (dimension of y[t], number of particles)
        y_hat = self.model.observation_equation(t, predict_ensemble)
        L = self.likelihood.compute(y, y_hat)

        # L.shape : (number of particles, )
        beta = L / np.sum(L)

        # cumulative probability.
        cumulative_proba = np.zeros(self.num_particles)
        cumulative_proba[0] = beta[0]
        for i in range(1, self.num_particles):
            cumulative_proba[i] = cumulative_proba[i-1] + beta[i]

        # deterministic resampling.
        zeta = (np.arange(self.num_particles) + 0.5) / self.num_particles

        filter_ensemble = np.zeros_like(predict_ensemble)
        for i in range(self.num_particles):

            j = np.where(zeta[i] <= cumulative_proba)[0][0]

            # ensemble.shape : (dimension of x[t], number of particles)
            filter_ensemble[:, i] = predict_ensemble[:, j]

        return filter_ensemble

    def predict(self, t, filter_ensemble, u):
        """Compute the prior, prediction ensemble."""

        # w_ensemble.shape : (dimension of w[t], number of particles)
        w_ensemble = self.system_noise(self.num_particles)

        # predict_ensemble.shape : (dimension of x[t], number of particles)
        predict_ensemble = self.model.state_equation(
                                            t, filter_ensemble,
                                            u, w_ensemble)

        return predict_ensemble

    def update_state_variable(self, t, y, u_prev):
        """Estimate the state variable."""

        # compute x[t|t-1]. need u[t-1], previous input.
        predict_ensemble = self.predict(
                                t-1, self.filter_ensemble, u_prev)

        # compute x[t|t]
        self.filter_ensemble = self.filtering(t, predict_ensemble, y)

        return np.mean(self.filter_ensemble, axis=1), np.mean(predict_ensemble, axis=1)
