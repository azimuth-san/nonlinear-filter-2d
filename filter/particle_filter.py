import numpy as np


class ParticleFilter:
    """Particle filter class."""

    def __init__(self, model, mu_x_init, cov_x_init,
                 system_noise, likelihood, num_particles):
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

        # initialize the ensembles.
        self.predict_ensemble = self._initialize_predict_ensemble(
                                        mu_x_init, cov_x_init, num_particles)
        self.filter_ensemble = None

    def _initialize_predict_ensemble(self, mu, cov, num):
        """Initialze a predict ensemble."""

        # ensemble.shape : (dimension of x[t], number of particle)
        x_ensemble_init = np.random.multivariate_normal(mu, cov, size=num).T

        return x_ensemble_init

    def _filtering(self, t, y):
        """Calculate the filter ensemble."""

        # observation_equation : h(t, x[t], v[t])
        # y_hat.shape : (dimension of y[t], number of particles)
        y_hat = self.model.observation_equation(
                                t, self.predict_ensemble)
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

        self.filter_ensemble = np.zeros_like(self.predict_ensemble)
        for i in range(self.num_particles):

            j = np.where(zeta[i] <= cumulative_proba)[0][0]

            # ensemble.shape : (dimension of x[t], number of particles)
            self.filter_ensemble[:, i] = self.predict_ensemble[:, j]

    def _predict(self, t, u):
        """Calculate the prediction ensemble."""

        # w_ensemble.shape : (dimension of w[t], number of particles)
        w_ensemble = self.system_noise(self.num_particles)

        # predict_ensemble.shape : (dimension of x[t], number of particles)
        self.predict_ensemble = self.model.state_equation(
                                        t, self.filter_ensemble,
                                        u, w_ensemble)

    def estimate(self, t, y, u=0):
        """Estimate the state variable."""

        self._filtering(t, y)
        self._predict(t, u)

        return np.mean(self.filter_ensemble, axis=1)
