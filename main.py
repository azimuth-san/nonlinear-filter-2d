import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from models import ConstantVelocity2d, GaussianNoise
from filter import ParticleFilter
from filter import ExtendedKalmanFilter
from filter import UnscentedendKalmanFilter
from filter import LikelihoodGaussian


def parse_arguments():
    """Parse optional arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str, default='ukf',
                        choices=['pf', 'ekf', 'ukf'],
                        help='type of nonlinear filters')
    parser.add_argument('--num_particles', type=int, default=300,
                        help='number of particles')
    parser.add_argument('--kappa', type=float, default=0,
                        help='parametr to compute weiths of sigma points')
    parser.add_argument('--decompose', default='cholesky',
                        choices=['cholesky', 'svd'],
                        help='method to decompose covariance matrix')

    parser.add_argument('--dt', type=float, default=0.1,
                        help='sampling period[sec]')
    parser.add_argument('--steps', type=int, default=150,
                        help='the number of simulation steps')
    parser.add_argument('--initial_point', type=float, nargs=4,
                        default=[-100, 25, 50, -25],
                        help='[m, m/s, m, m/s]')
    parser.add_argument('--initial_covariance', type=float, nargs=4,
                        default=[100, 800, 100, 800])

    parser.add_argument('--cov_w', type=float, nargs=2,
                        default=[300, 300],
                        help='covariance of process noise')
    parser.add_argument('--cov_v', type=float, nargs=2,
                        default=[25, 0.001],
                        help='covariance of observation noise')

    parser.add_argument('--seed', type=int, default=-1,
                        help='seed of random variables.')
    return parser.parse_args()


def create_nonlinear_filter(args, model, system_noise):

    if args.filter == 'pf':
        estimator = ParticleFilter(
                        model, system_noise,
                        LikelihoodGaussian(np.diag(args.cov_v)),
                        args.num_particles)

    elif args.filter == 'ekf':
        estimator = ExtendedKalmanFilter(model,
                                         np.diag(args.cov_w),
                                         np.diag(args.cov_v),
                                         )
    elif args.filter == 'ukf':
        estimator = UnscentedendKalmanFilter(
                        model,
                        np.diag(args.cov_w),
                        np.diag(args.cov_v),
                        args.kappa,
                        args.decompose,
                        )
    return estimator


def generate_time_series_data(num_steps, model, initial_point,
                              system_noise, observation_noise):

    x = np.zeros((model.NDIM['x'], num_steps))
    x[:, 0] = initial_point
    z = np.zeros((model.NDIM['y'], num_steps))

    # generate the target dataset．
    for t in range(num_steps):

        w = system_noise(1)
        v = observation_noise(1)

        if t > 0:
            x[:, t] = model.state_equation(t-1, x[:, t-1], 0, w)
        z[:, t] = model.observation_equation(t, x[:, t], v)

    return x, z


def estimate_all_samples(estimator, x, z, initial_state, initial_cov):

    x_est = np.zeros_like(x)
    begin = time.time()
    for t in range(z.shape[1]):

        if t == 0:
            estimator.init_state_variable(initial_state, initial_cov)
            x_est[:, t] = initial_state
        else:
            # estimate the state variable from a observation variable.
            x_est[:, t], _ = estimator.estimate(t, z[:, t])

        # check error
        error = x_est[:, t] - x[:, t]
        distance = np.sqrt(error[0]**2 + error[2]**2)
        if distance > 100:
            print(t, 'the estimated value is far from the true value.')

    time_est = (10**3) * (time.time() - begin) / z.shape[1]
    print(f'\nmean estimation time = {time_est:.3f} [msec].')
    return x_est


def main():

    args = parse_arguments()
    if args.seed >= 0:
        np.random.seed(args.seed)

    model = ConstantVelocity2d(args.dt)
    system_noise = GaussianNoise(np.zeros(model.NDIM['w']),
                                 np.diag(args.cov_w))
    observation_noise = GaussianNoise(np.zeros(model.NDIM['v']),
                                      np.diag(args.cov_v))
    initial_point = np.array(args.initial_point)

    num_steps = args.steps
    x, z = generate_time_series_data(num_steps, model, initial_point,
                                     system_noise, observation_noise)

    estimator = create_nonlinear_filter(args, model, system_noise)

    # set the initial state from the initial measurement.
    initial_state = np.array([z[0, 0]*np.cos(z[1, 0]), 0,
                              z[0, 0]*np.sin(z[1, 0]), 0])
    initial_cov = np.diag(args.initial_covariance)

    x_est = estimate_all_samples(estimator, x, z, initial_state, initial_cov)

    # evaluation
    error = np.zeros((2, num_steps))
    error[0] = x[0] - x_est[0]
    error[1] = x[2] - x_est[2]
    mean_error = np.mean(np.sqrt(np.sum(error ** 2, axis=0)))
    print(f'mean position error = {mean_error:.3f} [m].\n')

    # plot
    plt.figure(1, (8, 6))
    plt.plot(x[0], x[2], 'r')
    plt.scatter(x_est[0], x_est[2])
    plt.title(args.filter)
    plt.xlabel('x_t')
    plt.ylabel('y_t')
    plt.grid(True)
    plt.legend(['truth', 'estimation'])

    # save
    save_dir = 'result'
    os.makedirs(save_dir, exist_ok=True)
    if args.seed >= 0:
        fname = f'tracking_result_{args.filter}_seed{args.seed}.png'
    else:
        fname = f'tracking_result_{args.filter}.png'
    plt.savefig(os.path.join(save_dir, fname))
    # plt.show()


if __name__ == '__main__':
    main()
