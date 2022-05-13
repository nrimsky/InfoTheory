import numpy as np
from math import log2
from basics import log


def info_capacity_gaussian(p: float, n: float) -> float:
    """
    :param p: Average power capacity (average x_i^2 is maximum P)
    :param n: Noise variance / power
    :return: Information capacity of channel
    """
    return 0.5 * log2(1 + p / n)


def info_capacity_gaussian_bandlimited(p: float, n: float) -> float:
    """
    :param p: Average power capacity (average x_i^2 is maximum P)
    :param n: Noise variance / power
    :return: Information capacity of channel
    """
    return 0.5 * log2(1 + p / n)


def optimal_powers_parallel_gaussian(p: float, n: np.array) -> np.array:
    """
    :param p: Power constraint (average channel input power = p)
    :param n: Vector of noise power / variance per gaussian channel
    :return: Optimum allocation of power in each channel
    """
    v = p + n.mean()
    return - 1 * n + v


def info_capacity_parallel_gaussian(p: float, n: np.array) -> float:
    """
    :param p: Power constraint (average channel input power = p)
    :param n: Vector of noise power / variance per gaussian channel
    :return: Information capacity
    """
    v = p + n.mean()
    return 0.5 * log((-1 * n + v)/n + 1).mean()


def info_capacity_parallel_coloured_gaussian(p: float, k_z: np.array) -> float:
    """
    :param p: Power constraint (average channel input power = p)
    :param k_z: E[zz^T] where z is the noise vector
    :return: Information capacity
    """
    w, _ = np.linalg.eig(k_z)
    nu = p + w.mean()
    return 0.5 * log((-1 * w + nu)/w + 1).mean()


def optimal_powers_parallel_coloured_gaussian(k_z: np.array, p: float) -> np.array:
    """
    :param k_z: E[zz^T] where z is the noise vector
    :param p: Power constraint (average channel input power = p)
    :return: k_x = E[xx^T]
    """
    w, v = np.linalg.eig(k_z)
    nu = p + w.mean()
    return v * (np.diag((-1 * w) + nu)) * v.T

