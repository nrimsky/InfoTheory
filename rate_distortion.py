import numpy as np
from basics import entropy
from math import log2


def r_bernoulli_hamming(p_x: np.array, d: float) -> float:
    """
    :param p_x: Probability distribution of input
    :param d: Max hamming distortion wanted
    :return: Maximum rate of transmission
    """
    return max([entropy(p_x) - entropy(np.array([d, 1 - d])), 0])


def r_gaussian_mean_square(sigma: float, d: float) -> float:
    """
    :param sigma: Standard deviation of gaussian r.v.
    :param d: Max mean square distortion wanted
    :return: Maximum rate of transmission
    """
    return max([0.5 * log2((sigma ** 2) / d), 0])
