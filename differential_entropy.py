from math import pi, log2, e
import numpy as np


def h_gaussian(sigma: float) -> float:
    """
    :param sigma: Standard deviation of Gaussian r.v. with distribution N(mu, sigma^2)
    :return: differential entropy
    """
    return 0.5 * log2(2 * pi * e * (sigma ** 2))


def h_uniform(a: float, b: float) -> float:
    """
    :param a: Lower bound of variable
    :param b: Upper bound of variable (x ~ U(a,b))
    :return: differential entropy
    """
    return log2(b - a)


def h_multivariate_gaussian(covariance_matrix: np.array) -> float:
    """
    :param covariance_matrix: Symmetric positive definite covariance matrix K of r.v. vector x ~ N(m, K)
    :return: differential entropy
    """
    det = np.linalg.det(covariance_matrix)
    n = covariance_matrix.shape[0]
    return 0.5 * log2(((2 * pi * e) ** n) * det)
