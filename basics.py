import numpy as np


def log(inp: np.array) -> np.array:
    """
    :param inp: array to log
    :return: log base 2 elementwise, ignoring zero entries
    """
    return np.ma.log2(inp).filled(0)


def div(inp1: np.array, inp2: np.array) -> np.array:
    """
    :param inp1: numberator
    :param inp2: array to log
    :return: inp1 / inp2
    """
    return np.ma.divide(inp1, inp2).filled(0)


def entropy(p_distr: np.array) -> float:
    """
    :param p_distr: array of probabilities
    :return: entropy of random variable
    """
    assert p_distr.sum() == 1
    return -1 * (log(p_distr) * p_distr).sum()


def joint_entropy(xy_distr: np.array) -> float:
    """
    :param xy_distr: 2D array with probabilities that X=x and Y=y
                Y = y_1          Y = y_2                  ...
    X = x_1     p(Y = y_1, X = x_1)  p(Y = y_2, X = x_1)  ...
    X = x_2     p(Y = y_1, X = x_2)  p(Y = y_2, X = x_2)  ...
    :return: joint entropy
    """
    assert xy_distr.sum() == 1
    return -1 * (log(xy_distr) * xy_distr).sum()


def conditional_entropy(xy_cond_distr: np.array, xy_joint_dist: np.array) -> float:
    """
    :param xy_cond_distr: 2D array with conditional probabilities that Y=y given X=x
                Y = y_1               Y = y_2               ...
    X = x_1     p(Y = y_1 | X = x_1)  p(Y = y_2 | X = x_1)  ...
    X = x_2     p(Y = y_1 | X = x_2)  p(Y = y_2 | X = x_2)  ...
    ...
    :param xy_joint_dist: 2D array with probabilities that X=x and Y=y
                Y = y_1          Y = y_2                  ...
    X = x_1     p(Y = y_1, X = x_1)  p(Y = y_2, X = x_1)  ...
    X = x_2     p(Y = y_1, X = x_2)  p(Y = y_2, X = x_2)  ...
    :return: joint entropy
    """
    assert np.all(xy_cond_distr.sum(axis=-1) == 1)
    assert xy_joint_dist.sum() == 1
    return -1 * (log(xy_cond_distr) * xy_joint_dist).sum()


def mutual_information(xy_distr: np.array) -> float:
    """
    :param xy_distr: Joint probability distribution of x and y
    :return: Mutual information of x and y
    """
    assert xy_distr.sum() == 1
    p_x = xy_distr.sum(axis=-1)
    p_y = xy_distr.sum(axis=0)
    # I(X;Y) = H(X) - H(X|Y) = H(X) + H(Y) - H(X,Y)
    return entropy(p_x) + entropy(p_y) - joint_entropy(xy_distr)


def kl_divergence(p: np.array, q: np.array) -> float:
    """
    :param p: probability mass vector
    :param q: probability mass vector
    :return: relative entropy or k-l divergence
    """
    assert p.sum() == 1
    assert q.sum() == 1
    return (p * log(div(p, q))).sum()

