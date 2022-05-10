import numpy as np
from basics import log


def stationary_distribution(transition_matrix: np.array) -> np.array:
    """
    :param transition_matrix: Matrix of state transition probabilities
    :return: Stationary distribution of markov process
    """
    s, u = np.linalg.eig(transition_matrix.T)
    stationary = np.array(u[:, np.where(np.abs(s - 1.) < 1e-8)[0][0]].flat)
    return stationary / np.sum(stationary)


def entropy_rate_markov(transition_matrix: np.array) -> float:
    """
    :param transition_matrix: Matrix of state transition probabilities
    :return: Entropy rate of markov process
    """
    p_s = np.expand_dims(stationary_distribution(transition_matrix), axis=-1)
    return (-1 * p_s * transition_matrix * log(transition_matrix)).sum()



