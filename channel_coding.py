import numpy as np
from basics import entropy
from math import log2


def is_weakly_symmetric(q: np.array) -> bool:
    """
    :param q: Time-Invariant Transition-Probability Matrix
    :return: Whether the channel is weakly symmetric
    """
    assert np.all(q.sum(axis=-1) == 1)
    same_column_sum = np.all(q.sum(axis=0) == q.sum(axis=0)[0])
    rows_permutations = np.all(np.sort(q) == np.sort(q)[0])
    return rows_permutations and same_column_sum


def is_symmetric(q: np.array) -> bool:
    """
    :param q: Time-Invariant Transition-Probability Matrix
    :return: Whether the channel is symmetric
    """
    assert np.all(q.sum(axis=-1) == 1)
    cols_permutations = np.all(np.sort(q.T) == np.sort(q.T)[0])
    rows_permutations = np.all(np.sort(q) == np.sort(q)[0])
    return rows_permutations and cols_permutations


def info_capacity_ws(q: np.array, y_size: int) -> float:
    """
    :param q: Time-Invariant Transition-Probability Matrix
    :param y_size: Size of output set
    :return: Information capacity of channel
    """
    assert is_weakly_symmetric(q)
    row_h = entropy(q[0])
    # I(X;Y) = H(Y) - H(Q_row), maximum value of H(Y) is log|Y|
    return log2(y_size) - row_h


def info_capacity_bsc(f: float) -> float:
    """
    :param f: Bit switch probability
    :return: Information capacity of channel
    """
    return 1 - entropy(np.array([f, 1 - f]))


def info_capacity_bec(f: float) -> float:
    """
    :param f: Erasure probability
    :return: Information capacity of channel
    """
    # I(X;Y) = (1 -f)H(X). H(X) is maximum (1) when X is uniformly distributed which makes the max capacity 1 -f
    return 1 - f


def f_n(n: int) -> np.array:
    """
    :param n: f will be a n x n square matrix
    :return: n x n generator matrix for polar coding
    """
    assert n >= 2
    if n == 2:
        return np.array([[1, 0], [1, 1]])
    f_n_minus_1 = f_n(n - 1)
    row1 = np.concatenate((f_n_minus_1, np.zeros_like(f_n_minus_1)), axis=-1)
    row2 = np.concatenate((f_n_minus_1, f_n_minus_1), axis=-1)
    return np.concatenate((row1, row2), axis=0)


def polar_encode(u: np.array) -> np.array:
    """
    :param u: length-N input to the encoder
    :return: x = uF_N is the codeword
    """
    f = f_n(u.shape[0])
    # for 2 x 2, x_1 = u_1 + u_2, x_2 = u_2
    return np.matmul(u, f) % 2


if __name__ == '__main__':
    print(polar_encode(np.array([1, 1])))
