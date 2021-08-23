import numpy as np


def sinc(x):
    if np.abs(x) <= 1e-20:
        return 1
    else:
        return np.sin(x)/x


def sinc1(x):
    if np.abs(x) < 1e-6:
        return -1/3 + x*x/2/3/5
    return (x*np.cos(x)-np.sin(x))/x/x/x


def sinc2(x):
    if np.abs(x) < 1e-3:
        return 1/15 - x*x/210 + x*x*x*x / 7560
    return -((x*x-3)*np.sin(x) + 3*x*np.cos(x))/x**5    


def dsinc(x):
    """Derivative of sinc
    """
    if np.abs(x) < 1e-6:
        return -1/3*x + x*x*x/30
    return (x*np.cos(x)-np.sin(x))/x/x


def dsinc1(x):
    """sinc1 is dsinc/x
    dsinc1 is its derivative
    """
    return x*sinc2(x)


def sbmat(obj):
    """ Simpler version of numpy.bmat.
    Return an array instead of matrix
    """
    arr_rows = []
    for row in obj:
        if isinstance(row, np.ndarray):  # not 2-d
            return np.concatenate(obj, axis=-1)
        else:
            arr_rows.append(np.concatenate(row, axis=-1))
    return np.concatenate(arr_rows, axis=0)


def sym2(mat):
    return mat + mat.T

def linf(mat):
    return np.max(np.abs(mat))

