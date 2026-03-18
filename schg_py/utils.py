import numpy as np
from scipy import sparse

P_GLOBAL = 1.0


def set_global_p(value: float) -> None:
    global P_GLOBAL
    P_GLOBAL = float(value)


def get_global_p() -> float:
    return float(P_GLOBAL)


def labels_to_onehot(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    uniq, inv = np.unique(labels, return_inverse=True)
    y = np.zeros((labels.size, uniq.size), dtype=float)
    y[np.arange(labels.size), inv] = 1.0
    return y


def onehot_to_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError('y must be a 2D array.')
    return np.argmax(y, axis=1) + 1


def relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    _, inv = np.unique(labels, return_inverse=True)
    return inv + 1


def zero_diagonal(a):
    if sparse.issparse(a):
        a = a.tolil(copy=True)
        a.setdiag(0)
        return a.tocsr()
    a = np.array(a, copy=True)
    np.fill_diagonal(a, 0)
    return a


def zscore_rows(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"zscore_rows expects a 2D array, got shape {x.shape}")

    mu = x.mean(axis=1, keepdims=True)
    sigma = x.std(axis=1, ddof=0, keepdims=True)
    sigma[sigma < eps] = 1.0
    return (x - mu) / sigma
