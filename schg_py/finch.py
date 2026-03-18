from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .utils import labels_to_onehot, relabel_consecutive, zero_diagonal


def clust_rank(orig_sim: np.ndarray | sparse.spmatrix, initial_rank=None):
    """Python port of MATLAB clustRank.

    Parameters
    ----------
    orig_sim:
        Similarity matrix. Only used when initial_rank is None.
    initial_rank:
        1-nearest-neighbor indices. Can be 0-based or 1-based.

    Returns
    -------
    A : csr_matrix
        FINCH affinity matrix.
    orig_sim_out : ndarray | None
        Original similarity matrix (or None when initial_rank is provided).
    min_sim : float
        Minimum first-neighbor similarity.
    """
    if sparse.issparse(orig_sim):
        s = orig_sim.shape[0]
    else:
        orig_sim = np.asarray(orig_sim)
        s = orig_sim.shape[0]

    if initial_rank is not None and len(np.asarray(initial_rank).reshape(-1)) > 0:
        initial_rank = np.asarray(initial_rank).reshape(-1).astype(int)
        if initial_rank.min() >= 1:
            initial_rank = initial_rank - 1
        orig_sim_out = None
        min_sim = np.inf
    else:
        if sparse.issparse(orig_sim):
            sim = zero_diagonal(orig_sim).toarray()
        else:
            sim = zero_diagonal(orig_sim)
        initial_rank = np.argmax(sim, axis=1).astype(int)
        d = sim[np.arange(s), initial_rank]
        min_sim = float(np.min(d))
        orig_sim_out = sim

    rows = np.arange(s)
    A = sparse.csr_matrix((np.ones(s), (rows, initial_rank)), shape=(s, s))
    A = A + sparse.eye(s, format='csr')
    A = A @ A.T
    A = zero_diagonal(A)
    A.data[:] = 1.0
    A.eliminate_zeros()
    return A.tocsr(), orig_sim_out, min_sim


def get_clust(A: sparse.spmatrix, orig_dist=None, min_sim=np.inf) -> np.ndarray:
    A = A.tocsr(copy=True)
    if np.isfinite(min_sim) and orig_dist is not None:
        orig_dist = np.asarray(orig_dist)
        coo = A.tocoo()
        keep = (orig_dist[coo.row, coo.col] * coo.data) <= min_sim
        A = sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=A.shape)
    _, labels = connected_components(A, directed=True, connection='weak')
    return labels + 1


def _propagate_labels(prev_labels: np.ndarray, u: np.ndarray) -> np.ndarray:
    _, ig = np.unique(prev_labels, return_inverse=True)
    return u[ig]


def get_merge(c, u, data):
    u = np.asarray(u).reshape(-1)
    u = relabel_consecutive(u)
    onehot = labels_to_onehot(u)
    num_clust = onehot.shape[1]

    if c is not None and len(np.asarray(c).reshape(-1)) > 0:
        c = _propagate_labels(np.asarray(c).reshape(-1), u)
    else:
        c = u.copy()

    data = np.asarray(data, dtype=float)
    mat = onehot.T @ data @ onehot
    cnt_mul = np.sum(onehot, axis=0).reshape(-1, 1)
    cnt_mul = cnt_mul @ cnt_mul.T
    mat = mat / np.maximum(cnt_mul, np.finfo(float).eps)
    return c, int(num_clust), mat


def FINCH(mat, initial_rank=None, verbose: int = 1):
    min_sim = np.inf
    A, orig_sim, _ = clust_rank(mat, initial_rank)
    initial_rank = None

    group = get_clust(A, None, np.inf)
    c, num_clust, mat = get_merge(None, group, mat)
    if verbose == 1:
        print(f'Partition 1 : {num_clust} clusters')

    exit_clust = np.inf
    c_curr = c.copy()
    num_clust_list = [num_clust]
    partitions = [c.copy()]
    k = 2

    while exit_clust > 1:
        A, orig_sim, _ = clust_rank(mat, initial_rank)
        u = get_clust(A, orig_sim, min_sim)
        c_curr, num_clust_curr, mat = get_merge(c_curr, u, mat)

        num_clust_list.append(num_clust_curr)
        partitions.append(c_curr.copy())

        exit_clust = num_clust_list[-2] - num_clust_curr
        if num_clust_curr == 1 or exit_clust < 1:
            num_clust_list = num_clust_list[:-1]
            partitions = partitions[:-1]
            break

        if verbose == 1:
            print(f'Partition {k} : {num_clust_list[k - 1]} clusters')
        k += 1

    c_mat = np.column_stack(partitions) if partitions else np.empty((len(mat), 0), dtype=int)
    return c_mat, np.array(num_clust_list, dtype=int)


def _top_affinity(affinity: sparse.spmatrix, orig_sim: np.ndarray) -> sparse.csr_matrix:
    affinity = zero_diagonal(affinity).tocsr()
    coo = affinity.tocoo()
    if coo.nnz == 0:
        return affinity
    values = orig_sim[coo.row, coo.col]
    idx = np.argmax(values)
    i = int(coo.row[idx])
    j = int(coo.col[idx])
    out = sparse.csr_matrix(([1.0, 1.0], ([i, j], [j, i])), shape=affinity.shape)
    return out


def req_numclust(c: np.ndarray, data: np.ndarray, req_clust: int) -> np.ndarray:
    c = np.asarray(c).reshape(-1)
    iter_n = len(np.unique(c)) - int(req_clust)
    c_curr, _, mat = get_merge(None, c, data)
    for _ in range(iter_n):
        affinity, orig_sim, _ = clust_rank(mat, None)
        affinity = _top_affinity(affinity, orig_sim)
        u = get_clust(affinity, None, np.inf)
        c_curr, _, mat = get_merge(c_curr, u, mat)
    return relabel_consecutive(c_curr)


def finchpp(A0: np.ndarray, c: int) -> np.ndarray:
    esti_y, esti_num_clust = FINCH(A0, None, 0)
    matched = np.where(esti_num_clust == c)[0]
    if matched.size > 0:
        y_init = esti_y[:, matched[0]]
    elif np.any(esti_num_clust > c):
        refine_starter = np.where(esti_num_clust > c)[0][-1]
        y_init = req_numclust(esti_y[:, refine_starter], A0, c)
    else:
        raise ValueError('FINCH failed to find cluster')
    return labels_to_onehot(y_init)
