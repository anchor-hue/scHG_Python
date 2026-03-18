from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .utils import labels_to_onehot, get_global_p


def constructW_PKN(X, k: int = 5, issymmetric: int = 1):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    pcc = np.corrcoef(X)
    D = 1.0 - pcc
    D = np.nan_to_num(D, nan=1.0)

    W = sparse.lil_matrix((n, n), dtype=float)
    idx = np.argsort(D, axis=1)[:, : k + 2]
    D_sorted = np.take_along_axis(D, idx, axis=1)
    D_sorted = D_sorted[:, 1:]
    idx = idx[:, 1:]

    k_neighbor_dist = D_sorted[:, :k]
    sigma = D_sorted[:, k]
    numerator = sigma[:, None] - k_neighbor_dist
    denominator = k * sigma - np.sum(k_neighbor_dist, axis=1) + np.finfo(float).eps
    weights = numerator / denominator[:, None]

    row = np.repeat(np.arange(n), k)
    col = idx[:, :k].reshape(-1)
    W[row, col] = weights.reshape(-1)

    W = W.tocsr()
    if issymmetric:
        W = (W + W.T) * 0.5
    return W


def delete_class(X, Y, r):
    Y = np.asarray(Y).reshape(-1)
    unique_classes, class_labels = np.unique(Y, return_inverse=True)
    class_counts = np.bincount(class_labels)
    sort_idx = np.argsort(class_counts)
    sorted_classes = unique_classes[sort_idx]
    num_classes = len(sorted_classes)
    num_remove = round(r * num_classes)
    classes_to_remove = sorted_classes[:num_remove] if num_remove > 0 else np.array([], dtype=Y.dtype)
    mask = ~np.isin(Y, classes_to_remove)
    Y_processed = Y[mask]
    X_processed = [np.asarray(x)[mask, :] for x in X]
    return X_processed, Y_processed


def same_edge_precision(y1, y2):
    G1 = labels_to_onehot(np.asarray(y1).reshape(-1))
    G2 = labels_to_onehot(np.asarray(y2).reshape(-1))
    A1 = G1 @ G1.T > 0
    A2 = G2 @ G2.T > 0
    denom = np.sum(A1)
    return float(np.sum(A1 & A2) / denom) if denom > 0 else 0.0


def select_k_columns_by_var(mat_in, k):
    mat_in = np.asarray(mat_in)
    variances = np.var(mat_in, axis=0, ddof=1)
    sorted_idx = np.argsort(-variances)
    top_k_idx = sorted_idx[: min(k, mat_in.shape[1])]
    return mat_in[:, top_k_idx]


def struct_gn(first_gn, same_nn):
    first_gn = np.asarray(first_gn, dtype=int)
    n, m = first_gn.shape
    out = sparse.lil_matrix((n, n), dtype=int)
    for i in range(n):
        row_i = set(first_gn[i, :].tolist())
        for j in range(m):
            k = first_gn[i, j]
            if k < 0 or k >= n:
                continue
            if len(row_i.intersection(first_gn[k, :].tolist())) >= same_nn:
                out[i, k] = 1
    return out.tocsr()


def calc_laps(As):
    Ls = []
    for A in As:
        A = sparse.csr_matrix(A)
        n = A.shape[0]
        L = sparse.diags(np.asarray(A.sum(axis=1)).reshape(-1), 0, shape=(n, n)) - A
        Ls.append(L.tocsr())
    return Ls


def calc_view_objs(Ls, Y, grid_cnt=None):
    Y = np.asarray(Y, dtype=float)
    p = get_global_p()
    if grid_cnt is None or len(np.asarray(grid_cnt).reshape(-1)) == 0:
        n_grid = np.sum(Y, axis=0)
    else:
        grid_cnt = np.asarray(grid_cnt).reshape(-1)
        n_grid = Y.T @ grid_cnt
    yyn = 1.0 / np.maximum(n_grid**p, np.finfo(float).eps)

    objs = []
    for L in Ls:
        LY = L @ Y
        col_l1 = np.sum(np.abs(LY), axis=0)
        val = float(np.sqrt(col_l1 @ yyn))
        objs.append(val)
    return np.asarray(objs)


def struct_nn(first_gn, same_nn=None):
    first_gn = np.asarray(first_gn, dtype=int)
    n = first_gn.shape[0]
    out = sparse.lil_matrix((n, n), dtype=int)
    for i in range(n):
        j = first_gn[i, 0]
        out[i, j] = 1
        out[j, i] = 1
    return out.tocsr()


def graph_avg(As):
    out = sparse.csr_matrix(As[0].shape, dtype=float)
    for A in As:
        out = out + sparse.csr_matrix(A)
    return out / len(As)


def weighted_sum(As, coeff):
    out = sparse.csr_matrix(As[0].shape, dtype=float)
    for A, c in zip(As, coeff):
        out = out + sparse.csr_matrix(A) * float(c)
    return out


def first_nn_merge(As, k_n, same_nn, seed=None):
    rng = np.random.default_rng(seed)
    num_views = len(As)
    n = As[0].shape[0]

    As = [sparse.csr_matrix(A).copy() for A in As]
    for A in As:
        A.setdiag(0)
        A.eliminate_zeros()

    first_gns = []
    for A in As:
        dense = A.toarray()
        idx = np.argsort(-dense, axis=1)[:, :k_n]
        first_gns.append(idx)

    G_first_gns = [struct_gn(first_gn, same_nn) for first_gn in first_gns]

    G_shared = sparse.csr_matrix((n, n), dtype=int)
    for G in G_first_gns:
        G_shared = G_shared + G
    threshold = num_views // 2 + 1
    G_shared = (G_shared >= threshold).astype(int)
    G_shared = G_shared + G_shared.T
    G_shared = (G_shared > 0).astype(int)

    _, y0 = connected_components(G_shared, directed=False, connection='weak')
    y = y0.copy() + 1
    current_max = int(y.max())

    for c in np.unique(y):
        nodes = np.where(y == c)[0]
        if nodes.size <= 1:
            continue
        sub = G_shared[nodes][:, nodes]
        deg = np.asarray(sub.sum(axis=1)).reshape(-1)
        min_s = deg.min()
        max_s = deg.max()
        if max_s == min_s:
            pro = np.zeros_like(deg, dtype=float)
        else:
            pro = 1.0 - (deg - min_s) / (max_s - min_s)
        for node, pr in zip(nodes, pro):
            if rng.random() < pr:
                current_max += 1
                y[node] = current_max

    Y = labels_to_onehot(y)
    cnt = np.sum(Y, axis=0).astype(int)
    coarsened = [Y.T @ sparse.csr_matrix(A) @ Y for A in As]
    return y, cnt, coarsened
