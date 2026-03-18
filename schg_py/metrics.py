from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, rand_score


def best_map(L1, L2):
    L1 = np.asarray(L1).reshape(-1)
    L2 = np.asarray(L2).reshape(-1)
    if L1.shape != L2.shape:
        raise ValueError('size(L1) must == size(L2)')

    _, L1n = np.unique(L1, return_inverse=True)
    _, L2n = np.unique(L2, return_inverse=True)
    L1n += 1
    L2n += 1
    n_class = max(L1n.max(), L2n.max())

    G = np.zeros((n_class, n_class), dtype=int)
    for i in range(1, n_class + 1):
        for j in range(1, n_class + 1):
            G[i - 1, j - 1] = np.sum((L1n == i) & (L2n == j))

    row_ind, col_ind = linear_sum_assignment(-G)
    mapping = np.zeros(n_class + 1, dtype=int)
    mapping[col_ind + 1] = row_ind + 1
    newL2 = mapping[L2n]
    return newL2, mapping[1:]


def mutual_info_hat(L1, L2):
    L1 = np.asarray(L1).reshape(-1)
    L2 = np.asarray(L2).reshape(-1)
    return normalized_mutual_info_score(L1, L2)


def compute_f(T, H):
    T = np.asarray(T).reshape(-1)
    H = np.asarray(H).reshape(-1)
    if T.size != H.size:
        raise ValueError('T and H must have the same length.')

    numT = 0
    numH = 0
    numI = 0
    N = T.size
    for n in range(N):
        Tn = (T[n + 1 :] == T[n])
        Hn = (H[n + 1 :] == H[n])
        numT += int(np.sum(Tn))
        numH += int(np.sum(Hn))
        numI += int(np.sum(Tn & Hn))

    p = 1.0 if numH == 0 else numI / numH
    r = 1.0 if numT == 0 else numI / numT
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return f, p, r


def purity_score(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    classes = np.unique(y_pred)
    correct = 0
    for c in classes:
        in_cluster = y_true[y_pred == c]
        if in_cluster.size == 0:
            continue
        vals, counts = np.unique(in_cluster, return_counts=True)
        correct += counts.max()
    return correct / y_pred.size


def clustering_measure_new(Y, predY):
    Y = np.asarray(Y).reshape(-1)
    predY = np.asarray(predY).reshape(-1)

    res, _ = best_map(Y, predY)
    acc = np.mean(Y == res)
    nmi = mutual_info_hat(Y, res)
    purity = purity_score(Y, predY)
    f, p, r = compute_f(Y, res)
    ri = rand_score(Y, res)
    ari = adjusted_rand_score(Y, res)

    return {
        'ACC': float(acc),
        'NMI': float(nmi),
        'Purity': float(purity),
        'F': float(f),
        'P': float(p),
        'R': float(r),
        'RI': float(ri),
        'ARI': float(ari),
    }
