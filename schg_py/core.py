from __future__ import annotations

import time
import numpy as np
from scipy import sparse

from .finch import finchpp
from .graph_utils import calc_laps, calc_view_objs, weighted_sum, graph_avg, first_nn_merge
from .utils import get_global_p, onehot_to_labels


def solve_Y(L, Y, grid_cnt=None, max_iter: int = 50, early_stop: bool = True):
    p = get_global_p()
    Y = np.asarray(Y, dtype=float).copy()
    if sparse.issparse(L):
        L = L.toarray()
    else:
        L = np.asarray(L, dtype=float)

    n_cluster = np.sum(Y, axis=0)
    if grid_cnt is None or len(np.asarray(grid_cnt).reshape(-1)) == 0:
        n = Y.shape[0]
        grid_cnt = np.ones(n, dtype=float)
        n_grid = n_cluster.copy()
    else:
        grid_cnt = np.asarray(grid_cnt).reshape(-1).astype(float)
        n = int(np.sum(grid_cnt))
        n_grid = Y.T @ grid_cnt

    YL = Y.T @ L
    yLy = np.diag(YL @ Y).copy()
    yyn = np.maximum(n_grid**p, np.finfo(float).eps)
    p_all = np.argmax(Y, axis=1)

    obj = [float(np.sum(yLy / yyn))]
    for _iter in range(max_iter):
        for i in range(Y.shape[0]):
            m = p_all[i]
            if n_cluster[m] == 1:
                continue

            Lii = L[i, i]
            yLy_k = yLy + 2.0 * YL[:, i] + Lii
            yLy_k[m] = yLy[m]

            yyn_k = np.maximum((n_grid + grid_cnt[i]) ** p, np.finfo(float).eps)
            yyn_k[m] = yyn[m]

            yLy_0 = yLy.copy()
            yLy_0[m] = yLy[m] - 2.0 * YL[m, i] + Lii

            yyn_0 = yyn.copy()
            yyn_0[m] = max((n_grid[m] - grid_cnt[i]) ** p, np.finfo(float).eps)

            delta = yLy_k / yyn_k - yLy_0 / yyn_0
            r = int(np.argmin(delta))
            if r != m:
                yLy[m] = yLy_0[m]
                yyn[m] = yyn_0[m]
                yLy[r] = yLy_k[r]
                yyn[r] = yyn_k[r]

                Li = L[i, :]
                YL[r, :] = YL[r, :] + Li
                YL[m, :] = YL[m, :] - Li

                n_cluster[r] += 1
                n_cluster[m] -= 1

                n_grid[r] += grid_cnt[i]
                n_grid[m] -= grid_cnt[i]

                Y[i, r] = 1.0
                Y[i, m] = 0.0
                p_all[i] = r

        obj.append(float(np.sum(yLy / yyn)))
        if early_stop and len(obj) > 3:
            prev = obj[-2]
            curr = obj[-1]
            if abs((curr - prev) / max(abs(prev), np.finfo(float).eps)) < 1e-9:
                break
    return Y, np.asarray(obj)


def scHG(As, Y, grid_cnt=None, max_iter: int = 50):
    Ls = calc_laps(As)
    obj = []
    coeff = None
    for iter_idx in range(max_iter):
        view_objs = calc_view_objs(Ls, Y, grid_cnt)
        coeff = 1.0 / np.maximum(2.0 * view_objs, np.finfo(float).eps)
        obj.append(float(np.sum(view_objs)))
        if iter_idx > 10:
            prev = obj[-2]
            curr = obj[-1]
            if abs((curr - prev) / max(abs(prev), np.finfo(float).eps)) < 1e-20:
                break
        L = weighted_sum(Ls, coeff)
        Y, _ = solve_Y(L, Y, grid_cnt)
    return Y, np.asarray(obj), coeff


def run_scHG(As, num_clusters, use_grid=True, k_n=5, same_nn=1, current_seed=None):
    if use_grid:
        start = time.perf_counter()
        y_coar, coar_grid_cnt, As_coar = first_nn_merge(As, k_n, same_nn, seed=current_seed)
        evaltime = time.perf_counter() - start
        n = As_coar[0].shape[0]

        Y_init_coar = finchpp(graph_avg(As_coar).toarray(), num_clusters)

        start = time.perf_counter()
        y_pred, obj, coeff = scHG(As_coar, Y_init_coar, coar_grid_cnt)
        evaltime += time.perf_counter() - start
        y_pred = onehot_to_labels(y_pred)
        y_pred = y_pred[np.asarray(y_coar).reshape(-1) - 1]
        return y_pred, obj, coeff, n, y_coar, evaltime

    y_coar = None
    n = As[0].shape[0]
    Y_init = finchpp(graph_avg(As).toarray(), num_clusters)
    start = time.perf_counter()
    y_pred, obj, coeff = scHG(As, Y_init)
    evaltime = time.perf_counter() - start
    y_pred = onehot_to_labels(y_pred)
    return y_pred, obj, coeff, n, y_coar, evaltime
