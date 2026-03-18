from __future__ import annotations

import json
import numpy as np

from schg_py import (
    clustering_measure_new,
    constructW_PKN,
    load_labels,
    load_views,
    run_scHG,
    set_global_p,
    zscore_rows,
)


def main():
    # =========================
    # 对齐 MATLAB demo.m 的固定参数
    # =========================
    set_global_p(0.0)
    k_n = 5
    same_nn = 2
    # gamma = 10  # 10x
    gamma = 20  #
    current_seed = 142
    use_grid = True

    # =========================
    # 数据路径
    # 这里按你当前 CSV 版本写
    # 如果以后换数据，只改这里
    # =========================
    # feature_paths = [
    #     "./data/pbmc_10x_X1.csv",
    #     "./data/pbmc_10x_X2.csv",
    # ]
    # label_path = "./data/pbmc_10x_truth.csv"
    feature_paths = [
        "./data/atac_counts_dense.csv",
        "./data/rna_counts.csv",
    ]  # 大规模模拟数据集100,000个细胞
    label_path = "./data/metadata.csv"


    # =========================
    # 读取数据
    # =========================
    # Xs = load_views(feature_paths, transpose=True, has_header=False, has_index=False)  # 10x需要转置

    Xs = load_views(feature_paths, transpose=False, has_header=True, has_index=True)  #

    Xs = [zscore_rows(x) for x in Xs]

    y_true = load_labels(label_path)

    # num_clusters = 8 # 10x
    num_clusters = len(np.unique(y_true))
    

    print(f"[demo] num_views = {len(Xs)}")
    for i, x in enumerate(Xs):
        print(f"[demo] view {i} shape = {x.shape}")
    print(f"[demo] num_clusters = {num_clusters}")
    print(f"[demo] label_length = {len(y_true)}")

    # =========================
    # 构图
    # =========================
    As = [constructW_PKN(x, k=gamma, issymmetric=1) for x in Xs]

    for i, A in enumerate(As):
        print(f"[demo] graph {i} shape = {A.shape}")

    # =========================
    # 运行 scHG
    # =========================
    y_pred, obj, coeff, n_g, y_coar, evaltime = run_scHG(
        As,
        num_clusters=num_clusters,
        use_grid=use_grid,
        k_n=k_n,
        same_nn=same_nn,
        current_seed=current_seed,
    )

    # =========================
    # 评估
    # =========================
    metrics = clustering_measure_new(y_true, y_pred)

    report = {
        "num_clusters": int(num_clusters),
        "n_after_graph_stage": int(n_g),
        "evaltime": float(evaltime),
        "obj": np.asarray(obj, dtype=float).tolist(),
        "coeff": np.asarray(coeff, dtype=float).tolist(),
        "metrics": metrics,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()