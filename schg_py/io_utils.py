from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import io as sio
from scipy import sparse

from scipy.io import loadmat

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import loadmat

SUPPORTED_FEATURE_KEYS = (
    'X', 'data', 'features', 'feature', 'mat', 'view', 'A', 'W', 'adj', 'graph'
)
SUPPORTED_LABEL_KEYS = ('Y', 'y', 'label', 'labels', 'gt', 'truth')


def _find_first_key(d: dict, candidates: Iterable[str]):
    for k in candidates:
        if k in d:
            return k
    return None


# def load_array(path: str | Path, key: str | None = None) -> np.ndarray:
#     path = Path(path)
#     suffix = path.suffix.lower()
#     if suffix == '.npy':
#         return np.load(path, allow_pickle=True)
#     if suffix == '.npz':
#         obj = np.load(path, allow_pickle=True)
#         if key is None:
#             key = obj.files[0]
#         return obj[key]
#     if suffix in {'.csv', '.txt', '.tsv'}:
#         delimiter = ',' if suffix == '.csv' else None
#         return np.loadtxt(path, delimiter=delimiter)
#     if suffix == '.mat':
#         obj = sio.loadmat(path)
#         if key is None:
#             key = _find_first_key(obj, SUPPORTED_FEATURE_KEYS)
#         if key is None:
#             valid = [k for k in obj.keys() if not k.startswith('__')]
#             if len(valid) == 1:
#                 key = valid[0]
#             else:
#                 raise KeyError(f'Cannot infer variable key from {path}. Available keys: {valid}')
#         return np.asarray(obj[key])
#     raise ValueError(f'Unsupported file format: {path}')
# def load_array(path, key=None, delimiter=","):
#     path = Path(path)
#     suffix = path.suffix.lower()

#     if suffix == ".npy":
#         return np.load(path, allow_pickle=True)

#     elif suffix == ".npz":
#         data = np.load(path, allow_pickle=True)
#         if key is not None:
#             return data[key]
#         keys = list(data.keys())
#         if len(keys) == 1:
#             return data[keys[0]]
#         raise ValueError(f"{path} contains multiple arrays {keys}, please specify key")

#     elif suffix == ".mat":
#         data = loadmat(path)
#         if key is not None:
#             return data[key]
#         valid_keys = [k for k in data.keys() if not k.startswith("__")]
#         if len(valid_keys) == 1:
#             return data[valid_keys[0]]
#         raise ValueError(f"{path} contains multiple variables {valid_keys}, please specify key")

#     elif suffix in [".csv", ".txt", ".tsv"]:
#         if suffix == ".tsv":
#             delimiter = "\t"
#         return np.loadtxt(path, delimiter=delimiter)

#     else:
#         raise ValueError(f"Unsupported file type: {path}")
# def load_array(path, key=None, delimiter=","):
#     path = Path(path)
#     suffix = path.suffix.lower()

#     if suffix == ".csv":
#         df = pd.read_csv(path, index_col=0)
#         return df.values.astype(float)

#     elif suffix == ".txt":
#         return np.loadtxt(path, delimiter=delimiter)

#     elif suffix == ".npy":
#         return np.load(path)

#     elif suffix == ".mat":
#         data = loadmat(path)
#         if key is not None:
#             return data[key]
#         valid_keys = [k for k in data.keys() if not k.startswith("__")]
#         return data[valid_keys[0]]

#     else:
#         raise ValueError(f"Unsupported file type: {path}")
def load_array(path, key=None, delimiter=",", has_header=False, has_index=False):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        header = 0 if has_header else None
        index_col = 0 if has_index else None
        df = pd.read_csv(path, header=header, index_col=index_col)
        return df.values.astype(float)

    elif suffix == ".txt":
        return np.loadtxt(path, delimiter=delimiter)

    elif suffix == ".npy":
        return np.load(path)

    elif suffix == ".mat":
        data = loadmat(path)
        if key is not None:
            return data[key]
        valid_keys = [k for k in data.keys() if not k.startswith("__")]
        return data[valid_keys[0]]

    else:
        raise ValueError(f"Unsupported file type: {path}")




# def load_labels(path: str | Path, key: str | None = None) -> np.ndarray:
#     arr = load_array(path, key=key)
#     arr = np.asarray(arr).reshape(-1)
#     _, inv = np.unique(arr, return_inverse=True)
#     return inv + 1
def load_labels(path: str | Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, header=None)
        arr = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy()
    else:
        arr = load_array(path, key=key)
        arr = np.asarray(arr).reshape(-1)

    print("[load_labels] length:", len(arr))

    _, inv = np.unique(arr, return_inverse=True)
    return inv + 1


# def load_views(feature_paths: list[str | Path], keys: list[str | None] | None = None) -> list[np.ndarray]:
#     if keys is None:
#         keys = [None] * len(feature_paths)
#     if len(keys) != len(feature_paths):
#         raise ValueError('keys and feature_paths must have the same length.')
#     return [np.asarray(load_array(p, key=k), dtype=float) for p, k in zip(feature_paths, keys)]
# def load_views(feature_paths, keys=None):
#     if keys is None:
#         keys = [None] * len(feature_paths)

#     Xs = [np.asarray(load_array(p, key=k), dtype=float) for p, k in zip(feature_paths, keys)]

#     for i, x in enumerate(Xs):
#         print(f"View {i}: path={feature_paths[i]}, shape={x.shape}")

#     return Xs
# def load_views(feature_paths, keys=None, transpose=True):
#     if keys is None:
#         keys = [None] * len(feature_paths)

#     Xs = []
#     for p, k in zip(feature_paths, keys):
#         x = np.asarray(load_array(p, key=k), dtype=float)
#         if transpose:
#             x = x.T
#         Xs.append(x)
#     return Xs
# def load_views(feature_paths, keys=None,  transpose=False):
#     if keys is None:
#         keys = [None] * len(feature_paths)

#     Xs = []
#     for i, (p, k) in enumerate(zip(feature_paths, keys)):
#         x = np.asarray(load_array(p, key=k), dtype=float)

#         print(f"[load_views] view {i}: path={p}, shape={x.shape}")

#         if transpose:
#             x = x.T

#         Xs.append(x)

#     return Xs
def load_views(feature_paths, keys=None, transpose=False, has_header=False, has_index=False):
    if keys is None:
        keys = [None] * len(feature_paths)

    Xs = []
    for i, (p, k) in enumerate(zip(feature_paths, keys)):
        x = np.asarray(
            load_array(p, key=k, has_header=has_header, has_index=has_index),
            dtype=float
        )

        print(f"[load_views] view {i}: path={p}, raw shape={x.shape}")

        if transpose:
            x = x.T

        print(f"[load_views] view {i}: final shape={x.shape}")
        Xs.append(x)

    return Xs



def load_graphs(graph_paths: list[str | Path], keys: list[str | None] | None = None):
    if keys is None:
        keys = [None] * len(graph_paths)
    if len(keys) != len(graph_paths):
        raise ValueError('keys and graph_paths must have the same length.')

    graphs = []
    for p, k in zip(graph_paths, keys):
        arr = load_array(p, key=k)
        graphs.append(sparse.csr_matrix(np.asarray(arr, dtype=float)))
    return graphs



# from __future__ import annotations

# from pathlib import Path
# from typing import Iterable

# import numpy as np
# from scipy import io as sio
# from scipy import sparse

# from scipy.io import loadmat

# SUPPORTED_FEATURE_KEYS = (
#     'X', 'data', 'features', 'feature', 'mat', 'view', 'A', 'W', 'adj', 'graph'
# )
# SUPPORTED_LABEL_KEYS = ('Y', 'y', 'label', 'labels', 'gt', 'truth')


# def _find_first_key(d: dict, candidates: Iterable[str]):
#     for k in candidates:
#         if k in d:
#             return k
#     return None


# def load_array(
#     path: str | Path, 
#     key: str | None = None, 
#     transpose: bool = True  # 新增：是否转置矩阵（默认True，适配你的需求）
# ) -> np.ndarray:
#     path = Path(path)
#     suffix = path.suffix.lower()
#     if suffix == '.npy':
#         arr = np.load(path, allow_pickle=True)
#     elif suffix == '.npz':
#         obj = np.load(path, allow_pickle=True)
#         if key is None:
#             key = obj.files[0]
#         arr = obj[key]
#     elif suffix in {'.csv', '.txt', '.tsv'}:
#         delimiter = ',' if suffix == '.csv' else None
#         arr = np.loadtxt(path, delimiter=delimiter)
#     elif suffix == '.mat':
#         obj = sio.loadmat(path)
#         if key is None:
#             key = _find_first_key(obj, SUPPORTED_FEATURE_KEYS)
#         if key is None:
#             valid = [k for k in obj.keys() if not k.startswith('__')]
#             if len(valid) == 1:
#                 key = valid[0]
#             else:
#                 raise KeyError(f'Cannot infer variable key from {path}. Available keys: {valid}')
#         arr = np.asarray(obj[key])
#     else:
#         raise ValueError(f'Unsupported file format: {path}')
    
#     # 核心改动：添加矩阵转置逻辑
#     if transpose:
#         arr = arr.T  # 转置矩阵，替代你在WPS中的手动转置
    
#     return arr


# def load_labels(path: str | Path, key: str | None = None, transpose: bool = False) -> np.ndarray:
#     # 标签文件一般不需要转置，默认关闭
#     arr = load_array(path, key=key, transpose=transpose)
#     arr = np.asarray(arr).reshape(-1)
#     _, inv = np.unique(arr, return_inverse=True)
#     return inv + 1


# def load_views(
#     feature_paths: list[str | Path], 
#     keys: list[str | None] | None = None,
#     transpose: bool = True  # 新增：统一控制所有视图的转置
# ) -> list[np.ndarray]:
#     if keys is None:
#         keys = [None] * len(feature_paths)
#     if len(keys) != len(feature_paths):
#         raise ValueError('keys and feature_paths must have the same length.')
    
#     # 加载时传入转置参数
#     views = [np.asarray(load_array(p, key=k, transpose=transpose), dtype=float) for p, k in zip(feature_paths, keys)]
    
#     # 保留你之前的打印逻辑（可选，方便查看维度）
#     for i, x in enumerate(views):
#         print(f"View {i}: path={feature_paths[i]}, shape={x.shape}")
    
#     return views


# def load_graphs(
#     graph_paths: list[str | Path], 
#     keys: list[str | None] | None = None,
#     transpose: bool = False  # 图矩阵一般不需要转置，默认关闭
# ):
#     if keys is None:
#         keys = [None] * len(graph_paths)
#     if len(keys) != len(graph_paths):
#         raise ValueError('keys and graph_paths must have the same length.')

#     graphs = []
#     for p, k in zip(graph_paths, keys):
#         arr = load_array(p, key=k, transpose=transpose)  # 传入转置参数
#         graphs.append(sparse.csr_matrix(np.asarray(arr, dtype=float)))
#     return graphs