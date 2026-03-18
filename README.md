# 📌 scHG Python

A Python reimplementation of **scHG**.

This project ports the original MATLAB implementation into a modular and reproducible Python pipeline, supporting:

* Multi-omics single-cell data (e.g., RNA + ATAC)
* Graph construction via PCC-based kNN
* FINCH++ initialization
* scHG optimization
* Clustering evaluation (ARI / NMI / ACC / Purity)

The MATLAB version of scHG can be downloaded via the link: https://github.com/anchor-hue/scHG.

---

# 📂 Project Structure

```text
schg_python/
│
├── data/                      # Example datasets (CSV / MAT)
│
├── schg_py/                  # Core library
│   ├── __init__.py
│   ├── core.py               # scHG optimization (solve_Y, run_scHG)
│   ├── finch.py              # FINCH / FINCH++ clustering
│   ├── graph_utils.py        # Graph construction & operations
│   ├── io_utils.py           # Data loading (CSV / MAT / NPY)
│   ├── metrics.py            # Clustering evaluation
│   ├── utils.py              # Utilities (z-score, global params, etc.)
│
├── demo.py                   # MATLAB demo.m equivalent (recommended entry)
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

```bash
git clone https://github.com/anchor-hue/schg_python.git
cd schg_python

pip install -r requirements.txt
```

---

# 🚀 Quick Start

## 1️⃣ Run demo (recommended)

This script reproduces the behavior of the original `demo.m`.

```bash
python demo.py
```

---

## 2️⃣ Data format

### ✅ Feature matrices (multi-view)

Each view should be a matrix:

```
shape = (cells, features)
```

Example:

* RNA: `(6661, 2000)`
* ATAC: `(6661, 5000)`


---

### ✅ Label file

A single-column CSV:

```csv
1
2
1
3
...
```

---

# 🔬 Pipeline Overview

```text
Raw multi-omics data (RNA / ATAC / ...)
            │
            ▼
     z-score normalization
            │
            ▼
   constructW_PKN (PCC graph)
            │
            ▼
      Graph fusion / merge
            │
            ▼
        FINCH++ init
            │
            ▼
          scHG
            │
            ▼
       Clustering output
            │
            ▼
     ARI / NMI / ACC / Purity
```

---


# 📊 Output Example

```json
{
  "num_clusters": 8,
  "n_after_graph_stage": 6661,
  "evaltime": 4.21,
  "coeff": [0.68, 0.31],
  "metrics": {
    "ARI": 0.73,
    "NMI": 0.81,
    "ACC": 0.79,
    "Purity": 0.85
  }
}
```

---

# 🧠 Correspondence to MATLAB Version

| MATLAB                    | Python                   |
| ------------------------- | ------------------------ |
| `demo.m`                  | `demo.py`                |
| `constructW_PKN.m`        | `constructW_PKN`         |
| `run_scHG.m`              | `run_scHG`               |
| `ClusteringMeasure_new.m` | `clustering_measure_new` |
| `zscore(x,0,2)`           | `zscore_rows(x)`         |

---

# 📄 Citation

If you use this code, please cite the original scHG paper.

---

# ⭐ Acknowledgements

* Original MATLAB implementation of scHG
* FINCH clustering framework

---
