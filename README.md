# STDIL: Spatiotemporal-Decoupled Interactive Learning for Traffic Flow Prediction

This repository provides the official implementation of **STDIL (Spatiotemporal-Decoupled Interactive Learning for Traffic Flow Prediction)**, a unified spatio-temporal learning framework for traffic flow forecasting. STDIL is designed to jointly model **short-term temporal dynamics**, **long-term historical dependencies**, and **spatial correlations** through a dual-interval learning mechanism and masked temporal modeling strategy. The framework is built upon the **BasicTS** training infrastructure and supports large-scale real-world traffic datasets.

---

## 📌 Overview

Traffic flow prediction is a fundamental task in intelligent transportation systems. However, real-world traffic series exhibit strong **multi-scale temporal dependencies** and **complex spatial correlations**. STDIL addresses these challenges by:

STDIL has been validated on multiple public benchmarks, including **PEMS03, PEMS04, PEMS07, and PEMS08**.

---

## 📂 Project Structure

```

STDIL/
│
├── STDIL_arch/              # Core model architecture
│   └── mask/                # Temporal masking modules
│
├── STDIL_data/              # Dataset definition for pretraining & forecasting
│   ├── forecasting_dataset.py
│   └── pretraining_dataset.py
│
├── STDIL_runner/            # Training & evaluation runner
│   └── STDIL_runner.py
│
├── basicts/                 
│
├── datasets/                # Dataset directory
│
├── run.py                   # Main training entry
└── README.md                # Project documentation

````


## 🛠 Environment Setup

### 1️⃣ Create Python Environment

```bash
conda create -n stdil python=3.8
conda activate stdil
````

### 2️⃣ Install Dependencies

```bash
pip install torch numpy einops scikit-learn pyyaml tqdm
```

If you are using GPU acceleration:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Dataset Preparation

STDIL supports the following benchmark datasets:

* **PEMS03**
* **PEMS04**
* **PEMS07**
* **PEMS08**

You should manually prepare the datasets with the following structure:

```
datasets/
└── PEMS04/
    ├── data_in12_out12.pkl
    ├── index_in12_out12.pkl
    ├── adj_mx.pkl
    └── scaler_in12_out12.pkl
```

Each dataset directory should contain:

* `data_in12_out12.pkl` : Preprocessed traffic time series
* `index_in12_out12.pkl`: Train/validation/test split indices
* `adj_mx.pkl`          : Graph adjacency matrix
* `scaler_in12_out12.pkl`: Normalization scaler
* 
---

## 🚀 Training & Evaluation

The main execution entry is:

```bash
run.py
```

### ✅ Run Training

```bash
python run.py -c STDIL/STDIL_PEMS04.py --gpus 0
```

Where:

* `-c` specifies the config file
* `--gpus` assigns visible GPU IDs

---

## 📈 Evaluation Metrics

STDIL adopts standard masked evaluation metrics, including:

* **MAE** – Mean Absolute Error
* **RMSE** – Root Mean Square Error
* **MAPE** – Mean Absolute Percentage Error

All metrics are computed under masked settings to handle missing values.

---

---

## 🔬 Reproducibility

All experimental results reported in the corresponding paper can be reproduced by:

1. Downloading prepared datasets
2. Placing them under the `datasets/` directory
3. Running the provided training scripts via `run.py`

---

## 📜 Citation

If you find STDIL useful in your research, please consider citing:

```
@article{STDIL,
  title   = {STDIL: Spatiotemporal-Decoupled Interactive Learning for Traffic Flow Prediction
},
  author  = { },
  journal = {Under Review},
  year    = {2025}
}
```

---

## 📄 License

This project is released **for academic research purposes only**.
Commercial usage is strictly prohibited without explicit permission from the authors.

---



