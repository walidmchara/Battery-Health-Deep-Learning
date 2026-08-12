# 🔋 Battery Health Deep Learning

### Deep Learning & Transformer-Based Lithium-Ion Battery State-of-Health Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Research-Battery%20AI-green.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research repository for **AI-driven lithium-ion battery State-of-Health (SOH) estimation and intelligent health diagnosis** using hybrid deep learning, wavelet-based feature extraction, attention mechanisms, and Transformer architectures.

This repository is part of the [AI Research Portfolio](https://github.com/walidmchara/AI-Research-Portfolio) of **Walid Mchara, PhD**.

---

## 🎯 Research Objective

Accurate estimation of lithium-ion battery health is essential for:

* electric vehicle safety,
* battery management systems,
* predictive maintenance,
* degradation monitoring,
* energy management,
* and reliable Remaining Useful Life assessment.

Battery degradation is nonlinear, operating-condition dependent, and difficult to model using conventional approaches.

This project investigates data-driven and hybrid AI architectures capable of learning both **local degradation patterns and long-term temporal dependencies** from battery cycling data.

---

## 🧠 Research Framework

The general modeling strategy combines several complementary learning mechanisms:

```text
Battery Cycling Data
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Wavelet / Multi-Scale Decomposition
        │
        ▼
CNN-Based Feature Extraction
        │
        ▼
BiLSTM / Sequential Modeling
        │
        ▼
Attention / Transformer
        │
        ▼
SOH Regression
        │
        ▼
Evaluation & Generalization Analysis
```

The architecture is designed to capture:

* short-term degradation signatures,
* nonlinear temporal dynamics,
* multi-scale battery behavior,
* long-range dependencies,
* and cross-battery degradation patterns.

---

## 🔬 Research Topics

This repository covers:

* Lithium-ion battery State-of-Health estimation
* Battery health diagnosis
* Battery degradation modeling
* Deep learning for battery management
* Wavelet-enhanced neural networks
* CNN feature extraction
* LSTM and BiLSTM modeling
* Attention mechanisms
* Transformer architectures
* Cross-cell generalization
* Multi-dataset evaluation
* Explainable battery intelligence

---

## 📊 Datasets

### NASA Battery Dataset

Experiments use publicly available lithium-ion battery aging data from the **NASA Prognostics Center of Excellence**.

The datasets contain battery charge/discharge cycles under different operating and aging conditions.

Typical information includes:

* voltage,
* current,
* temperature,
* capacity,
* cycle information,
* charge/discharge measurements.

---

### CALCE Battery Dataset

Additional experiments use lithium-ion battery degradation data from the **Center for Advanced Life Cycle Engineering (CALCE)**.

Using multiple battery datasets enables evaluation of model robustness and generalization across cells and experimental conditions.

> Large raw datasets are not directly stored in this repository. Instructions for obtaining and preparing the datasets will be provided in `data/README.md`.

---

## 📐 State-of-Health Definition

A commonly used capacity-based SOH definition is:

[
SOH_t = \frac{C_t}{C_{rated}} \times 100
]

where:

* (C_t) is the measured battery capacity at cycle (t),
* (C_{rated}) is the reference or rated battery capacity.

The model learns:

[
f(X_{1:t}) \rightarrow SOH_t
]

where (X_{1:t}) represents the battery measurements available up to cycle (t).

---

## 🏗️ Models

The repository will progressively provide implementations and benchmarks for:

### Baseline Models

* Linear Regression
* Random Forest
* Support Vector Regression
* XGBoost
* LSTM
* BiLSTM
* GRU

### Deep Learning Models

* CNN-LSTM
* CNN-BiLSTM
* Wavelet-enhanced neural networks
* Attention-based recurrent networks

### Transformer Models

* Transformer Encoder
* Attention-based Transformer
* Efficient Transformer variants

### Hybrid Architectures

The main research direction combines:

```text
Wavelet
   +
CNN
   +
BiLSTM
   +
Attention / Transformer
```

to exploit complementary multi-scale, spatial-feature, temporal, and long-range dependency learning.

---

## 📂 Repository Structure

```text
Battery-Health-Deep-Learning/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── configs/
│   ├── nasa.yaml
│   └── calce.yaml
│
├── data/
│   ├── README.md
│   └── sample/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_soh_prediction_demo.ipynb
│
├── src/
│   ├── dataset.py
│   ├── preprocessing.py
│   │
│   ├── models/
│   │   ├── lstm.py
│   │   ├── cnn_bilstm.py
│   │   ├── transformer.py
│   │   └── hybrid_model.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── results/
│   ├── figures/
│   └── metrics/
│
└── tests/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/walidmchara/Battery-Health-Deep-Learning.git
cd Battery-Health-Deep-Learning
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

or Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

Once the dataset has been prepared:

```bash
python -m src.train --config configs/nasa.yaml
```

Evaluate a trained model:

```bash
python -m src.evaluate --config configs/nasa.yaml
```

A demonstration notebook will also be available in:

```text
notebooks/03_soh_prediction_demo.ipynb
```

---

## 📏 Evaluation Metrics

Models are evaluated using:

### Mean Absolute Error

[
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
]

### Root Mean Squared Error

[
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}
]

### Mean Absolute Percentage Error

[
MAPE =
\frac{100}{N}
\sum_{i=1}^{N}
\left|
\frac{y_i-\hat{y}_i}{y_i}
\right|
]

### Coefficient of Determination

[
R^2 =
1-
\frac{
\sum_i(y_i-\hat{y}_i)^2
}{
\sum_i(y_i-\bar{y})^2
}
]

---

## 📈 Experimental Results

Verified experimental results will be reported here as the corresponding implementations and evaluation pipelines are released.

The benchmark table will follow this structure:

| Model        | Dataset | RMSE | MAE | MAPE | R² |
| ------------ | ------- | ---: | --: | ---: | -: |
| LSTM         | NASA    |    — |   — |    — |  — |
| BiLSTM       | NASA    |    — |   — |    — |  — |
| Transformer  | NASA    |    — |   — |    — |  — |
| Hybrid Model | NASA    |    — |   — |    — |  — |
| Hybrid Model | CALCE   |    — |   — |    — |  — |

Only results reproducible from the released experimental configuration will be reported.

---

## 🔄 Cross-Battery Generalization

A major objective is evaluating whether models trained on selected batteries can generalize to previously unseen cells.

The repository will support experiments of the form:

```text
Training Batteries
        │
        ▼
Model Training
        │
        ▼
Previously Unseen Battery
        │
        ▼
SOH Prediction
        │
        ▼
Generalization Evaluation
```

This setting is particularly important for practical battery-management applications.

---

## 🔍 Explainability

Future releases will include model interpretation tools such as:

* feature importance,
* attention visualization,
* SHAP analysis,
* degradation-regime analysis,
* prediction-error analysis.

The objective is to better understand **why** a model predicts a particular health state.

---

## 🔁 Reproducibility

Experiments will document:

* random seeds,
* dataset splits,
* preprocessing parameters,
* model hyperparameters,
* training configuration,
* software dependencies,
* evaluation protocols.

Configuration files will be stored under:

```text
configs/
```

---

## 🗺️ Roadmap

* [ ] Dataset preparation pipeline
* [ ] NASA preprocessing
* [ ] CALCE preprocessing
* [ ] LSTM baseline
* [ ] BiLSTM baseline
* [ ] CNN-BiLSTM implementation
* [ ] Transformer implementation
* [ ] Wavelet-enhanced hybrid architecture
* [ ] Cross-battery validation
* [ ] Benchmark experiments
* [ ] Attention visualization
* [ ] SHAP explainability
* [ ] Reproducible demonstration notebook

---

## 📚 Related Research

This repository is connected to my research on **AI-based lithium-ion battery health diagnosis, hybrid deep learning, wavelet-enhanced modeling, attention mechanisms, and intelligent electric-vehicle energy systems**.

Associated peer-reviewed publications will be linked here with their DOI and corresponding reproducible experiments.

---

## 👨‍🔬 Author

**Walid Mchara, PhD**

AI & Data Science Researcher

Research areas:

`Deep Learning` • `Transformers` • `Battery Intelligence` • `Energy AI` • `Time-Series Forecasting` • `Generative AI`

GitHub: [walidmchara](https://github.com/walidmchara)

Research portfolio: [AI-Research-Portfolio](https://github.com/walidmchara/AI-Research-Portfolio)

---

## 📄 License

This project is released under the **MIT License**.

---

## ⭐ Citation

If you use code or methodological components from this repository in academic research, please cite the corresponding publication associated with the implemented model.

Full BibTeX references will be provided alongside the released implementations.
