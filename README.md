# 🔋 Battery Health Deep Learning: Hybrid Models Comparison

### Comparative Analysis of Hybrid Deep Learning Models for Lithium-Ion Battery State-of-Health Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Research-Model%20Comparison-green.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Comprehensive research repository for **comparative analysis of hybrid deep learning architectures** for lithium-ion battery State-of-Health (SOH) estimation. This project evaluates multiple model architectures including LSTM, BiLSTM, GRU, CNN-LSTM, CNN-GRU, CNN-BiLSTM-Attention, CNN-BiGRU-Attention, and Transformer-based models across NASA and CALCE datasets.

This repository is part of the [AI Research Portfolio](https://github.com/walidmchara/AI-Research-Portfolio) of **Walid Mchara, PhD**.

---

## 🎯 Research Objective

This project systematically compares and benchmarks multiple hybrid deep learning architectures for accurate lithium-ion battery State-of-Health (SOH) estimation using:

* **NASA Battery Dataset** - Standard benchmark for battery degradation studies
* **CALCE Battery Dataset** - Real-world battery aging under varied operating conditions

Key research goals:

* Compare performance across diverse model architectures (RNN, CNN-RNN, Attention-based, Transformer)
* Identify optimal hybrid combinations for battery health prediction
* Evaluate generalization across different datasets and operating conditions
* Provide empirical evidence for model selection in battery management systems
* Enable predictive maintenance and remaining useful life (RUL) assessment

Battery degradation is nonlinear, operating-condition dependent, and difficult to model using conventional approaches. This project investigates which hybrid AI architectures best capture both **local degradation patterns and long-term temporal dependencies** from battery cycling data.

---

## 🧠 Hybrid Model Architectures Comparison

This project implements and compares the following deep learning architectures:

### Sequential Models
- **LSTM Regressor** - Standard LSTM for temporal sequence modeling
- **BiLSTM Regressor** - Bidirectional LSTM capturing both past and future context
- **GRU Regressor** - Gated Recurrent Unit with simpler gate mechanism

### Hybrid CNN-RNN Models
- **CNN-LSTM** - Convolutional feature extraction followed by LSTM temporal modeling
- **CNN-GRU** - Convolutional feature extraction followed by GRU temporal modeling
- **CNN-BiLSTM-Attention** - CNN features → BiLSTM → Attention mechanism
- **CNN-BiGRU-Attention** - CNN features → BiGRU → Attention mechanism

### Transformer-Based Models
- **Transformer Regressor** - Self-attention based encoder for sequence-to-regression

### Unified Pipeline

```text
Battery Cycling Data (NASA / CALCE)
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering & Normalization
        │
        ▼
Sequence Construction (Time Windows)
        │
        ├─────────────────────────────────────────────┐
        │                                             │
        ▼                                             ▼
Sequential Models                        Hybrid CNN-RNN Models
(LSTM, BiLSTM, GRU)                   (CNN-LSTM, CNN-GRU, etc.)
        │                                             │
        └─────────────────────────────────────────────┤
                        │                             │
                        ├──────────────────────────────┤
                        │                              │
                        ▼                              ▼
                Attention Models          Transformer Encoder
        (CNN-BiLSTM-Attention,                       │
         CNN-BiGRU-Attention)                        │
                        │                            │
                        └────────────────────────────┬┘
                                    │
                                    ▼
                        SOH Regression Predictions
                                    │
                                    ▼
                        Model Evaluation & Comparison
                    (MAE, RMSE, R², Generalization)
```

### Model Comparison Metrics
- **Mean Absolute Error (MAE)** - Average prediction deviation
- **Root Mean Squared Error (RMSE)** - Sensitivity to large errors
- **R² Score** - Explained variance ratio
- **Cross-Dataset Generalization** - Performance on unseen datasets
- **Inference Time** - Computational efficiency

---

## 🔬 Research Scope

This comparative study covers:

* **Model Architecture Comparison** - Performance across 8+ hybrid deep learning models
* **Multi-Dataset Evaluation** - NASA and CALCE battery datasets
* **State-of-Health Estimation** - Accurate SOH prediction for battery monitoring
* **Hybrid Model Design** - CNN-RNN, Attention, and Transformer combinations
* **Temporal Sequence Modeling** - LSTM, BiLSTM, GRU variants
* **Attention Mechanisms** - Context-aware feature weighting
* **Transformer Architectures** - Self-attention based sequence modeling
* **Cross-Dataset Generalization** - Model robustness across different data sources
* **Comparative Benchmarking** - Performance metrics and trade-offs
* **Battery Degradation Analysis** - Understanding nonlinear aging patterns
* **Predictive Maintenance** - Enabling RUL (Remaining Useful Life) assessment
* **Explainable Battery Intelligence** - Model interpretability and insights

---

## 📊 Datasets & Model Comparison Framework

### NASA Battery Dataset

The **NASA Prognostics Center of Excellence** dataset is a standard benchmark containing lithium-ion battery aging data under controlled conditions:

* **Cells**: B0005, B0006, B0007, B0018
* **Operating Conditions**: Controlled temperature and charge/discharge protocols
* **Measurements**: Voltage, current, temperature, capacity
* **Characteristics**: Well-defined degradation patterns, consistent experimental setup

### CALCE Battery Dataset

The **Center for Advanced Life Cycle Engineering (CALCE)** dataset provides complementary real-world battery aging data:

* **Operating Conditions**: Varied temperatures and discharge rates
* **Measurements**: Voltage, current, temperature, capacity
* **Characteristics**: More realistic degradation patterns, diverse operating environments

### Multi-Dataset Strategy

This project uses **dual-dataset evaluation** to assess model generalization:

| Aspect | Purpose |
|--------|---------|
| **Single-Dataset Training** | Establish baseline performance and model characteristics |
| **Cross-Dataset Evaluation** | Test generalization across different operating conditions |
| **Combined Training** | Improve robustness through diverse data sources |
| **Comparative Benchmarking** | Identify which architectures generalize best |

> Large raw datasets are not directly stored in this repository. Instructions for obtaining and preparing the datasets are provided in [data/README.md](data/README.md).

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

## 🏗️ Model Architectures

### Implemented Models

This project implements and compares 8 hybrid deep learning architectures:

#### Sequential Models (Baseline RNN Variants)
| Model | Architecture | Parameters |
|-------|--------------|-----------|
| **LSTM** | Single-direction LSTM | input_size, hidden_size=64, num_layers=2, dropout=0.1 |
| **BiLSTM** | Bidirectional LSTM | input_size, hidden_size=64, num_layers=2, dropout=0.1 |
| **GRU** | Gated Recurrent Unit | input_size, hidden_size=64, num_layers=2, dropout=0.1 |

#### Hybrid CNN-RNN Models (Local + Temporal Features)
| Model | Architecture | Components |
|-------|--------------|-----------|
| **CNN-LSTM** | Conv1D (3×1) → LSTM | CNN feature extraction + LSTM temporal modeling |
| **CNN-GRU** | Conv1D (3×1) → GRU | CNN feature extraction + GRU temporal modeling |

#### Attention-Enhanced Hybrid Models (Context-Aware)
| Model | Architecture | Mechanism |
|-------|--------------|----------|
| **CNN-BiLSTM-Attention** | Conv1D → BiLSTM → Attention | CNN features → bidirectional context → weighted attention |
| **CNN-BiGRU-Attention** | Conv1D → BiGRU → Attention | CNN features → bidirectional context → weighted attention |

#### Transformer Models (Self-Attention Based)
| Model | Architecture | Key Features |
|-------|--------------|--------------|
| **Transformer** | Multi-head self-attention encoder | Positional encoding, 2 layers, 4 heads, d_model=64 |

### Model Selection Strategy

```text
Model Complexity & Interpretability vs Performance

Simple & Fast                              Complex & Expressive
│                                          │
LSTM ─ BiLSTM ─ GRU                         │
       │                                   │
       └─ CNN-LSTM ─ CNN-GRU               │
              │                            │
              └─ CNN-BiLSTM-Attention      │
                 CNN-BiGRU-Attention       │
                       │                   │
                       └─ Transformer ─────┘
```

---

## 🎓 Comparative Analysis Framework

Each model is evaluated on:

1. **Single-Dataset Performance**
   - Training on NASA dataset
   - Training on CALCE dataset
   
2. **Cross-Dataset Generalization**
   - Train on NASA → Test on CALCE
   - Train on CALCE → Test on NASA
   
3. **Combined Training**
   - Train on both datasets together
   
4. **Performance Metrics**
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - R² Score
   - Inference time
   - Model size

### Expected Outcomes

- **Sequential Models (LSTM/BiLSTM/GRU)**: Fast, interpretable baseline
- **Hybrid CNN-RNN Models**: Improved local feature extraction
- **Attention Models**: Better context awareness and long-range dependencies
- **Transformer Models**: Superior self-attention capabilities, potentially best generalization

---

## 📂 Repository Structure

```text
Battery-Health-Deep-Learning/
│
├── README.md                    # Project overview and guide
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
│
├── configs/
│   ├── nasa.yaml               # NASA dataset configuration
│   └── calce.yaml              # CALCE dataset configuration
│
├── data/
│   ├── README.md               # Dataset preparation guide
│   └── raw/                    # Raw battery data
│       ├── B0005.mat
│       ├── B0006.mat
│       ├── B0007.mat
│       └── B0018.mat
│
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Dataset loading and preprocessing
│   ├── preprocessing.py        # Feature engineering and normalization
│   ├── train.py                # Training pipeline
│   ├── evaluate.py             # Model evaluation and benchmarking
│   │
│   └── models/                 # Model implementations
│       ├── __init__.py
│       ├── lstm.py             # LSTM Regressor
│       ├── bilstm.py           # BiLSTM Regressor
│       ├── gru.py              # GRU Regressor
│       ├── cnn_lstm.py         # CNN-LSTM Hybrid
│       ├── cnn_gru.py          # CNN-GRU Hybrid
│       ├── transformer.py      # Transformer Encoder
│       ├── cnn_bilstm_attention.py    # CNN-BiLSTM-Attention
│       └── cnn_bigru_attention.py     # CNN-BiGRU-Attention
│
├── results/
│   ├── scaler.joblib           # Fitted data scaler
│   ├── transformer_best.pt     # Best model weights
│   ├── transformer_metrics.json # Performance metrics
│   ├── transformer_predictions.csv # Model predictions
│   └── training_history.json   # Training curves
│
└── notebooks/                  # Jupyter notebooks (optional)
    ├── 01_data_exploration.ipynb
    ├── 02_preprocessing.ipynb
    └── 03_model_comparison.ipynb
```

**Model Files Summary:**
- 3 Sequential Models: LSTM, BiLSTM, GRU
- 2 Hybrid CNN-RNN Models: CNN-LSTM, CNN-GRU  
- 2 Attention-Enhanced Models: CNN-BiLSTM-Attention, CNN-BiGRU-Attention
- 1 Transformer Model: Transformer Encoder
- **Total: 8 models for comparative analysis**

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

### Train a Single Model

Train a specific model on NASA dataset:

```bash
# Train LSTM
python -m src.train --config configs/nasa.yaml --model lstm

# Train BiLSTM
python -m src.train --config configs/nasa.yaml --model bilstm

# Train Transformer
python -m src.train --config configs/nasa.yaml --model transformer

# Train CNN-BiLSTM-Attention
python -m src.train --config configs/nasa.yaml --model cnn_bilstm_attention
```

### Evaluate Trained Models

Evaluate a trained model on test data:

```bash
python -m src.evaluate --config configs/nasa.yaml --model_path results/model_best.pt
```

### Comparative Analysis

Run all models for systematic comparison:

```bash
python -m src.train --config configs/nasa.yaml --compare_all
```

Generate comparison report:

```bash
python -m src.evaluate --compare_all --output results/comparison_report.json
```

---

## 📏 Evaluation Metrics

All models are evaluated using standard regression metrics:

### Primary Metrics

**Mean Absolute Error (MAE)**
$$MAE = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

**Root Mean Squared Error (RMSE)**
$$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

**Coefficient of Determination (R²)**
$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

### Secondary Metrics

**Mean Absolute Percentage Error (MAPE)**
$$MAPE = \frac{100}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

**Inference Time** - Model prediction speed on test set

**Model Size** - Number of parameters and memory footprint

---

## 📈 Experimental Results & Benchmarks

This section will be populated with comprehensive comparative results across all models.

### Expected Comparison Table

| Model | Dataset | RMSE | MAE | R² | Gen. (CALCE) | Inf. Time |
|-------|---------|------|-----|-----|-------------|-----------|
| LSTM | NASA | — | — | — | — | — |
| BiLSTM | NASA | — | — | — | — | — |
| GRU | NASA | — | — | — | — | — |
| CNN-LSTM | NASA | — | — | — | — | — |
| CNN-GRU | NASA | — | — | — | — | — |
| CNN-BiLSTM-Attn | NASA | — | — | — | — | — |
| CNN-BiGRU-Attn | NASA | — | — | — | — | — |
| Transformer | NASA | — | — | — | — | — |

**Notes:**
- "Gen. (CALCE)" = Cross-dataset generalization (trained on NASA, tested on CALCE)
- "Inf. Time" = Average inference time per batch
- Results will be added as experiments complete

---

## 🔄 Cross-Dataset Generalization Analysis

A major research goal is evaluating model robustness across different battery datasets:

### Generalization Experiments

Systematic evaluation of generalization:

```text
Scenario 1: Single-Dataset Training
├── Train on NASA → Test on NASA (in-distribution)
├── Train on CALCE → Test on CALCE (in-distribution)

Scenario 2: Cross-Dataset Transfer
├── Train on NASA → Test on CALCE (cross-dataset)
├── Train on CALCE → Test on NASA (cross-dataset)

Scenario 3: Combined Training
└── Train on NASA + CALCE → Test on both (domain adaptation)
```

### Key Research Questions

1. Which model architecture generalizes best across datasets?
2. Does bidirectional modeling (BiLSTM, BiGRU) improve cross-dataset performance?
3. Do attention mechanisms help with domain transfer?
4. How does Transformer performance compare to hybrid CNN-RNN models?
5. What is the trade-off between model complexity and generalization?

---

## 🎯 Model Comparison Summary

| Aspect | LSTM | BiLSTM | GRU | CNN-LSTM | CNN-GRU | CNN-BiLSTM-Attn | CNN-BiGRU-Attn | Transformer |
|--------|:----:|:------:|:---:|:--------:|:-------:|:---------------:|:--------------:|:-----------:|
| Complexity | Low | Low | Low | Medium | Medium | High | High | High |
| Speed | Fast | Fast | Fast | Medium | Medium | Slow | Slow | Medium |
| Parameters | Few | Few | Few | Medium | Medium | Many | Many | Many |
| Local Features | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Bidirectional | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| Attention | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| Self-Attention | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

---

## 🔍 Explainability & Analysis

Future releases will include interpretation tools:

* **Feature Importance** - Which input features drive predictions?
* **Attention Visualization** - What time steps does the model focus on?
* **Prediction Error Analysis** - When and why do models fail?
* **Degradation Regime Analysis** - Different predictions for different battery states?
* **SHAP Values** - Shapley-based model interpretability
* **Saliency Maps** - Gradient-based input importance

The objective is to understand **why** each model predicts specific health states and identify which architectures are best suited for different operating conditions.

---

## 🔁 Reproducibility & Configuration

All experiments document and ensure:

* Random seeds and reproducibility
* Dataset splits and preprocessing parameters
* Model hyperparameters and training configuration
* Software dependencies (see requirements.txt)
* Evaluation protocols and metrics
* Configuration files in `configs/`

All results are generated from reproducible experimental pipelines.

---

## 🗺️ Development Roadmap

### Phase 1: Core Implementation ✅
- [x] LSTM implementation
- [x] BiLSTM implementation
- [x] GRU implementation
- [x] CNN-LSTM implementation
- [x] CNN-GRU implementation
- [x] CNN-BiLSTM-Attention implementation
- [x] CNN-BiGRU-Attention implementation
- [x] Transformer implementation

### Phase 2: Data & Training 🔄
- [ ] NASA dataset preprocessing
- [ ] CALCE dataset preprocessing
- [ ] Training pipeline for all models
- [ ] Model checkpointing and logging
- [ ] Hyperparameter optimization

### Phase 3: Evaluation & Comparison 📋
- [ ] Single-dataset benchmarking (NASA)
- [ ] Single-dataset benchmarking (CALCE)
- [ ] Cross-dataset generalization tests
- [ ] Performance comparison table
- [ ] Inference time benchmarking
- [ ] Model size analysis

### Phase 4: Analysis & Insights 🔍
- [ ] Attention visualization
- [ ] Feature importance analysis
- [ ] Prediction error analysis
- [ ] Cross-dataset transfer insights
- [ ] Model selection recommendations
- [ ] Comprehensive comparison report

### Phase 5: Documentation 📚
- [ ] Dataset preparation guide
- [ ] Model training guide
- [ ] Evaluation guide
- [ ] Jupyter notebooks with examples
- [ ] Hyperparameter tuning guide

---

## 🤝 Contributing

Contributions are welcome! Areas for collaboration:

* **New Model Architectures** - Implement and benchmark additional hybrid models
* **Dataset Integration** - Add support for other battery datasets
* **Optimization** - Model acceleration and quantization techniques
* **Analysis Tools** - Visualization and interpretability tools
* **Documentation** - Tutorials and usage guides
* **Bug Fixes** - Identify and fix issues

Please open an issue or pull request with your contributions.

---

## 📚 Related Research & Publications

This repository is part of comprehensive research on **AI-driven battery health diagnosis** using:

* Deep Learning & Transformer architectures
* Hybrid CNN-RNN models
* Attention mechanisms for time-series prediction
* Cross-dataset generalization and transfer learning
* Intelligent Battery Management Systems
* Energy AI and electric vehicle applications

Associated peer-reviewed publications, datasets, and reproducible experiments will be linked here with their DOI.

---

## 👨‍🔬 Author

**Walid Mchara, PhD**

AI & Data Science Researcher | Deep Learning & Battery Intelligence Specialist

**Research Areas:**
- `Deep Learning` • `Transformer Architectures` • `Battery Intelligence`
- `Time-Series Forecasting` • `Generative AI` • `Energy Systems`
- `Hybrid Model Design` • `Cross-Dataset Transfer Learning`

**Contact & Links:**
- GitHub: [walidmchara](https://github.com/walidmchara)
- Research Portfolio: [AI-Research-Portfolio](https://github.com/walidmchara/AI-Research-Portfolio)

---

## 📄 License

This project is released under the **MIT License** — see [LICENSE](LICENSE) file for details.


## 🔔 Citation & Attribution

This work is part of the [AI Research Portfolio](https://github.com/walidmchara/AI-Research-Portfolio) by Walid Mchara, PhD.

W. Mchara, M. A. Khalfa and L. Manai, "Improved Diagnosis of Lithium-Ion Battery Health in Electric Vehicles via a Hybrid Deep Learning Model Incorporating Wavelet Transform and Attention Mechanism," 2024 IEEE International Conference on Artificial Intelligence & Green Energy (ICAIGE), Yasmine Hammamet, Tunisia, 2024, pp. 1-6, doi: 10.1109/ICAIGE62696.2024.10776740.
keywords: {Wavelet transforms;Lithium-ion batteries;Deep learning;Attention mechanisms;Accuracy;Predictive models;Prediction algorithms;Convolutional neural networks;Reliability;Long short term memory;Convolutional Neural Network;attention mechanism(AM);Deep Neural Network(DNN);State of Health;lithium-ion batteries},
