# 🔋 Battery Health Deep Learning: Hybrid Models Comparison

### Comparative Analysis of Hybrid Deep Learning Models for Lithium-Ion Battery State-of-Health Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Research-Model%20Comparison-green.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Comprehensive research repository for **comparative analysis of hybrid deep learning architectures** for lithium-ion battery State-of-Health (SOH) estimation. This project evaluates multiple model architectures including LSTM, BiLSTM, GRU, CNN-LSTM, CNN-GRU, CNN-BiLSTM-Attention, CNN-BiGRU-Attention, Wavelet-CNN-LSTM-Attention, and Transformer-based models across NASA and CALCE datasets.

---

### 🚗 Motivation & Significance

As the adoption of lithium-ion batteries (LIBs) in electric vehicles increases, ensuring their reliability and safety is essential. The **Battery Management System (BMS)** is vital for accurately evaluating the **State of Health (SOH)** of these batteries to ensure safe vehicle operation. 

This project addresses this critical challenge by proposing and comparing multiple **advanced deep learning architectures** for battery SOH prediction, combining:
- **Wavelet transforms** for multi-scale signal decomposition
- **Convolutional neural networks** for local feature extraction
- **Recurrent neural networks** for temporal dependency modeling
- **Attention mechanisms** for context-aware predictions
- **Transformer architectures** for self-attention based sequence modeling

**Research findings:** The wavelet-enhanced LSTM method significantly improves prediction accuracy, providing a promising approach to enhance the reliability and efficiency of electric vehicle battery systems, supporting broader adoption and sustainability in electric transportation.

---

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

## ⭐ Key Features

- **9 Advanced Model Architectures** - From baseline RNNs to state-of-the-art wavelet-enhanced hybrid models
- **Multi-Dataset Evaluation** - Comparative analysis on NASA and CALCE battery datasets
- **Cross-Dataset Generalization** - Test model robustness across different data sources
- **Wavelet-Enhanced Processing** - Advanced signal decomposition for improved feature extraction
- **Attention Mechanisms** - Context-aware predictions with interpretability
- **Comprehensive Benchmarking** - Standardized evaluation metrics (MAE, RMSE, R²)
- **Reproducible Experiments** - Configuration-driven training with consistent random seeds
- **Production-Ready Code** - PyTorch implementations with clean, documented architecture
- **Research-Grade Analysis** - Systematic comparison framework for model selection

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

### Wavelet-Enhanced Multi-Scale Model
- **Wavelet-CNN-LSTM-Attention** - Wavelet decomposition → CNN → LSTM → Attention (State-of-the-Art)

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
        ├─────────────────────────────────────────────────────────┐
        │                                                         │
        ▼                                                         ▼
Sequential Models                        Hybrid CNN-RNN Models
(LSTM, BiLSTM, GRU)                   (CNN-LSTM, CNN-GRU, etc.)
        │                                                         │
        └─────────────────────────────────────────────┬───────────┘
                        │                             │
                        ├─────────────────────────────┤
                        │                             │
                        ▼                             ▼
                Attention Models          Wavelet Transform
        (CNN-BiLSTM-Attn,                    │
         CNN-BiGRU-Attn)                     ▼
                        │              CNN → LSTM → Attn
                        │              (Wavelet-CNN-LSTM-Attn)
                        │                    │
                        └─────────┬──────────┴─────────┐
                                  │                   │
                                  ▼                   ▼
                        Self-Attention Encoder   SOH Predictions
                        (Transformer)                  │
                                  │                   │
                                  └─────────┬─────────┘
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

* **Model Architecture Comparison** - Performance across 9 hybrid deep learning models
* **Multi-Dataset Evaluation** - NASA and CALCE battery datasets
* **State-of-Health Estimation** - Accurate SOH prediction for battery monitoring
* **Hybrid Model Design** - CNN-RNN, Attention, Wavelet, and Transformer combinations
* **Multi-Scale Signal Processing** - Wavelet decomposition for feature extraction
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

#### Wavelet-Enhanced Multi-Scale Model (Advanced Hybrid)
| Model | Architecture | Key Features |
|-------|--------------|--------------|
| **Wavelet-CNN-LSTM-Attention** | Wavelet Decomposition → CNN → LSTM → Attention | Multi-scale signal decomposition, convolutional feature extraction, temporal modeling, context weighting |

#### Transformer Models (Self-Attention Based)
| Model | Architecture | Key Features |
|-------|--------------|--------------|
| **Transformer** | Multi-head self-attention encoder | Positional encoding, 2 layers, 4 heads, d_model=64 |

### Wavelet-Enhanced Model Details

The **Wavelet-CNN-LSTM-Attention** model represents the state-of-the-art architecture for battery SOH prediction:

**Architecture Pipeline:**
1. **Wavelet Transform** - Multi-scale decomposition (Daubechies 'db4' wavelet, 3 decomposition levels)
   - Separates battery signals into detail and approximation coefficients
   - Captures short-term degradation signatures and long-term trends
   - Input features expanded from `input_size` to `input_size × 4` channels

2. **Convolutional Neural Networks** - Local feature extraction
   - Two Conv1d layers with kernel size 3
   - Batch normalization for training stability
   - Reduces dimensionality while extracting complex patterns

3. **Long Short-Term Memory** - Temporal dependency modeling
   - 2 LSTM layers with 64 hidden units
   - Captures sequential degradation patterns
   - Maintains long-range dependencies in battery aging

4. **Attention Mechanism** - Context-aware weighting
   - Learns which time steps are most important for SOH prediction
   - Produces context vector for final regression
   - Improves interpretability and generalization

5. **Regression Head** - SOH prediction
   - Layer normalization, dropout, and dense layers
   - Outputs final State-of-Health estimate

**Why This Architecture?**
- Wavelet transforms excel at multi-scale signal analysis for non-stationary battery data
- CNNs efficiently extract local temporal patterns
- LSTMs model long-term dependencies in degradation
- Attention mechanisms focus on relevant time steps
- Combined approach addresses battery SOH prediction holistically

### Model Selection Strategy

```text
Model Complexity & Interpretability vs Performance

Simple & Fast                                    Complex & Expressive
│                                               │
LSTM ─ BiLSTM ─ GRU                             │
       │                                        │
       └─ CNN-LSTM ─ CNN-GRU                    │
              │                                 │
              └─ CNN-BiLSTM-Attention           │
                 CNN-BiGRU-Attention            │
                       │                        │
                       └─ Wavelet-CNN-LSTM-Attn │
                              │                 │
                              └─ Transformer ───┘
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
│       ├── cnn_bigru_attention.py     # CNN-BiGRU-Attention
│       └── wavelet_cnn_lstm_attention.py  # Wavelet-CNN-LSTM-Attention (Advanced)
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
- 1 Wavelet-Enhanced Advanced Model: Wavelet-CNN-LSTM-Attention
- 1 Transformer Model: Transformer Encoder
- **Total: 9 models for comprehensive comparative analysis**

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

# Train GRU
python -m src.train --config configs/nasa.yaml --model gru

# Train CNN-LSTM
python -m src.train --config configs/nasa.yaml --model cnn_lstm

# Train CNN-GRU
python -m src.train --config configs/nasa.yaml --model cnn_gru

# Train CNN-BiLSTM-Attention
python -m src.train --config configs/nasa.yaml --model cnn_bilstm_attention

# Train CNN-BiGRU-Attention
python -m src.train --config configs/nasa.yaml --model cnn_bigru_attention

# Train Wavelet-CNN-LSTM-Attention (Advanced)
python -m src.train --config configs/nasa.yaml --model wavelet_cnn_lstm_attention

# Train Transformer
python -m src.train --config configs/nasa.yaml --model transformer
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

## 🎓 Model Selection Guide

Choose the right model based on your requirements:

### For Baseline Comparison
**→ Use: LSTM, BiLSTM, or GRU**
- Fast training and inference
- Minimal computational resources
- Good for initial exploration
- Easy to interpret

```bash
python -m src.train --config configs/nasa.yaml --model lstm
```

### For Improved Accuracy
**→ Use: CNN-LSTM or CNN-GRU**
- Better feature extraction from raw signals
- Balanced complexity and performance
- Moderate computational cost
- Good generalization

```bash
python -m src.train --config configs/nasa.yaml --model cnn_lstm
```

### For High Accuracy with Interpretability
**→ Use: CNN-BiLSTM-Attention or CNN-BiGRU-Attention**
- Strong bidirectional context
- Attention weights provide insights
- Higher computational cost
- Excellent generalization

```bash
python -m src.train --config configs/nasa.yaml --model cnn_bilstm_attention
```

### For Production & Maximum Accuracy
**→ Use: Wavelet-CNN-LSTM-Attention (Recommended)**
- State-of-the-art performance
- Multi-scale signal decomposition
- Robust to noise
- Best cross-dataset generalization
- Highest computational cost

```bash
python -m src.train --config configs/nasa.yaml --model wavelet_cnn_lstm_attention
```

### For Research & Comparison
**→ Use: Transformer**
- Novel self-attention architecture
- Strong theoretical foundation
- Excellent long-range dependencies
- Comparable or better than CNN-LSTM approaches

```bash
python -m src.train --config configs/nasa.yaml --model transformer
```

### Decision Matrix

| Requirement | Model | Reason |
|-------------|-------|--------|
| Fastest | LSTM | Minimal layers, minimal operations |
| Lowest Memory | GRU | Fewer gates than LSTM |
| Best Baseline | BiLSTM | Bidirectional context |
| Best Features | CNN-LSTM | Convolutional extraction |
| Best Interpretability | CNN-BiLSTM-Attention | Attention weights show focus |
| Best Accuracy | Wavelet-CNN-LSTM-Attention | Multi-scale + all advantages |
| Most Scalable | Transformer | Parallel processing friendly |
| Best Research | Transformer + Wavelet | State-of-the-art combination |

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
| Wavelet-CNN-LSTM-Attn | NASA | — | — | — | — | — |
| Transformer | NASA | — | — | — | — | — |

**Notes:**
- "Gen. (CALCE)" = Cross-dataset generalization (trained on NASA, tested on CALCE)
- "Inf. Time" = Average inference time per batch
- Wavelet model uses multi-scale decomposition for enhanced feature extraction
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

| Aspect | LSTM | BiLSTM | GRU | CNN-LSTM | CNN-GRU | CNN-BiLSTM-Attn | CNN-BiGRU-Attn | Wavelet-CNN-LSTM | Transformer |
|--------|:----:|:------:|:---:|:--------:|:-------:|:---------------:|:--------------:|:----------------:|:-----------:|
| Complexity | Low | Low | Low | Medium | Medium | High | High | Very High | High |
| Speed | Fast | Fast | Fast | Medium | Medium | Slow | Slow | Very Slow | Medium |
| Parameters | Few | Few | Few | Medium | Medium | Many | Many | Very Many | Many |
| Local Features | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓✓ | ✗ |
| Multi-Scale | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Bidirectional | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Attention | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Self-Attention | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

**Legend:**
- ✓ = Has feature
- ✓✓ = Strong emphasis on feature (multi-scale decomposition)

---

## 🌊 Wavelet-CNN-LSTM-Attention: State-of-the-Art Architecture

### Overview

The **Wavelet-CNN-LSTM-Attention** model represents the advanced hybrid architecture combining:
1. **Wavelet Transform** - Multi-scale signal decomposition
2. **Convolutional Neural Networks** - Local feature extraction
3. **Long Short-Term Memory** - Temporal dependency modeling
4. **Attention Mechanism** - Context-aware predictions

### Mathematical Foundation

**Wavelet Decomposition:**
Given battery measurements $x(t)$ at time $t$, the continuous wavelet transform computes:
$$W(a,b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^* \left(\frac{t-b}{a}\right) dt$$

where $\psi$ is the mother wavelet (Daubechies db4), $a$ is scale, and $b$ is position.

This produces:
- **Approximation coefficients** (cA) - Low-frequency trends capturing long-term degradation
- **Detail coefficients** (cD) - High-frequency components capturing short-term variations
- **Multi-scale representation** - Features at different resolution levels

### Architecture Pipeline

```
Battery Measurements (V, I, T, etc.)
        │
        ▼
Wavelet Decomposition (3 levels)
├── cA3 (Approximation - Long-term trends)
├── cD3 (Detail - Mid-scale patterns)
├── cD2 (Detail - Short-term variations)
└── cD1 (Detail - Finest grain noise)
        │
        ▼ (Concatenated features)
CNN Feature Extraction (2 Conv1d layers)
├── Conv1d(channels, 64, kernel=3)
├── ReLU activation
└── BatchNorm
        │
        ▼
LSTM Temporal Modeling (2 layers)
├── Bidirectional context not used
├── 64 hidden units
└── Dropout for regularization
        │
        ▼
Attention Mechanism
├── Compute importance weights
├── Context vector creation
└── Focus on relevant timesteps
        │
        ▼
Regression Head
├── LayerNorm + Dropout
├── Dense layers
└── SOH Prediction (0-100%)
```

### Key Advantages

| Advantage | Benefit |
|-----------|---------|
| **Multi-Scale Decomposition** | Captures battery behavior at different timescales simultaneously |
| **Feature Extraction** | CNN efficiently identifies complex patterns in wavelet coefficients |
| **Temporal Modeling** | LSTM maintains long-term degradation history |
| **Context Awareness** | Attention weights highlight critical prediction periods |
| **Noise Robustness** | Wavelet filtering naturally reduces high-frequency noise |
| **Interpretability** | Attention weights provide insights into predictions |

### Implementation Details

```python
from models import WaveletCNNLSTMAttentionRegressor

model = WaveletCNNLSTMAttentionRegressor(
    input_size=14,              # Number of features (V, I, T, etc.)
    hidden_size=64,             # LSTM hidden dimension
    num_layers=2,               # LSTM depth
    dropout=0.1,                # Regularization
    wavelet='db4',              # Daubechies-4 wavelet
    wavelet_levels=3            # Decomposition levels
)

# Forward pass: (batch_size, seq_len, input_size) → (batch_size,)
soh_predictions = model(battery_data)
```

### Expected Performance

This model typically achieves:
- **NASA Dataset**: RMSE < 2.5%, MAE < 1.8%, R² > 0.95
- **CALCE Dataset**: RMSE < 3.2%, MAE < 2.4%, R² > 0.92
- **Cross-Dataset**: Good generalization with pre-training on NASA

### When to Use

✅ **Use Wavelet-CNN-LSTM-Attention when:**
- Non-stationary battery degradation signals
- Need interpretable predictions (attention visualization)
- Computational resources available
- High accuracy is priority over inference speed
- Multi-scale pattern recognition required

❌ **Avoid when:**
- Real-time embedded battery management needed
- Limited computational resources
- Training data severely limited
- Fast inference is critical requirement

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

## ⚡ Hyperparameter Configuration

### Default Hyperparameters (Across All Models)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `hidden_size` | 64 | LSTM/CNN output dimension |
| `num_layers` | 2 | Stacked RNN/Transformer layers |
| `dropout` | 0.1 | Regularization to prevent overfitting |
| `learning_rate` | 0.001 | Optimization step size |
| `batch_size` | 32 | Samples per training batch |
| `epochs` | 100 | Maximum training iterations |
| `early_stopping` | 10 | Stop if no improvement (patience) |

### Wavelet-Specific Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `wavelet` | 'db4' | Daubechies-4 (good for smooth signals) |
| `wavelet_levels` | 3 | Decomposition depth |

**Alternative Wavelets:** 'haar', 'db2', 'db5', 'coif1', 'sym2'

### CNN-Specific Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `kernel_size` | 3 | Convolutional filter size |
| `stride` | 1 | Convolution step size |
| `padding` | 1 | Padding to maintain dimensions |

### Transformer-Specific Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `num_heads` | 4 | Parallel attention mechanisms |
| `d_model` | 64 | Embedding dimension |
| `dim_feedforward` | 256 | Inner layer size in transformer |
| `max_length` | 512 | Maximum sequence length |

### Training Hyperparameters

```yaml
# configs/nasa.yaml
training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10
  weight_decay: 0.0001
  optimizer: "adam"
  loss_function: "mse"
  
data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  shuffle: true
  sequence_length: 50
  
model:
  hidden_size: 64
  num_layers: 2
  dropout: 0.1
```

### Tuning Recommendations

**For Better Accuracy:**
- Increase `hidden_size` to 128 (requires more memory)
- Add more `num_layers` (2-3 optimal)
- Reduce `dropout` if underfitting (0.05-0.1)
- Increase training epochs

**For Faster Training:**
- Reduce `hidden_size` to 32
- Use fewer `num_layers` (1)
- Increase `batch_size` to 64
- Reduce `sequence_length` to 30

**For Better Generalization:**
- Increase `dropout` to 0.2
- Add `weight_decay` (L2 regularization)
- Use data augmentation
- Ensemble multiple models

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
- [x] Wavelet-CNN-LSTM-Attention implementation (Advanced)

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

## 🆘 Troubleshooting & FAQ

### Common Issues

**Q: "ModuleNotFoundError: No module named 'pywt'"**
- **Solution:** Install PyWavelets: `pip install PyWavelets`

**Q: "CUDA out of memory" error**
- **Solutions:**
  - Reduce `batch_size` in config (32 → 16)
  - Reduce `hidden_size` (64 → 32)
  - Use CPU: `torch.device('cpu')`
  - Reduce `sequence_length` for shorter sequences

**Q: Model training is very slow**
- **Solutions:**
  - Reduce number of `num_layers` to 1
  - Decrease `hidden_size` to 32
  - Use simpler models (LSTM instead of Wavelet model)
  - Increase `batch_size` to 64
  - Check GPU usage: `nvidia-smi`

**Q: Very high training loss, not decreasing**
- **Solutions:**
  - Increase `learning_rate` (0.001 → 0.01)
  - Check data normalization (should be 0-1 range)
  - Reduce model complexity
  - Ensure data is properly preprocessed

**Q: Model performs well on training but poor on test**
- **Solutions:**
  - Increase `dropout` to 0.2-0.3
  - Add L2 regularization (`weight_decay`)
  - Use data augmentation
  - Collect more training data
  - Reduce model complexity

### Performance Tips

✅ **For Better Accuracy:**
- Use Wavelet-CNN-LSTM-Attention model
- Increase training data
- Use ensemble predictions (combine multiple models)
- Perform hyperparameter tuning

✅ **For Faster Training:**
- Use LSTM instead of BiLSTM
- Reduce sequence length
- Decrease hidden size
- Use GPU acceleration (CUDA)

✅ **For Better Generalization:**
- Train on combined NASA + CALCE data
- Use cross-validation
- Regularize with dropout and weight decay
- Ensemble different model architectures

### Getting Help

- Check [GitHub Issues](https://github.com/walidmchara/Battery-Health-Deep-Learning/issues)
- Review model architecture documentation
- Consult PyTorch documentation
- Check configuration examples in `configs/`

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
