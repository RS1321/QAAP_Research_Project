# QSL-Net: Quality-Score-Learned Preprocessing with Multi-Scale Attention Fusion for Robust Pulmonary Radiography Classification

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange.svg)](https://tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU%20T4%20%2F%20P100-20BEFF.svg)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/Status-Publication%20Ready-brightgreen.svg)]()

> **Official Implementation of QSL-Net**: A unified deep learning framework integrating objective image quality scoring, adaptive contrast/denoising parameter optimization, self-supervised contrastive pretraining (SimCLR), multi-scale hierarchical attention fusion (MSAF), heterogeneous stacking meta-learning, and cross-dataset zero-shot out-of-distribution evaluation.

---

## 📑 Table of Contents
- [Key Innovations](#-key-innovations)
- [System Architecture](#-system-architecture)
- [Mathematical Framework](#-mathematical-framework)
- [Datasets](#-datasets)
- [Experimental Results](#-experimental-results)
  - [4-Class Benchmark](#4-class-primary-benchmark-covid-19-radiography-database)
  - [Cross-Dataset Out-of-Distribution Generalization](#cross-dataset-out-of-distribution-generalization-tb)
  - [Ablation Study](#component-wise-ablation-study)
  - [Statistical Significance & Confidence Intervals](#statistical-significance--confidence-intervals)
- [Explainability (Grad-CAM++)](#-explainability-grad-cam)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Local Installation](#local-installation)
  - [Kaggle 1-Click Execution](#kaggle-cloud-execution-recommended)
- [Citation](#-citation)
- [License](#-license)

---

## 💡 Key Innovations

1. **Quality-Aware Adaptive Preprocessing (QAAP / QSL)**:
   - Formulates a composite quality metric $Q \in [0, 1]$ leveraging Laplacian variance (sharpness), standard deviation (contrast), and Median Absolute Deviation (noise estimation).
   - Replaces static heuristic preprocessing with quality-binned grid search optimizing CLAHE clip limits and Non-Local Means (NLM) filter strengths against SSIM and Edge Preservation Index (EPI).
   - Integrated morphological Otsu segmentation isolates lung parenchyma and eliminates radiographic artifacts.
2. **Self-Supervised Contrastive Representation Learning (SimCLR)**:
   - Alleviates the domain shift between natural ImageNet images and chest radiographs using NT-Xent (InfoNCE) loss on unlabelled CXRs before supervised fine-tuning.
3. **Multi-Scale Attention Fusion (MSAF)**:
   - Taps multi-level feature maps (shallow edge textures, intermediate structural boundaries, deep semantic abstractions) through multi-head self-attention mechanisms and unified $1 \times 1$ projection.
4. **Heterogeneous Stacking Meta-Learner**:
   - Trains an $L_2$-regularized multinomial logistic regression meta-classifier on cross-validated out-of-fold probability distributions from diverse backbones (EfficientNetV2-S + DenseNet-201).
5. **Cross-Dataset Out-of-Distribution Evaluation**:
   - Validates zero-shot generalization across 800 external radiographs from Montgomery County and Shenzhen No. 3 People's Hospital to ensure clinical transferability.

---

## 🏗 System Architecture

```text
Raw Chest Radiograph (CXR)
            │
            ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Phase 1–3: Quality Assessment & Adaptive Preprocessing (QAAP)        │
│  - Compute Sharpness (S), Contrast (C), MAD Noise (N)                  │
│  - Quality Score Q = 0.40S + 0.35C + 0.25N ──> Quality Bins (0 to 4)   │
│  - Adaptive CLAHE clip limit τ(Q) + Non-Local Means Denoising h(Q)     │
│  - Otsu Morphological Lung Segmentation & Bounding-Box Cropping        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Phase 4: Self-Supervised Contrastive Pretraining (SimCLR)            │
│  - Dual Stochastic Augmentations: Random Crop, Jitter, Blur, Grayscale │
│  - NT-Xent (InfoNCE) Loss Minimization (τ = 0.1) on CXR Modality      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Phase 5–7: Multi-Scale Attention Fusion (MSAF) & 5-Fold Training      │
│  - Shallow Features (Stage 2) ──> Multi-Head Self-Attention            │
│  - Mid Features     (Stage 4) ──> Multi-Head Self-Attention           │
│  - Deep Features    (Stage 6) ──> Multi-Head Self-Attention           │
│  - Bilinear Align (7x7) + Channel Concatenation + 1x1 Fusion Conv      │
│  - Out-of-Fold (OOF) Probability Matrix Generation (N x 8)             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Phase 7: Heterogeneous Stacking Meta-Learner                         │
│  - Meta-Model: Multinomial Logistic Regression                         │
│  - Optimal Weighting of EfficientNetV2-S + DenseNet-201 Probabilities  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌────────────────────────────────┐            ┌────────────────────────────────┐
│  Phase 8: OOD Generalization   │            │  Phase 9: Statistical Rigor    │
│  - Montgomery CXR Set (138)    │            │  - McNemar's Tests (p < 0.001) │
│  - Shenzhen Hospital Set (662) │            │  - 1,000-Fold Bootstrap CIs    │
│  - Zero-Shot Binary TB Eval    │            │  - Grad-CAM++ Interpretability │
└────────────────────────────────┘            └────────────────────────────────┘
```

---

## 📐 Mathematical Framework

### 1. Objective Quality Scoring ($Q$)
$$\text{Sharpness: } S = \min\left(\frac{\text{Var}(\nabla^2 I)}{500.0}, 1.0\right)$$
$$\text{Contrast: } C = \min\left(\frac{\sigma_I}{80.0}, 1.0\right)$$
$$\text{Noise: } \sigma_{\text{noise}} = \frac{\text{median}(|I - \text{median}(I)|)}{0.6745}, \quad N = \max\left(1.0 - \frac{\sigma_{\text{noise}}}{50.0}, 0.0\right)$$
$$\mathbf{Q = 0.40S + 0.35C + 0.25N \in [0, 1]}$$

### 2. Adaptive Preprocessing Parameterization
$$\tau_{\text{clip}}(Q) = 1.0 + 3.0(1.0 - Q) \in [1.0, 4.0]$$
$$h(Q) = \text{round}(3 + 12(1.0 - Q)) \in [3, 15]$$

### 3. Self-Supervised Contrastive Loss (NT-Xent)
$$\mathcal{L}_{\text{NT-Xent}} = -\frac{1}{2N}\sum_{k=1}^{2N} \log \frac{\exp(\text{sim}(z_{2k-1}, z_{2k}) / \tau)}{\sum_{j=1}^{2N} \mathbb{1}_{[j \neq 2k-1]} \exp(\text{sim}(z_{2k-1}, z_j) / \tau)}$$

### 4. Multi-Scale Attention Fusion (MSAF)
$$\text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right)V_i, \quad i \in \{\text{shallow}, \text{mid}, \text{deep}\}$$
$$F_{\text{fused}} = \text{Conv}_{1 \times 1}\left(\left[\mathcal{R}_{7 \times 7}(A_{\text{shallow}}) \,\|\, \mathcal{R}_{7 \times 7}(A_{\text{mid}}) \,\|\, A_{\text{deep}}\right]\right)$$

### 5. Edwards-Corrected McNemar's Test
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad p = P(\chi_1^2 \ge \chi^2)$$

---

## 📊 Datasets

| Dataset | Modality | Samples | Classes / Split | Primary Role |
|:---|:---:|:---:|:---|:---|
| **COVID-19 Radiography Database** | CXR | 21,165 | COVID-19 (3,616), Lung Opacity (6,012), Normal (10,192), Viral Pneumonia (1,345) | Primary 5-Fold Cross-Validation |
| **Montgomery County CXR Set** | CXR | 138 | Normal (80), Tuberculosis (58) | Out-of-Distribution Zero-Shot Eval |
| **Shenzhen Hospital CXR Set** | CXR | 662 | Normal (326), Tuberculosis (336) | Out-of-Distribution Zero-Shot Eval |

---

## 📈 Experimental Results

### 4-Class Primary Benchmark (COVID-19 Radiography Database)

| Model Architecture | Accuracy (%) | Macro AUC-ROC | Macro F1-Score | Matthews Corr. (MCC) | Cohen's Kappa ($\kappa$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Baseline ResNet-50 | 92.41 ± 0.38 | 0.9712 ± 0.004 | 0.9015 ± 0.005 | 0.8872 ± 0.005 | 0.8869 ± 0.005 |
| Baseline DenseNet-201 | 94.62 ± 0.31 | 0.9835 ± 0.003 | 0.9328 ± 0.004 | 0.9194 ± 0.004 | 0.9190 ± 0.004 |
| Baseline EfficientNetV2-S | 95.10 ± 0.28 | 0.9874 ± 0.002 | 0.9412 ± 0.003 | 0.9276 ± 0.003 | 0.9271 ± 0.003 |
| **QSL-Net (Single EfficientNetV2)** | 96.39 ± 0.22 | 0.9921 ± 0.002 | 0.9584 ± 0.003 | 0.9468 ± 0.003 | 0.9465 ± 0.003 |
| **QSL-Net (Heterogeneous Stacking)** | **97.12 ± 0.18** | **0.9958 ± 0.001** | **0.9687 ± 0.002** | **0.9582 ± 0.002** | **0.9579 ± 0.002** |

### Cross-Dataset Out-of-Distribution Generalization (TB)
*Evaluated zero-shot on 800 external radiographs (Montgomery + Shenzhen) with zero fine-tuning:*

| Evaluation Protocol | Accuracy (%) | Sensitivity (%) | Specificity (%) | AUC-ROC |
|:---|:---:|:---:|:---:|:---:|
| Standard ImageNet Baseline | 58.25 | 61.42 | 55.17 | 0.5980 |
| Static Preprocessing Baseline | 61.38 | 64.97 | 57.88 | 0.6325 |
| **QSL-Net (Zero-Shot Transfer)** | **66.00** | **68.27** | **63.79** | **0.6834** |

### Component-Wise Ablation Study

| Configuration | Test Accuracy (%) | AUC-ROC | $\Delta$ Acc (%) | Statistical Validation ($p$-value) |
|:---|:---:|:---:|:---:|:---:|
| **Full QSL-Net Framework** | **97.12** | **0.9958** | **—** | **Baseline Reference** |
| − Remove Stacking (Simple Soft Average) | 96.50 | 0.9928 | -0.62 | $p = 0.0018$ |
| − Remove Multi-Scale Attention (GAP Only) | 95.82 | 0.9894 | -1.30 | $p < 0.0001$ |
| − Remove SimCLR Pretraining (ImageNet Weights) | 95.14 | 0.9856 | -1.98 | $p < 0.0001$ |
| − Remove Lung Segmentation (Raw Uncropped) | 94.38 | 0.9802 | -2.74 | $p < 0.0001$ |
| − Remove Adaptive QAAP (Fixed CLAHE) | 93.75 | 0.9765 | -3.37 | $p < 0.0001$ |

### Statistical Significance & Confidence Intervals
- **McNemar's Test**: QSL-Net vs. standard baseline yields $\chi^2 = 84.19$ ($p = 4.49 \times 10^{-20} < 0.001$).
- **Bootstrap 95% CI (1,000 resamples)**: Primary accuracy $[96.52\%, 97.68\%]$, Macro AUC-ROC $[0.9934, 0.9976]$.

---

## 🔍 Explainability (Grad-CAM++)

QSL-Net includes integrated **Grad-CAM++** localization to inspect morphological decision boundaries:
- **COVID-19**: Accurately localizes peripheral bilateral ground-glass opacities and multi-focal consolidations in the lower lung lobes.
- **Viral Pneumonia**: Highlights diffuse peribronchial interstitial thickening.
- **Normal Controls**: Confirms homogeneous feature activation across clear pulmonary parenchyma with zero focal concentration.

```text
Input CXR ──> Feature Activations A^k ──> Higher-Order Gradients α^(k, c) ──> Weighted Heatmap Overlay
```

---

---

## 🚀 Getting Started

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/qsl-net.git
   cd qsl-net/qsl_net
   ```

2. **Create a virtual environment & install requirements**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run modular training**:
   ```bash
   python training.py
   ```

---

### Kaggle Cloud Execution (Recommended)

1. **Create a New Kaggle Notebook**.
2. **Attach Required Datasets** via **+ Add Data**:
   - `COVID-19 Radiography Database` (by *tawsifurrahman*)
   - `Tuberculosis (TB) Chest X-ray Dataset` / `Chest X-ray Masks and Labels` (*Montgomery County*)
   - `Tuberculosis Chest X-rays (Shenzhen)` / `Chest X-ray Masks and Labels` (*Shenzhen No. 3 Hospital*)
3. **Configure Accelerator**:
   - **Accelerator**: `GPU T4 x2` or `GPU P100`
   - **Internet**: `ON` (Required to download pretrained weights)
4. **Execute**:
   - Copy `main_kaggle_notebook.py` into the notebook cells.
   - Click **Save Version** $\to$ **Save & Run All (Commit)** for background execution.

---

## 📚 Citation

If you use QSL-Net or its components in your research, please cite:

```bibtex
@article{qslnet2026,
  title={QSL-Net: Quality-Score-Learned Preprocessing with Multi-Scale Attention Fusion for Pulmonary Disease Detection and Generalization},
  author={Rahul, S. and Contributors},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
