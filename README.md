# Ψ-Vortex: A Physics-Informed Framework for Automated Coupling Inference and Compact Modeling in Three-Dimensional Neuromorphic Devices

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Automation](https://img.shields.io/badge/Automation-3%20of%205%20steps-brightgreen.svg)](#-automation-status)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jurjsorinliviu/PSI-Vortex)

> ⚠️ **Scope notice**
>
> This repository contains implementation code, experiments, and technical documentation accompanying a manuscript under peer review.  
> The manuscript is the authoritative source for formal claims, definitions, and evaluation.

Ψ-Vortex is an automated framework for converting raw electrical measurement data into compact, high-fidelity Verilog-A models with minimal manual tuning. By combining physics-aware initialization with information-theoretic structure discovery, it accelerates training, compresses model size, and enables latent-state inference from voltage–current data alone. All validation is performed on synthetic datasets calibrated to published device parameters; experimental validation on fabricated devices remains future work. The framework is designed for compact-model development, SPICE-compatible deployment, and virtual prototyping of complex electronic and multi-physics systems.

---

## 🎉 AUTOMATION STATUS

```mermaid
flowchart LR
    subgraph IN[" "]
        A["📊 Raw Data"]
    end
    
    subgraph P1["PHASE 1: Configuration"]
        B["🔍 Auto-Symmetry<br>Eq. 6"]
        C["🏗️ Auto-Architecture<br>Eq. 9"]
    end
    
    subgraph P2["PHASE 2: Training"]
        D["⚡ Physics Init<br>Eq. 5"]
        E["🧠 Two-Phase<br>MSE→BIC"]
    end
    
    subgraph P3["PHASE 3: Extraction"]
        F["📐 Adaptive BIC<br>Eq. 7-8"]
        G["🗜️ Distillation<br>98.6%"]
    end
    
    subgraph OUT[" "]
        H["📄 Verilog-A"]
    end
    
    A --> B --> C --> D --> E --> F --> G --> H
    
    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style B fill:#fff9c4
    style C fill:#fff9c4
    style D fill:#fff9c4
    style E fill:#fff9c4
    style F fill:#fff9c4
    style G fill:#fff9c4
```

**Note:** The framework automates three of five principal manual steps (structural topology discovery, symmetry detection, architecture configuration) within a user-defined architecture search space and BIC bandwidth. Training data specification and search-space definition remain user responsibilities. All results are from synthetic benchmarks; see the manuscript for scope and limitations.

### Automation Status (3 of 5 Steps)

| Component                  | Method                   | Status      | Performance                   |
| -------------------------- | ------------------------ | ----------- | ----------------------------- |
| **Symmetry Mask (M_sym)**  | Auto-Detection (Eq. 6)   | ✅ Automated | 1.09× expert                  |
| **Architecture (h, L, m)** | Validation-Based (Eq. 9) | ✅ Automated | **0.25× manual** (4× better!) |
| **Physics-Aware Init**     | Auto-Sym → Eq. 5         | ✅ Automated | 3.63× speedup                 |
| **Clusters (K)**           | Adaptive BIC (Eq. 7-8)   | ✅ Automated | Negative loss convergence     |
| **Matrix Rank (r*)**       | Adaptive BIC (Eq. 7-8)   | ✅ Automated | 98.6% compression             |
| **Verilog-A Generation**   | Auto-Generation          | ✅ Automated | 0.984 correlation             |

*Note: Physics-Aware Initialization (Eq. 5) automatically uses the symmetry mask from Auto-Detection (Eq. 6), making the automated steps of the training pipeline fully automatic. Training data and search-space boundaries remain user-specified.*

---

## 📋 Overview

**Ψ-Vortex** addresses three fundamental challenges in Physics-Informed Neural Networks (PINNs):

1. **Computational Bottleneck**: The prohibitive O(N²) complexity of training recurrent teachers
2. **Manual Tuning Bottleneck**: The reliance on domain expertise for structure extraction
3. **Architecture Selection**: The need for expert knowledge to choose network size

### Key Innovations

| Innovation                                          | Problem Solved                 | Performance                   |
| --------------------------------------------------- | ------------------------------ | ----------------------------- |
| **Physics-Aware Initialization** (Eq. 5)            | O(N²) computational bottleneck | 3.63× speedup                 |
| **Automatic Symmetry Detection** (Eq. 6)            | Manual symmetry specification  | 1.09× expert                  |
| **Validation-Based Architecture Selection** (Eq. 9) | Manual architecture tuning     | **0.25× manual (4× better!)** |
| **Adaptive BIC-Inspired Regularization** (Eq. 7-8)  | Manual K, r* selection         | Automated                     |
| **Two-Phase Training**                              | Speed vs. structure tradeoff   | Synergistic                   |

---

## 🚀 Headline Results

| Metric                      | Value                                |
| --------------------------- | ------------------------------------ |
| **Automation Level**        | **3 of 5 steps**                     |
| Auto-Architecture vs Manual | **0.25× MSE** (4× better!)           |
| Auto-Symmetry vs Expert     | 1.09× (within 9%)                    |
| **Convergence Speedup**     | **6.74×** (p = 2.54×10⁻¹³)           |
| MSE Improvement             | 85.3% lower (9.59×10⁻⁸ vs 6.51×10⁻⁷) |
| Effect Size (Cohen's d)     | 2.03 (very large)                    |
| Parameter Compression       | 98.6%                                |
| Memory Reduction            | 70× (353 KB → 5 KB)                  |
| Verilog-A Correlation       | 0.984                                |
| Symmetry Detection Accuracy | 100%                                 |

---

## 📊 Comprehensive Experimental Validation

We conducted **21 experiments** (original 16 + 2 architecture + 3 statistical validation) to validate all claims.

### NEW: Statistical Validation Results (20-Seed Experiments)

#### Experiment 19: Robustness Experiments (n=20 seeds)

| Metric                  | Ψ-Vortex Init         | Random Init           | p-value        |
| ----------------------- | --------------------- | --------------------- | -------------- |
| MSE (mean ± std)        | 9.59×10⁻⁸ ± 4.95×10⁻⁸ | 6.51×10⁻⁷ ± 3.84×10⁻⁷ | 2.64×10⁻⁶      |
| Epochs to target        | 65.5 ± 36.4           | 441.1 ± 83.7          | **2.54×10⁻¹³** |
| **Convergence Speedup** | **6.74×**             | -                     | -              |
| **Cohen's d**           | **2.03** (very large) | -                     | -              |

**Key Finding:** Physics-aware initialization provides statistically significant speedup across ALL random seeds!

#### Experiment 20: Noise Robustness (SNR 20-60 dB)

| SNR (dB)   | Ψ-Vortex MSE | Random Init MSE | Speedup |
| ---------- | ------------ | --------------- | ------- |
| 60 (clean) | 9.59×10⁻⁸    | 6.51×10⁻⁷       | ~7×     |
| 40         | ~1×10⁻⁷      | ~7×10⁻⁷         | ~7×     |
| 20 (noisy) | ~2×10⁻⁷      | ~1.4×10⁻⁶       | ~7×     |

**Key Finding:** Speedup remains stable at ~7× regardless of noise level!

#### Experiment 21: BIC Runtime Analysis

| Component           | Time (ms/epoch) | Percentage |
| ------------------- | --------------- | ---------- |
| Backward Pass       | 88.62           | 94.8%      |
| Forward Pass        | 2.65            | 2.8%       |
| Optimizer Step      | 1.43            | 1.5%       |
| **BIC Computation** | **0.69**        | **0.7%**   |
| Loss (MSE)          | 0.05            | 0.1%       |

**Key Finding:** BIC forward computation is negligible (0.7%). Overhead is dominated by backward pass gradient computation through O(W²) pairwise distances.

### NEW: Experiment 17 - Automatic Architecture Selection

Systematic grid evaluation consistently outperforms a single manual expert guess:

| Complexity  | Manual Config | Manual MSE | Auto Config  | Auto MSE  | Ratio     |
| ----------- | ------------- | ---------- | ------------ | --------- | --------- |
| Simple      | (h=32, L=1)   | 1.36×10⁻³  | (h=128, L=2) | 4.62×10⁻⁵ | **0.03×** |
| Medium      | (h=64, L=2)   | 5.62×10⁻³  | (h=128, L=2) | 1.33×10⁻³ | **0.24×** |
| Complex     | (h=128, L=3)  | 6.53×10⁻⁴  | (h=128, L=2) | 3.64×10⁻⁴ | **0.56×** |
| **Average** | -             | -          | -            | -         | **0.28×** |

**Key Finding:** Systematic grid evaluation achieves **0.28× MSE** on average compared to a single manual expert guess (not an exhaustive expert search).

---

### NEW: Experiment 18 - Full Pipeline Integration

Complete automation within user-specified bounds: Auto-Arch + Auto-Sym + Physics Init + BIC + Verilog-A

| Complexity  | Auto Arch    | Auto Sym | Full Auto MSE | Manual MSE | Ratio       |
| ----------- | ------------ | -------- | ------------- | ---------- | ----------- |
| Simple      | (h=128, L=2) | odd      | 6.56×10⁻⁴     | 7.42×10⁻⁴  | **0.88×** ✓ |
| Medium      | (h=128, L=2) | odd      | 2.10×10⁻³     | 5.77×10⁻³  | **0.36×** ✓ |
| Complex     | (h=128, L=2) | odd      | 9.89×10⁻⁴     | 5.96×10⁻⁴  | 1.66×       |
| **Average** | -            | -        | -             | -          | **0.97×**   |

**Key Finding:** Full automation achieves **0.97× manual MSE** with minimal human intervention!

---

### Experiment 1-16: Original Validation Suite

<details>
<summary>Click to expand all 16 original experiments</summary>

#### Experiment 1: Multi-Seed Statistical Significance (5 seeds × 4 configs)

| Configuration     | Epochs (Mean±Std) | Time (Mean±Std)   | Final Loss (Mean±Std)  |
| ----------------- | ----------------- | ----------------- | ---------------------- |
| Baseline          | 323.6 ± 60.9      | 1.41s ± 0.28s     | 9.94e-07 ± 5.5e-09     |
| Identity          | 49.0 ± 12.9       | 0.21s ± 0.05s     | 7.49e-07 ± 2.0e-07     |
| **Auto-Detected** | **49.0 ± 12.9**   | **0.21s ± 0.05s** | **7.49e-07 ± 2.0e-07** |
| Expert Manual     | 44.6 ± 19.5       | 0.19s ± 0.09s     | 8.12e-07 ± 4.8e-08     |

#### Experiment 2: Synthetic Symmetry Detection

| Symmetry Type       | Expected | Detected | Confidence | Correct |
| ------------------- | -------- | -------- | ---------- | ------- |
| Odd (f(-x) = -f(x)) | odd      | odd      | 99.94%     | ✅       |
| Even (f(-x) = f(x)) | even     | even     | 99.93%     | ✅       |
| None (asymmetric)   | none     | none     | 34.65%     | ✅       |

**Detection Accuracy: 100% (3/3)**

#### Experiment 3: Ψ-Family Lineage Comparison

| Method       | Architecture   | Initialization | Epochs | Speedup vs Ψ-HDL |
| ------------ | -------------- | -------------- | ------ | ---------------- |
| **Ψ-HDL**    | MLP (4 layers) | Random Xavier  | 121    | 1.00×            |
| **Ψ-xLSTM**  | mLSTM + sLSTM  | Random Xavier  | 193    | 0.63×            |
| **Ψ-Vortex** | mLSTM + sLSTM  | Physics-Aware  | **32** | **3.78×**        |

#### Experiment 4: Extended Ablation (2×2×2 Grid)

| Configuration       | Epochs | Time (s) | Final Loss   |
| ------------------- | ------ | -------- | ------------ |
| Random+NoBIC+NoSym  | 125    | 0.50     | 8.76e-07     |
| **Physics+BIC+Sym** | **19** | **0.11** | **9.81e-07** |

#### Experiment 5: λ_BIC and γ Sensitivity

| λ_BIC     | γ=0.05 | γ=0.1  | γ=0.2 | γ=0.5 |
| --------- | ------ | ------ | ----- | ----- |
| 0.001-0.1 | 32     | **19** | 19    | 19    |

**Hyperparameters are robust** across wide range.

#### Experiment 6: Model Size Scalability

| Hidden Size | Baseline Epochs | Ψ-Vortex Epochs | Speedup   |
| ----------- | --------------- | --------------- | --------- |
| 32          | 436             | 48              | **9.08×** |
| 64          | 193             | 70              | **2.76×** |
| 128         | 331             | 92              | **3.60×** |

#### Experiment 7: Frequency Response (10kHz - 500kHz)

| Frequency  | Ψ-Vortex Epochs   | Speedup   |
| ---------- | ----------------- | --------- |
| 10-500 kHz | **47** (constant) | 2.0-2.96× |

#### Experiment 8: Learning Rate Sensitivity

| Learning Rate | Speedup    |
| ------------- | ---------- |
| 1e-4          | **29.41×** |
| 1e-2          | 1.72×      |
| **Mean**      | **14.56×** |

#### Experiment 9: Compression vs Accuracy

| Student Hidden | Compression | vs Teacher  |
| -------------- | ----------- | ----------- |
| **16**         | **98.0%**   | **0.65×** ✅ |

#### Experiment 10: Verilog-A Accuracy

| Metric                | Value      |
| --------------------- | ---------- |
| **Model Correlation** | **0.9841** |

#### Experiment 11: Cross-Device Generalization

Generalization gap expected for device-to-device variation.

#### Experiment 12: Long Sequence Test

Speedup **maintained** for sequences up to 10,000 timesteps.

#### Experiment 13: Noise Robustness

Detection correct up to **10% noise**.

#### Experiment 14: End-to-End Pipeline

**Pipeline Status: ✅ AUTOMATED WITHIN USER-SPECIFIED BOUNDS**

#### Experiment 15: 3D Thermal Crosstalk Inference (Synthetic Data)

| Metric             | Value                         |
| ------------------ | ----------------------------- |
| Validation Loss    | **3.64×10⁻⁹**                 |
| Inferred α_thermal | 0.08 (post-hoc OLS-recovered) |

**Note**: the extracted coefficient represents an effective system-level coupling parameter for behavioral simulation; it is not claimed to be a universal material constant. All thermal inference results are from synthetic data calibrated to published device parameters. Recovery error is 6.4% at α=0.08 but degrades to 132% at α=0.05, indicating a practical detection threshold.

#### Experiment 16: Detection Threshold Sensitivity

Recommended threshold: 0.7 provides good balance.

</details>

---

## 🎯 Validation Criteria - ALL PASSED

| Criterion                      | Required     | Achieved              | Status |
| ------------------------------ | ------------ | --------------------- | ------ |
| Auto-architecture ≤ manual MSE | ≤1.5×        | **0.28×**             | ✅      |
| Auto-symmetry ≤ expert         | ≤1.3×        | **1.09×**             | ✅      |
| Auto speedup vs baseline       | ≥2.0×        | **6.74×** (p < 10⁻¹³) | ✅      |
| Statistical significance       | p < 0.05     | **p = 2.54×10⁻¹³**    | ✅      |
| Effect size (Cohen's d)        | > 0.8        | **2.03** (very large) | ✅      |
| Detection accuracy             | ≥66%         | **100%**              | ✅      |
| Valid Verilog-A output         | Yes          | **Yes**               | ✅      |
| Full pipeline MSE              | ≤1.5× manual | **0.97×**             | ✅      |

**VERDICT: ✅ AUTOMATION VALIDATED WITHIN USER-SPECIFIED BOUNDS**

---

## 🔄 Evolution: Ψ-Vortex vs Predecessors

```mermaid
flowchart TB
    subgraph evolution["Ψ-Family Evolution Timeline"]
        A["🧬 Ψ-NN<br>Structure Discovery<br>❌Manual ❌HF"]
        B["🔧 Ψ-HDL<br>Verilog-A Gen<br>❌Manual ❌HF"]
        C["📈 Ψ-xLSTM<br>High-Frequency<br>❌Manual ✅HF"]
        D["🌀 Ψ-Vortex<br>3/5 Steps Auto<br>✅Auto ✅HF"]
    end
    
    A --> B --> C --> D
    
    style A fill:#ffcdd2
    style B fill:#fff9c4
    style C fill:#c8e6c9
    style D fill:#b3e5fc,stroke:#0288d1,stroke-width:3px
```

### Complete Feature Comparison

| Capability                            | Ψ-HDL | Ψ-xLSTM | **Ψ-Vortex** |
| ------------------------------------- | :---: | :-----: | :----------: |
| Structural Interpretability           |   ✅   |    ✅    |      ✅       |
| Verilog-A Generation                  |   ✅   |    ✅    |      ✅       |
| High-Frequency Fidelity               |   ❌   |    ✅    |      ✅       |
| Efficient Training                    |   ✅   |    ❌    |      ✅       |
| Auto K, ε, r Selection                |   ❌   |    ❌    |      ✅       |
| **Auto-Symmetry Detection**           |   ❌   |    ❌    |  ✅ **NEW**   |
| **Auto-Architecture Selection**       |   ❌   |    ❌    |  ✅ **NEW**   |
| Multi-Physics Modeling                |   ❌   | Limited |      ✅       |
| **Automated Structure/Symmetry/Arch** |   ❌   |    ❌    |  ✅ **NEW**   |

### Performance Gains

| Metric     | vs Ψ-HDL                     | vs Ψ-xLSTM                   |
| ---------- | ---------------------------- | ---------------------------- |
| Speed      | 3.78× faster                 | 6.03× faster                 |
| Automation | Manual → 3/5 steps automated | Manual → 3/5 steps automated |
| Accuracy   | Same                         | Same                         |

---

## 📁 Repository Structure

```
PSI-Vortex
├── Core Library Modules
│   ├── core_psi_xlstm.py           # PSI-xLSTM with matrix memory (Eq. 3)
│   ├── core_rrad_loss.py           # RRAD distillation loss (Eq. 4)
│   ├── core_physics_init.py        # Physics-aware initialization (Eq. 5)
│   ├── core_auto_symmetry.py       # Automatic symmetry detection (Eq. 6) ✨NEW
│   ├── core_adaptive_bic.py        # Differentiable BIC-inspired regularizer (Eq. 7-8)
│   ├── core_auto_architecture.py   # Validation-based architecture (Eq. 9) ✨NEW
│   └── core_vortex_trainer.py      # Two-phase training orchestration
│
├── Runnable Experiments (21 total)
│   ├── 01_plot_3d_results.py
│   ├── 02_psi-vortex_speed_benchmark.py
│   ├── 03_psi_vortex_experiment.py
│   ├── 04_psi-vortex_ablation.py
│   ├── 05_psi-vortex_compression_analysis.py
│   ├── 06_auto_symmetry_experiment.py
│   ├── 07_comprehensive_auto_symmetry_validation.py
│   ├── 08_end_to_end_pipeline.py
│   ├── 09_extended_experiments.py
│   ├── 10_final_experiments.py
│   ├── 11_robustness_experiments.py          ✨NEW (20-seed validation)
│   ├── 12_scalability_experiments.py         ✨NEW (BIC runtime analysis)
│   └── 13_numerically_stable_bic.py          ✨NEW (gradient stability)
│
├── Generated Results (CSV + PNG)
│   ├── robustness_experiments.png             ✨NEW (statistical validation)
│   ├── scalability_experiments.png            ✨NEW (BIC overhead)
│   ├── bic_gradient_analysis.png              ✨NEW (gradient stability)
│   ├── bic_bandwidth_sensitivity.png          ✨NEW (h parameter)
│   ├── auto_arch_vs_manual.csv
│   ├── auto_arch_multi_seed.csv
│   ├── full_auto_pipeline.csv
│   ├── full_automation_status.csv
│   └── ... (14 more CSV files)
│
├── Generated Verilog-A Models
│   ├── psi_vortex_3d_thermal.va
│   ├── psi_vortex_memristor_auto.va
│   └── psi_vortex_thermal_auto.va
│
└── Data Files
    ├── printed_memristor_training_data.csv
    └── 3d_thermal_crosstalk_data.csv
```

---

## 🔧 Installation & Quick Start

### 🚀 Option 1: GitHub Codespaces (Recommended - Zero Setup!)

The fastest way to get started - runs entirely in your browser with everything pre-configured:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jurjsorinliviu/PSI-Vortex)

**Steps:**
1. Click the badge above or go to the repository and click "Code" → "Codespaces" → "Create codespace on main"
2. Wait ~2 minutes for the environment to build
3. Run experiments directly in the terminal!

**What's Included:**
- ✅ Python 3.10 with PyTorch pre-installed
- ✅ All dependencies automatically configured
- ✅ Jupyter notebook support
- ✅ VS Code extensions for Python development
- ✅ Ready to run all 21 experiments immediately

### 💻 Option 2: Local Installation

#### Requirements

```bash
pip install torch>=2.0.0 numpy pandas matplotlib scipy
```

#### Quick Start

```bash
git clone https://github.com/jurjsorinliviu/PSI-Vortex.git
cd PSI-Vortex

# Run full automation validation (recommended first test)
python 12_auto_architecture_experiment.py
```

### Expected Output

```
🎉 AUTO-ARCHITECTURE SELECTION: FULLY VALIDATED
   Network architecture is now AUTOMATED within user-specified bounds!

AUTOMATION STATUS:
    Component           Method      Status Validation
Symmetry Mask   Auto-Detection ✓ Automated    Exp 6-7
 Architecture Validation-Based ✓ Automated     Exp 12
 Clusters (K)     Adaptive BIC ✓ Automated     Eq 7-8
    Rank (r*)     Adaptive BIC ✓ Automated     Eq 7-8
    Verilog-A  Auto-Generation ✓ Automated      Exp 8

======================================================================
Ψ-VORTEX: 3 OF 5 MANUAL STEPS AUTOMATED
======================================================================
```

---

## 🔄 Two-Phase Training Strategy

```mermaid
flowchart LR
    subgraph phase1["Phase 1: Rapid Convergence"]
        A["Physics-Aware<br>Init Eq.5+6"] --> B["MSE<br>Optimization"] --> C["6.74×<br>Speedup"]
    end
    
    subgraph phase2["Phase 2: Structure Refinement"]
        D["Adaptive BIC<br>Eq.7-8"] --> E["Entropy<br>Minimization"] --> F["Loss: -14.8<br>to -16.6"]
    end
    
    C --> D
    
    style A fill:#e3f2fd
    style C fill:#c8e6c9
    style F fill:#c8e6c9
```

| Phase       | Duration    | Method                          | Result                             |
| ----------- | ----------- | ------------------------------- | ---------------------------------- |
| **Phase 1** | ~65 epochs  | Physics-Aware Init (Eq. 5+6)    | **6.74× speedup** (p = 2.54×10⁻¹³) |
| **Phase 2** | +100 epochs | Adaptive BIC-Inspired (Eq. 7-8) | Negative loss convergence          |

**Statistically Validated Benefits (n=20 seeds):**
- Ψ-Vortex Init: 65.5 ± 36.4 epochs to target
- Random Init: 441.1 ± 83.7 epochs to target
- **Speedup**: 6.74× with Cohen's d = 2.03 (very large effect)
- **Combined**: 1.36× baseline speedup + optimal structure in 0.72s

---

## 🔍 Automatic Symmetry Detection

```mermaid
flowchart TD
    A["📊 Input Data<br>X, Y pairs"] --> B{"🔬 Analyze<br>Symmetry"}
    
    B --> C["Method 1:<br>Direct Pairing"]
    B --> D["Method 2:<br>Interpolation"]
    B --> E["Method 3:<br>Correlation"]
    
    C --> F["🗳️ Ensemble<br>Voting"]
    D --> F
    E --> F
    
    F --> G{"📋 Decision<br>Threshold: 0.7"}
    G -->|"C_odd > 0.7"| H["🔴 Odd Symmetry<br>f(-x) = -f(x)"]
    G -->|"C_even > 0.7"| I["🔵 Even Symmetry<br>f(-x) = f(x)"]
    G -->|"else"| J["⚪ No Symmetry<br>Identity Init"]
    
    H --> K["M_sym<br>Symmetry Mask"]
    I --> K
    J --> K
    
    K --> L["⚡ Physics-Aware<br>Init Eq. 5"]
    
    style H fill:#ffcdd2
    style I fill:#bbdefb
    style J fill:#f5f5f5
    style L fill:#c8e6c9
```

---

## 🧮 Mathematical Framework

### Equation 3: Matrix Memory Update (mLSTM)
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)
```

### Equation 4: RRAD Loss
```
L_RRAD = ||f_S(x) - f_T(x)||² + β·||∂h_S/∂t - ∂h_T/∂t||²
```

### Equation 5: Physics-Aware Initialization
```
θ_Vortex = M_sym ⊙ W_orth + ε·N(0,σ²)
```

### Equation 6: Automatic Symmetry Detection ✨NEW
```
M_sym = AutoDetect(X, Y) = argmax_{g∈{odd,even,none}} C_g(X, Y)

where:
  C_odd = Corr(Y(X), -Y(-X))
  C_even = Corr(Y(X), Y(-X))
```

### Equations 7-8: Differentiable BIC-Inspired Regularizer
```
L_Vortex = L_RRAD + λ_BIC · R_BIC(θ_S)

R_BIC(θ) = (log(n)/2n) · Σ 1/(Σ exp(-(w_i - w_j)²/h²))
```

*Note: No formal model-selection consistency guarantee of classical BIC is claimed for this differentiable surrogate.*

### Equation 9: Validation-Based Architecture Selection ✨NEW
```
a* = argmin_{a ∈ A} MSE_val(a)

where A = {(h, L, m) : h ∈ {16,32,64,128}, L ∈ {1,2,3}, m ∈ {8,16,32}}
```

*Note: This is a grid evaluation over 36 bounded candidates, not a neural architecture search method.*

---

## 📈 Generated Verilog-A Model

The framework automatically generates thermal-aware compact models:

```verilog
// Ψ-Vortex Auto-Generated Verilog-A Compact Model
// Architecture: AUTO-SELECTED (h=128, L=2, m=32)
// Symmetry: AUTO-DETECTED (odd, 99.9% confidence)
// Compression: AUTO-BIC (K=5, r*=4)
module psi_vortex_auto(p, n);
    inout p, n;
    electrical p, n;
    
    // Extracted Physics Parameters (Ψ-Vortex BIC-inspired optimization, post-hoc OLS recovery)
    parameter real r_off = 1.352479e+06;
    parameter real r_on = 9.156440e+02;
    parameter real alpha = 1.000000e+00;
    
    analog begin
        real V_in;
        V_in = V(p, n);
        
        // ODD SYMMETRY: I(-V) = -I(V)
        I(p, n) <+ V_in / r_off * (1 + 0.1 * sinh(alpha * V_in));
    end
endmodule
```

---

## 🌡️ 3D Thermal Crosstalk Inference (Synthetic Data)

```mermaid
flowchart TB
    subgraph stack["1000-Layer 3D Stack"]
        A["🔴 Driver Layer n<br>Active Switching"]
        B["📄 Paper Substrate<br>100μm thickness"]
        C["🔵 Victim Layer n+1<br>Passive Read"]
    end
    
    A -->|"Joule<br>Heating"| B
    B -->|"Thermal<br>Coupling"| C
    
    subgraph inference["Ψ-Vortex Inference Process"]
        D["📊 Observe<br>I-V Only"] --> E["🧠 Learn Latent<br>Thermal State"]
        E --> F["📐 Extract<br>α_thermal"]
        F --> G["📄 Generate<br>Verilog-A"]
    end
    
    C --> D
    
    G --> H["🔌 thermal_pin<br>Multi-Physics Port"]
    
    style A fill:#ffcdd2
    style C fill:#bbdefb
    style H fill:#c8e6c9
```

**Inference Results (Synthetic Data):**
| Metric             | Value                         |
| ------------------ | ----------------------------- |
| Inferred α_thermal | 0.08 (post-hoc OLS-recovered) |
| Validation Loss    | 3.64×10⁻⁹                     |
| Output             | thermal_pin for simulation    |

**Result:** Validation loss **3.64×10⁻⁹** on synthetic data without explicit temperature measurements. Recovery degrades at weak coupling (α=0.05, 132% error), bounding the practical detection regime.

---

## 🔗 Related Work

| Framework    | Repository                                                   | Automation Level        |
| ------------ | ------------------------------------------------------------ | ----------------------- |
| Ψ-NN         | [github.com/ZitiLiu/Psi-NN](https://github.com/ZitiLiu/Psi-NN) | Manual                  |
| Ψ-HDL        | [github.com/jurjsorinliviu/Psi-HDL](https://github.com/jurjsorinliviu/Psi-HDL) | Manual                  |
| Ψ-xLSTM      | [github.com/jurjsorinliviu/Psi-xLSTM](https://github.com/jurjsorinliviu/Psi-xLSTM) | Manual                  |
| **Ψ-Vortex** | This repository                                              | **3/5 steps automated** |

---

## 📄 Citation

```bibtex
@misc{jurj_psivortex_2025,
  title        = {Ψ-Vortex: A Physics-Informed Framework for Automated Coupling Inference and Compact Modeling in Three-Dimensional Neuromorphic Devices},
  author       = {Jurj, Sorin Liviu},
  year         = {2026},
  note         = {Manuscript submitted},
  howpublished = {GitHub repository}
}
```

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file.

---

## 🏗️ Architecture Selection Process

```mermaid
flowchart TD
    A["🎯 Define Search<br>Space A"] --> B["📋 Candidate<br>Architectures"]
    
    B --> C1["(h=16, L=1)"]
    B --> C2["(h=32, L=1)"]
    B --> C3["..."]
    B --> C4["(h=128, L=3)"]
    
    C1 --> D["🏋️ Train Each<br>Configuration"]
    C2 --> D
    C3 --> D
    C4 --> D
    
    D --> E["📊 Evaluate<br>MSE_val"]
    
    E --> F{"🏆 Select<br>Best"}
    F --> G["a* = argmin<br>MSE_val(a)<br>Eq. 9"]
    
    G --> H["✅ Optimal<br>h=128, L=2"]
    
    style A fill:#e3f2fd
    style G fill:#fff9c4
    style H fill:#c8e6c9
```

**Result:** Systematic grid evaluation achieves **0.25× MSE** compared to a single manual expert guess (not an exhaustive expert search).

---

## 🏆 Summary

**Ψ-Vortex is a physics structure-informed neural network framework that automates structural topology discovery, symmetry detection, and architecture selection:**

- ✅ **Reduced domain expertise required** (search space and training data remain user-specified)
- ✅ **Outperforms single manual expert guess by 4×** (0.25× MSE)
- ✅ **6.74× convergence speedup** (statistically validated, p = 2.54×10⁻¹³)
- ✅ **85.3% MSE improvement** over random initialization
- ✅ **3 of 5 manual steps automated** from raw data to Verilog-A
- ✅ **Validated across 21 comprehensive experiments** (including 20-seed statistical validation)
- ✅ **All validation criteria passed**
- ⚠️ **All validation on synthetic data** — fabricated-device validation remains future work

```mermaid
flowchart LR
    A[📊 Data] --> B[🤖 Ψ-Vortex] --> C[📄 Verilog-A]
    
    style A fill:#e1f5fe
    style B fill:#fff9c4,stroke:#ff9800,stroke-width:3px
    style C fill:#c8e6c9
```



## ❓ Quick FAQ

**Is Ψ-Vortex a new neural network architecture?**  
No. Ψ-Vortex is a framework built on top of xLSTM that adds physics-aware initialization, automated structure discovery, and Verilog-A generation for engineering deployment.

**How is Ψ-Vortex different from Ψ-xLSTM?**  
Ψ-xLSTM demonstrated high-frequency device modeling, but still required manual tuning and expert intervention. Ψ-Vortex adds automated symmetry detection, architecture selection, accelerated convergence, and deployable structure extraction.

**Does Ψ-Vortex require prior device-physics knowledge?**  
No explicit domain knowledge is required for the automated steps (symmetry detection, architecture selection, structural topology). However, the user must specify training data, the architecture search space, and the BIC bandwidth. These inputs define the bounds within which automation operates.

**What does Ψ-Vortex output?**  
It generates compact, SPICE-compatible Verilog-A models suitable for simulation, compact-model development, and virtual prototyping.

**Can Ψ-Vortex infer latent physical effects?**  
Yes, provided those effects leave a measurable signature in the observed data above the detection threshold. In the demonstrated case study on synthetic data, Ψ-Vortex inferred a latent state consistent with inter-layer thermal coupling using only voltage–current measurements. Recovery degrades for weak coupling signals (α ≤ 0.05). Experimental validation on fabricated devices remains future work.

**When should Ψ-Vortex not be used?**  
If a system is purely static, low-frequency, or already well described by a simple analytical model, Ψ-Vortex may be unnecessary. It is most useful when dynamics are complex, multi-timescale, or partially unobservable.

**Where does Ψ-Vortex fit into real EDA workflows?**  
Ψ-Vortex is designed for compact-model development and SPICE-based simulation workflows. Its main value is reducing manual effort and accelerating virtual prototyping before fabrication.

**Where can I read the full conceptual background?**  
See the accompanying [technical rationale document](psi-vortex-technical-rationale.pdf), which provides the design lineage, architectural motivations, and scope boundaries of the framework. Note: This document provides architectural and historical context for the Ψ-Vortex framework. It complements, but does not replace, the peer-reviewed manuscript. I will upload it as soon as the manuscript gets published. Thank you for your understanding and patience.
