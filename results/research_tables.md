# Experimental Benchmark & Research Comparative Study

**Generated on:** 2026-09-02 08:09:20 UTC
**Evaluation Mode:** Strict 6-Model Benchmark Suite (M1–M6)
**Dataset:** Real Photographic Covers (500 images, BOSSbase/DIV2K subset, Seed: 42)

---

## Table 1: Literature Comparison & Model Characteristics

| Model Code | Model & Paper Citation | Venue / Year | Backbone Architecture | Feature Extraction / Attention | Parameter Count | Model Size (MB) | FLOPs | Checkpoint Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M1** | Proposed Model 1: LF-RINN + Adaptive EMD-OPAP | Proposed (2026) | 4-Block Invertible Haar-DWT Wavelet Coupling Network | Dual-Path (Wavelet Subband Residuals + Multi-scale Edge CNN Branch) | 72,833 (0.073M) | 0.29 MB | 0.18 GFLOPs / 512x512 image | `AVAILABLE` |
| **M2** | Proposed Model 2: CostMapCNN + Adaptive EMD-OPAP | Proposed (2026) | 5-layer 2D Convolutional Feed-Forward Network | Multi-scale spatial convolutions with 3x3 receptive fields | 28,193 (0.028M) | 0.11 MB | 0.07 GFLOPs / 512x512 image | `AVAILABLE` |
| **M3** | Paper 1 Model: Joint CNN (Iqbal et al. 2026) | Sci. Rep. (2026) | Encoder-Decoder Convolutional Architecture + KeyMixer Module | Multi-layer 3x3 Conv + BatchNorm + LeakyReLU | 1,824,000 (1.82M) | 7.3 MB | 4.60 GFLOPs / 512x512 image | `MISSING` |
| **M4** | Paper 2 Model: CycleGAN Adversarial Steg (Abdollahi et al. 2023) | JISA (2023) | Three-Player CycleGAN (Generator G, Extractor F, Steganalyzer D, Hiding Net H) | ResNet-based 9 residual blocks with instance normalization | 11,400,000 (11.4M) | 45.6 MB | 28.40 GFLOPs / 512x512 image | `MISSING` |
| **M5** | Paper 3 Model: Block Prep Net (Dabhade et al. 2026) | MTAP (2026) | Block Preparation Network + Spatial Hiding CNN + Reveal CNN | Block-wise 8x8 feature tokenization & depthwise separable convolutions | 2,450,000 (2.45M) | 9.8 MB | 6.20 GFLOPs / 512x512 image | `MISSING` |
| **M6** | Lower Baseline Model: Sequential Naive LSB | Classical Baseline | None (Direct Pixel Bit Manipulation) | None (Uniform sequential spatial scanning) | 0 (0.0M) | 0.0 MB | 0.00 GFLOPs | `AVAILABLE` |

---

## Table 2: Quantitative Benchmark Results & Comparative Evaluation

| Model Code | Model Name | Category | Status / Source | PSNR (dB) ↑ | SSIM ↑ | MSE ↓ | BPP (Cap) ↑ | Stego Det. Rate ↓ | Security Score ↑ | BER / Ext. Acc. (%) ↑ | Latency (ms) ↓ | Params |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M1** | Proposed Model 1: LF-RINN + Adaptive EMD-OPAP | Proposed | `EXPERIMENTAL` | **68.84** | **0.9998** | 0.0085 | 0.52 | **0.048** | **95.2%** | 100.0% | 14.8 ms | 72,833 (0.073M) |
| **M2** | Proposed Model 2: CostMapCNN + Adaptive EMD-OPAP | Proposed | `EXPERIMENTAL` | 62.45 | 0.9985 | 0.0371 | 0.50 | 0.092 | 90.8% | 100.0% | 8.4 ms | 28,193 (0.028M) |
| **M3** | Paper 1 Model: Joint CNN (Iqbal et al. 2026) | Existing | `PUBLISHED` | 38.40 | 0.9410 | 9.3800 | 0.40 | 0.240 | 76.0% | 98.4% | 42.5 ms | 1,824,000 (1.82M) |
| **M4** | Paper 2 Model: CycleGAN Adversarial Steg (Abdollahi et al. 2023) | Existing | `PUBLISHED` | 36.20 | 0.9180 | 15.5600 | 0.35 | 0.310 | 69.0% | 96.8% | 88.0 ms | 11,400,000 (11.4M) |
| **M5** | Paper 3 Model: Block Prep Net (Dabhade et al. 2026) | Existing | `PUBLISHED` | 39.80 | 0.9530 | 6.7900 | 0.45 | 0.190 | 81.0% | 98.9% | 34.2 ms | 2,450,000 (2.45M) |
| **M6** | Lower Baseline Model: Sequential Naive LSB | Baseline | `EXPERIMENTAL` | 51.14 | 0.9912 | 0.4990 | 0.33 | 0.582 | 41.8% | 100.0% | 1.2 ms | 0 (0.0M) |

> **Note on Data Sources:** `EXPERIMENTAL` denotes live measured inference on identical test images using trained checkpoints. `PUBLISHED` indicates published literature figures from respective original peer-reviewed papers due to unreleased proprietary model checkpoints.

---

## Table 3: Architectural Advantage & Ablative Comparison

| Comparison Dimension | Proposed Model 1 (LF-RINN) | Proposed Model 2 (CostMapCNN) | Paper 1 (Joint CNN) | Paper 2 (CycleGAN) | Paper 3 (Block Prep Net) | Lower Baseline (LSB) | Key Scientific Advantage of Proposed Model |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wavelet Subband Isolation** | Haar-DWT (LL, LH, HL, HH) Invertible Decomposition | None (Spatial Only) | None | None | None | None | Isolates high-energy low-frequency bands from embedding distortions, confining changes to high-entropy residual subbands. |
| **Cost-Map Driven Adaptive Zoning** | Neural Invertible Cost Map (Tri-Zone A/B/C) | 5-Layer CNN Cost Map | Uniform KeyMixer | Adversarial Latent | 8x8 Spatial Blocks | Uniform 1-bit | Selects minimal-distortion pixel coordinates adaptively, preventing statistical signature clustering in smooth regions. |
| **Embedding Mechanics** | EMD (Base-5, n=2) + Multi-bit OPAP | EMD (Base-5) + OPAP | End-to-End Decoder | GAN Generator Cycle | CNN Spatial Reveal | Direct LSB Bit Flip | EMD embeds 2.32 bits per pixel pair with at most +/-1 modification on a single pixel, yielding peak PSNR (>68 dB). |
| **Model Efficiency & Parameter Footprint** | 72.8K Params (0.29 MB) | 28.2K Params (0.11 MB) | 1.82M Params (7.3 MB) | 11.4M Params (45.6 MB) | 2.45M Params (9.8 MB) | 0 Params (0.00 MB) | Over 25x-150x smaller parameter footprint than existing GAN and deep CNN steganography architectures, enabling real-time edge execution. |
| **Steganalysis Resistance (SRM/Xu-Net)** | 4.8% Detection Rate | 9.2% Detection Rate | 24.0% Detection Rate | 31.0% Detection Rate | 19.0% Detection Rate | 58.2% Detection Rate | Tri-zone high-frequency placement preserves first- and second-order spatial pixel statistics, achieving state-of-the-art security score (95.2%). |

---

## Table 4: Overall Multi-Criteria Ranking & Scientific Summary

| Rank | Model Code | Model Name | Category | Quality Score (35%) | Security Score (35%) | Capacity Score (15%) | Efficiency Score (15%) | Composite Score | Overall Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **#1** | **M1** | Proposed Model 1: LF-RINN + Adaptive EMD-OPAP | Proposed | 98.83 | 95.2 | 52.0 | 59.04 | **84.57** | State-of-the-Art Overall Champion (Highest Quality & Security) |
| **#2** | **M2** | Proposed Model 2: CostMapCNN + Adaptive EMD-OPAP | Proposed | 92.41 | 90.8 | 50.0 | 67.65 | **81.77** | Strong Lightweight Proposed Baseline |
| **#3** | **M5** | Paper 3 Model: Block Prep Net (Dabhade et al. 2026) | Existing | 68.39 | 81.0 | 45.0 | 46.31 | **65.98** | Competitive Existing Deep Literature Baseline |
| **#4** | **M6** | Lower Baseline Model: Sequential Naive LSB | Baseline | 80.88 | 41.8 | 33.0 | 97.23 | **62.47** | Existing Literature Model (Moderate Security / Higher Complexity) |
| **#5** | **M3** | Paper 1 Model: Joint CNN (Iqbal et al. 2026) | Existing | 66.63 | 76.0 | 40.0 | 43.01 | **62.37** | Existing Literature Model (Moderate Security / Higher Complexity) |
| **#6** | **M4** | Paper 2 Model: CycleGAN Adversarial Steg (Abdollahi et al. 2023) | Existing | 63.74 | 69.0 | 35.0 | 31.94 | **56.5** | Vulnerable Classical Lower Baseline |

---
