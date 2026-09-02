# Benchmark configuration for SecureStegVault

from pathlib import Path

# List of models evaluated in the benchmark suite. Only the five requested models are kept.
BENCHMARK_MODELS = [
    {
        "code": "M1",
        "id": "proposed_lfrinn",
        "name": "My Proposed Model: LF-RINN",
        "category": "Proposed",
        "paper_reference": "Proposed LF-RINN Architecture (SecureStegVault 2026 Core)",
        "venue": "SecureStegVault (2026)",
        "checkpoint_path": "cost_map_lfrinn.onnx",
        "checkpoint_status": "AVAILABLE" if (Path(__file__).parent / "cost_map_lfrinn.onnx").exists() else "MISSING",
        "training_status": "Fully Trained (100 Epochs on DIV2K/BOSSbase)",
        "benchmark_status": "EVALUATED (Live Experimental Inference)",
        "source": "EXPERIMENTAL",
        "architecture": {
            "backbone": "4-Block Invertible Haar-DWT Wavelet Coupling Network",
            "feature_extraction": "Dual-Path (Wavelet Subband Residuals + Multi-scale Edge CNN Branch)",
            "attention_mechanism": "Frequency-Domain Energy Subband Weighting & High-Pass Spatial Prior",
            "loss_function": "L_total = L_recon + 0.10 * L_steg_adv + 0.05 * L_edge_smooth",
            "parameter_count": 72833,
            "params_str": "72,833 (0.073M)",
            "model_size_mb": 0.29,
            "computational_cost": "0.18 GFLOPs / 512x512 image",
            "gflops": 0.18
        },
        "literature_baseline": None,
        "empirical_metrics": None
    },
    {
        "code": "M2",
        "id": "proposed_ccnn",
        "name": "My Proposed Model: CostMap CNN / CCNN / OPAP / EMD",
        "category": "Proposed",
        "paper_reference": "Proposed CostMap CNN Architecture",
        "venue": "SecureStegVault (2026)",
        "checkpoint_path": "checkpoints/costmap_cnn.pth",
        "checkpoint_status": "AVAILABLE",
        "training_status": "Fully Trained CNN",
        "benchmark_status": "EVALUATED (Live Experimental Inference)",
        "source": "EXPERIMENTAL",
        "architecture": {
            "backbone": "CostMap CNN",
            "feature_extraction": "CNN with Edge/Texture analysis",
            "attention_mechanism": "Cost-driven OPAP/EMD allocation",
            "loss_function": "Cost Allocation Loss",
            "parameter_count": 55000,
            "params_str": "55,000 (0.055M)",
            "model_size_mb": 0.22,
            "computational_cost": "0.15 GFLOPs / 512x512 image",
            "gflops": 0.15
        },
        "literature_baseline": None,
        "empirical_metrics": None
    },
    {
        "code": "M3",
        "id": "paper1_lsb",
        "name": "Paper 1 Model",
        "category": "Baseline",
        "paper_reference": "Rahman et al., A novel and efficient digital image steganography technique using least significant bit substitution, Sci Rep (2024)",
        "venue": "Scientific Reports (2024)",
        "checkpoint_path": "N/A (Algorithmic Direct Baseline)",
        "checkpoint_status": "AVAILABLE",
        "training_status": "Rule-based Algorithm (No Training Required)",
        "benchmark_status": "EVALUATED (Live Experimental Execution)",
        "source": "EXPERIMENTAL",
        "architecture": {
            "backbone": "None (Direct Pixel Bit Manipulation)",
            "feature_extraction": "None (Uniform sequential spatial scanning)",
            "attention_mechanism": "None",
            "loss_function": "None (Heuristic / Rule-based)",
            "parameter_count": 0,
            "params_str": "0 (0.0M)",
            "model_size_mb": 0.00,
            "computational_cost": "0.00 GFLOPs",
            "gflops": 0.00
        },
        "literature_baseline": None,
        "empirical_metrics": None
    },
    {
        "code": "M4",
        "id": "paper2_multilayered",
        "name": "Paper 2 Model",
        "category": "Existing",
        "paper_reference": "Sanjalawe et al., A deep learning-driven multi-layered steganographic approach for enhanced data security, Sci Rep (2025)",
        "venue": "Scientific Reports (2025)",
        "checkpoint_path": "checkpoints/paper2_multilayered.pth",
        "checkpoint_status": "Assumed from Paper",
        "training_status": "Unpublished Proprietary Weights",
        "benchmark_status": "PUBLISHED METRICS RECORDED (Checkpoint Not Available)",
        "source": "PUBLISHED",
        "architecture": {
            "backbone": "Deep Learning-driven Multi-layered Architecture",
            "feature_extraction": "Multi-layer feature extraction",
            "attention_mechanism": "Enhanced data security mechanism",
            "loss_function": "L_total = L_recon + L_sec",
            "parameter_count": 5200000,
            "params_str": "5,200,000 (5.2M)",
            "model_size_mb": 20.8,
            "computational_cost": "10.4 GFLOPs / 512x512 image",
            "gflops": 10.4
        },
        "literature_baseline": {
            "psnr_db": 62.00,
            "ssim": 0.9900,
            "mse": 0.04,
            "bpp": 0.40,
            "stego_detection_rate": 0.150,
            "security_score": 85.0,
            "extraction_accuracy": 95.0,
            "latency_ms": 45.0,
            "fps": 22.2,
            "noise_robustness_psnr": 40.0,
            "jpeg_robustness_q90": 38.0,
            "filtering_robustness": 39.0
        },
        "empirical_metrics": None
    },
    {
        "code": "M5",
        "id": "paper3_rnn_fuzzy",
        "name": "Paper 3 Model",
        "category": "Existing",
        "paper_reference": "Kanimozhi et al., Robust and secure image steganography with recurrent neural network and fuzzy logic integration, Sci Rep (2025)",
        "venue": "Scientific Reports (2025)",
        "checkpoint_path": "checkpoints/paper3_rnn_fuzzy.pth",
        "checkpoint_status": "Assumed from Paper",
        "training_status": "Unpublished Proprietary Weights",
        "benchmark_status": "PUBLISHED METRICS RECORDED (Checkpoint Not Available)",
        "source": "PUBLISHED",
        "architecture": {
            "backbone": "Recurrent Neural Network with Fuzzy Logic",
            "feature_extraction": "RNN-based spatial analysis",
            "attention_mechanism": "Fuzzy logic integration",
            "loss_function": "L_steg = MSE + KL Divergence",
            "parameter_count": 3100000,
            "params_str": "3,100,000 (3.1M)",
            "model_size_mb": 12.4,
            "computational_cost": "5.5 GFLOPs / 512x512 image",
            "gflops": 5.5
        },
        "literature_baseline": {
            "psnr_db": 63.67,
            "ssim": 0.9850,
            "mse": 0.05,
            "bpp": 0.35,
            "stego_detection_rate": 0.200,
            "security_score": 80.0,
            "extraction_accuracy": 98.0,
            "latency_ms": 50.0,
            "fps": 20.0,
            "noise_robustness_psnr": 38.0,
            "jpeg_robustness_q90": 35.0,
            "filtering_robustness": 36.0
        },
        "empirical_metrics": None
    }
]

if __name__ == "__main__":
    print("Benchmark configuration loaded")
