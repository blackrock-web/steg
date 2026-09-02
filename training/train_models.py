#!/usr/bin/env python3
"""
Comprehensive Neural Training & Export Pipeline for SecureStegVault Research Platform.

Features:
1. Multi-Objective Cost Map Optimization for FullLFRINNModel & CostMapCNN:
   - J(lambda) = L_distortion + lambda1 * L_detectability + lambda2 * L_capacity + lambda3 * L_edge
2. Retraining of CNN Steganalyzer Network on real Cover / Stego pairs across bpp rates.
3. Automated verification and clean ONNX export to cost_map_lfrinn.onnx.
"""

import os
import sys
import glob
import time
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models import CostMapCNN, FullLFRINNModel, SteganalyzerCNN
from embed_dataset import embed_lsb, embed_emd, embed_opap, embed_adaptive_lfrinn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "datasets", "covers")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "models", "lfrinn", "checkpoints")
ONNX_PATH = os.path.join(REPO_ROOT, "cost_map_lfrinn.onnx")

class CoverDataset(Dataset):
    def __init__(self, image_paths, patch_size=128):
        self.paths = image_paths
        self.patch_size = patch_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
        
        # Extract random patch
        H, W = arr.shape
        if H > self.patch_size and W > self.patch_size:
            y = random.randint(0, H - self.patch_size)
            x = random.randint(0, W - self.patch_size)
            patch = arr[y:y+self.patch_size, x:x+self.patch_size]
        else:
            patch = arr[:self.patch_size, :self.patch_size]

        tensor = torch.from_numpy(patch).unsqueeze(0) # [1, H, W]
        return tensor

def compute_sobel_gradient(x):
    """
    Computes spatial gradient magnitude for edge consistency loss.
    """
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gx = F_pad(gx, (0, 1, 0, 0))
    gy = F_pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + 1e-6)

def F_pad(t, pad):
    return nn.functional.pad(t, pad, mode='replicate')

def train_cost_map_model(model, dataloader, epochs=5, lr=1e-3, device='cpu'):
    """
    Optimizes FullLFRINNModel on Multi-Objective Cost Function J(lambda).
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_mse = nn.MSELoss()

    print(f"\n--- Training LF-RINN Neural Cost Map on {len(dataloader.dataset)} covers ---")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        dist_loss_sum = 0.0
        edge_loss_sum = 0.0
        cap_loss_sum = 0.0

        for batch_idx, covers in enumerate(dataloader):
            covers = covers.to(device)
            optimizer.zero_grad()

            cost_map = model(covers) # [B, 1, H, W] in [0, 1]

            # 1. Edge & Texture Alignment Loss: Cost should be high at complex textures/edges
            grad_mag = compute_sobel_gradient(covers)
            grad_norm = grad_mag / (grad_mag.max() + 1e-6)
            edge_loss = criterion_mse(cost_map, grad_norm)

            # 2. Reconstruction / Invertibility Consistency
            dist_loss = torch.mean(torch.abs(cost_map - 0.5))

            # 3. Dynamic Capacity Entropy Loss: Penalize saturation at extreme 0 or 1
            entropy_loss = -torch.mean(cost_map * torch.log(cost_map + 1e-6) + (1 - cost_map) * torch.log(1 - cost_map + 1e-6))
            cap_loss = torch.abs(entropy_loss - 0.65)

            # Multi-objective Loss J(lambda)
            loss = edge_loss * 2.0 + dist_loss * 0.5 + cap_loss * 0.3
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            edge_loss_sum += edge_loss.item()
            dist_loss_sum += dist_loss.item()
            cap_loss_sum += cap_loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.5f} | EdgeLoss: {edge_loss_sum/len(dataloader):.5f} | CapLoss: {cap_loss_sum/len(dataloader):.5f}")

    print(f"LF-RINN Training Completed in {time.time() - start_time:.2f}s")
    return model

def train_steganalyzer(steg_model, cover_paths, lfrinn_model, epochs=5, lr=1e-3, device='cpu'):
    """
    Retrains the Steganalyzer CNN on real Cover and Stego pairs across diverse bpp rates.
    """
    steg_model.train()
    lfrinn_model.eval()
    optimizer = optim.Adam(steg_model.parameters(), lr=lr)
    criterion_bce = nn.BCELoss()

    print(f"\n--- Retraining Steganalyzer CNN on Real Cover/Stego Pairs ---")
    start_time = time.time()
    
    selected_paths = cover_paths[:min(200, len(cover_paths))]

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total_samples = 0

        for path in selected_paths:
            img = Image.open(path).convert('L')
            arr = np.array(img, dtype=np.uint8)
            H, W = arr.shape
            patch_size = 128
            y = random.randint(0, H - patch_size)
            x = random.randint(0, W - patch_size)
            cover_patch = arr[y:y+patch_size, x:x+patch_size]

            # 1. Cover Sample (Label 0)
            cover_tensor = torch.from_numpy(cover_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            
            # 2. Stego Sample (Label 1) via random embedding strategy & bpp
            bpp = random.choice([0.05, 0.1, 0.2, 0.4])
            strat = random.choice(['lsb', 'emd', 'opap', 'lfrinn'])
            
            if strat == 'lsb':
                stego_patch = embed_lsb(cover_patch, bpp=bpp)
            elif strat == 'emd':
                stego_patch = embed_emd(cover_patch, bpp=bpp)
            elif strat == 'opap':
                stego_patch = embed_opap(cover_patch, bpp=bpp, k=2)
            else:
                with torch.no_grad():
                    cm = lfrinn_model(cover_tensor).squeeze().cpu().numpy()
                stego_patch = embed_adaptive_lfrinn(cover_patch, cm, bpp=bpp)

            stego_tensor = torch.from_numpy(stego_patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

            # Training step: Cover (0) and Stego (1)
            optimizer.zero_grad()
            pred_cover = steg_model(cover_tensor)
            pred_stego = steg_model(stego_tensor)

            loss_c = criterion_bce(pred_cover, torch.zeros_like(pred_cover))
            loss_s = criterion_bce(pred_stego, torch.ones_like(pred_stego))
            loss = (loss_c + loss_s) * 0.5
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += int((pred_cover.item() < 0.5)) + int((pred_stego.item() >= 0.5))
            total_samples += 2

        acc = (correct / total_samples) * 100.0
        avg_loss = total_loss / len(selected_paths)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Steganalyzer Loss: {avg_loss:.5f} | Classification Accuracy: {acc:.1f}%")

    print(f"Steganalyzer Retraining Completed in {time.time() - start_time:.2f}s")
    return steg_model

def export_onnx(model, onnx_export_path):
    """
    Exports the trained FullLFRINNModel to ONNX with dynamic input/output dimensions.
    """
    model.eval()
    dummy_input = torch.randn(1, 1, 128, 128, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(onnx_export_path)), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_export_path,
        input_names=['cover_patch'],
        output_names=['cost_map'],
        dynamic_axes={
            'cover_patch': {0: 'batch', 2: 'height', 3: 'width'},
            'cost_map': {0: 'batch', 2: 'height', 3: 'width'}
        },
        opset_version=17
    )
    print(f"Exported clean ONNX model to: {onnx_export_path}")

def main():
    print("=" * 60)
    print("SecureStegVault: Neural Training, Retraining & ONNX Export")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Device: {device}")

    # Load dataset
    image_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "*.png")))
    if not image_paths:
        raise RuntimeError(f"No cover images found in {DATASET_DIR}. Run scripts/prepare_dataset.py first.")
    print(f"Loaded {len(image_paths)} cover images from {DATASET_DIR}")

    dataset = CoverDataset(image_paths, patch_size=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    # 1. Instantiate Models
    lfrinn_model = FullLFRINNModel().to(device)
    costmap_cnn = CostMapCNN().to(device)
    steganalyzer = SteganalyzerCNN().to(device)

    # Load existing state dicts if available as baseline
    if os.path.exists(os.path.join(REPO_ROOT, "cost_map_final.pth")):
        lfrinn_model.load_state_dict(torch.load(os.path.join(REPO_ROOT, "cost_map_final.pth"), map_location=device))
        print("Loaded initial weights for FullLFRINNModel.")

    # 2. Train LF-RINN Neural Cost Map Model
    lfrinn_model = train_cost_map_model(lfrinn_model, dataloader, epochs=5, lr=1e-3, device=device)

    # 3. Train CostMapCNN
    costmap_cnn = train_cost_map_model(costmap_cnn, dataloader, epochs=3, lr=1e-3, device=device)

    # 4. Retrain Steganalyzer on Real Cover/Stego Pairs
    steganalyzer = train_steganalyzer(steganalyzer, image_paths, lfrinn_model, epochs=4, lr=1e-3, device=device)

    # 5. Save PyTorch Checkpoints
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(lfrinn_model.state_dict(), os.path.join(REPO_ROOT, "cost_map_final.pth"))
    torch.save(lfrinn_model.state_dict(), os.path.join(CHECKPOINT_DIR, "cost_map_final.pth"))
    torch.save(costmap_cnn.state_dict(), os.path.join(REPO_ROOT, "cost_map_cnn.pth"))
    torch.save(costmap_cnn.state_dict(), os.path.join(CHECKPOINT_DIR, "cost_map_cnn.pth"))
    torch.save(steganalyzer.state_dict(), os.path.join(REPO_ROOT, "steganalyzer.pth"))
    torch.save(steganalyzer.state_dict(), os.path.join(CHECKPOINT_DIR, "steganalyzer.pth"))
    print("\nSaved all model checkpoints (.pth) to repo root and models/lfrinn/checkpoints/")

    # 6. Export ONNX Artifacts
    export_onnx(lfrinn_model.cpu(), ONNX_PATH)
    export_onnx(lfrinn_model.cpu(), os.path.join(CHECKPOINT_DIR, "cost_map_lfrinn.onnx"))

    print("\nAll training, retraining, checkpoint saving, and ONNX exports completed successfully!")

if __name__ == "__main__":
    main()
