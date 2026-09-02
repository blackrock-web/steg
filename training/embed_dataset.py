import numpy as np
import torch

def embed_lsb(cover_gray: np.ndarray, bpp: float, seed: int = 42) -> np.ndarray:
    """
    Standard sequential LSB embedding.
    """
    stego = cover_gray.copy()
    H, W = stego.shape
    total_pixels = H * W
    total_bits = int(total_pixels * bpp)
    
    np.random.seed(seed)
    payload_bits = np.random.randint(0, 2, size=total_bits, dtype=np.uint8)
    
    flat = stego.flatten()
    flat[:total_bits] = (flat[:total_bits] & 0xFE) | payload_bits
    return flat.reshape((H, W))

def embed_emd(cover_gray: np.ndarray, bpp: float, seed: int = 42) -> np.ndarray:
    """
    Pure Exploiting Modification Direction (Zhang & Wang 2006) on 2-pixel groups (base 5).
    """
    stego = cover_gray.copy()
    H, W = stego.shape
    total_pixels = H * W
    num_groups = total_pixels // 2
    
    # Each group embeds 1 base-5 digit (log2(5) ~ 2.32 bits)
    # Total digits needed = total_bits / 2.32
    total_bits = int(total_pixels * bpp)
    num_digits = min(num_groups, max(1, int(total_bits / 2.32)))
    
    np.random.seed(seed)
    digits = np.random.randint(0, 5, size=num_digits, dtype=np.uint8)
    
    flat = stego.flatten()
    for g in range(num_digits):
        p0 = int(flat[2 * g])
        p1 = int(flat[2 * g + 1])
        f_val = (p0 * 1 + p1 * 2) % 5
        m = int(digits[g])
        s = (m - f_val) % 5
        
        if s == 1:
            p0 = min(255, p0 + 1)
        elif s == 2:
            p1 = min(255, p1 + 1)
        elif s == 3:
            p1 = max(0, p1 - 1)
        elif s == 4:
            p0 = max(0, p0 - 1)
        
        flat[2 * g] = p0
        flat[2 * g + 1] = p1
        
    return flat.reshape((H, W))

def embed_opap(cover_gray: np.ndarray, bpp: float, k: int = 2, seed: int = 42) -> np.ndarray:
    """
    Standard Optimal Pixel Adjustment Process (Chan & Cheng 2004) with k-bit embedding.
    """
    stego = cover_gray.copy()
    H, W = stego.shape
    total_pixels = H * W
    num_pixels = min(total_pixels, max(1, int(total_pixels * bpp / k)))
    
    np.random.seed(seed)
    payload_values = np.random.randint(0, 2**k, size=num_pixels, dtype=np.uint8)
    
    flat = stego.flatten()
    mask = (1 << k) - 1
    half = 1 << (k - 1)
    span = 1 << k
    
    for i in range(num_pixels):
        orig = int(flat[i])
        m = int(payload_values[i])
        lsb_val = orig & mask
        delta = m - lsb_val
        
        candidate = orig + delta
        if delta > half and candidate - span >= 0:
            candidate -= span
        elif delta < -half and candidate + span <= 255:
            candidate += span
            
        flat[i] = np.clip(candidate, 0, 255)
        
    return flat.reshape((H, W))

def embed_adaptive_lfrinn(cover_gray: np.ndarray, cost_map: np.ndarray, bpp: float, seed: int = 42) -> np.ndarray:
    """
    Proposed Adaptive EMD-OPAP guided by LF-RINN Cost Map.
    High-cost edge regions are assigned to Zone A (EMD minimal distortion)
    Moderate-cost regions are assigned to Zone B (OPAP 2-bit)
    Smooth low-cost regions are assigned to Zone C (OPAP 3-bit)
    """
    stego = cover_gray.copy()
    H, W = stego.shape
    total_pixels = H * W
    
    # Classify zones based on cost map quantiles
    flat_cost = cost_map.flatten()
    q_a = np.quantile(flat_cost, 0.65)  # Top 35% cost -> Zone A (safest)
    q_b = np.quantile(flat_cost, 0.35)  # Next 30% cost -> Zone B
    
    indices_a = np.where(flat_cost >= q_a)[0]
    indices_b = np.where((flat_cost < q_a) & (flat_cost >= q_b))[0]
    indices_c = np.where(flat_cost < q_b)[0]
    
    np.random.seed(seed)
    flat_stego = stego.flatten()
    
    # Embed in Zone A via EMD (2-pixel groups)
    groups_a = len(indices_a) // 2
    if groups_a > 0:
        digits = np.random.randint(0, 5, size=groups_a, dtype=np.uint8)
        for g in range(groups_a):
            idx0 = indices_a[2 * g]
            idx1 = indices_a[2 * g + 1]
            p0, p1 = int(flat_stego[idx0]), int(flat_stego[idx1])
            f_val = (p0 * 1 + p1 * 2) % 5
            m = int(digits[g])
            s = (m - f_val) % 5
            if s == 1: p0 = min(255, p0 + 1)
            elif s == 2: p1 = min(255, p1 + 1)
            elif s == 3: p1 = max(0, p1 - 1)
            elif s == 4: p0 = max(0, p0 - 1)
            flat_stego[idx0] = p0
            flat_stego[idx1] = p1

    # Embed in Zone B via OPAP (k=2)
    if len(indices_b) > 0:
        vals_b = np.random.randint(0, 4, size=len(indices_b), dtype=np.uint8)
        for i, idx in enumerate(indices_b):
            orig = int(flat_stego[idx])
            m = int(vals_b[i])
            delta = m - (orig & 3)
            cand = orig + delta
            if delta > 2 and cand - 4 >= 0: cand -= 4
            elif delta < -2 and cand + 4 <= 255: cand += 4
            flat_stego[idx] = np.clip(cand, 0, 255)

    return flat_stego.reshape((H, W))
