#!/usr/bin/env python3
"""Inspect atom covariances and metric anisotropy on/off manifold."""
import torch
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from utils.metric_helpers import load_metric_bundle

metric_path = Path('outputs/low_data_models/ellipses_auto_scaling_v6/aniso/N100/seed013/2026-02-23_10-57-58/rhvae_metric.pt')
centroids, atoms, temperature, regularization, cfg_dict = load_metric_bundle(metric_path)

print(f'Temperature: {float(temperature):.4f}')
print(f'Centroids: {centroids.shape}, Atoms: {atoms.shape}')

# === Atom covariances Sigma_k = M_k @ M_k^T ===
cov = torch.bmm(atoms, atoms.transpose(-1, -2))
all_eigs = torch.linalg.eigvalsh(cov)
cond_numbers = all_eigs[:, -1] / all_eigs[:, 0].clamp(min=1e-10)

print('\n=== Individual Atom Covariances ===')
print(f'  min eig range: [{float(all_eigs[:, 0].min()):.4f}, {float(all_eigs[:, 0].max()):.4f}]')
print(f'  max eig range: [{float(all_eigs[:, 1].min()):.4f}, {float(all_eigs[:, 1].max()):.4f}]')
print(f'  cond: min={float(cond_numbers.min()):.2f}, max={float(cond_numbers.max()):.2f}, mean={float(cond_numbers.mean()):.2f}')
for i in [0, 25, 50, 75, 99]:
    e = all_eigs[i]
    print(f'  Sigma_{i}: eigs=[{float(e[0]):.4f}, {float(e[1]):.4f}], cond={float(e[1]/e[0].clamp(min=1e-10)):.2f}')

# === Kernel weights at centroid 0 ===
tau = float(temperature)
z = centroids[0:1]
diff = centroids.unsqueeze(0) - z.unsqueeze(1)
dists_sq = (diff**2).sum(-1)
weights = torch.exp(-dists_sq / (tau**2))
sorted_w, sorted_idx = weights[0].sort(descending=True)
eff_n = float((weights.sum()**2) / (weights**2).sum())
print(f'\n=== Kernel weights at centroid 0 (tau={tau}) ===')
print(f'  Top 10 weights: {[f"{float(w):.4f}" for w in sorted_w[:10]]}')
print(f'  Sum: {float(weights.sum()):.4f}, Effective centroids: {eff_n:.1f}')

# === Weighted sum → G_inv ===
weighted_cov = torch.einsum('bk,kij->bij', weights, cov)
G_inv_manual = weighted_cov[0] + regularization * torch.eye(2)
eig_sum = torch.linalg.eigvalsh(G_inv_manual)
print(f'\n=== G_inv at centroid 0 (manual) ===')
print(f'  eigs=[{float(eig_sum[0]):.4f}, {float(eig_sum[1]):.4f}], cond={float(eig_sum[1]/eig_sum[0]):.2f}')

# === Top contributing atoms — orientation ===
print(f'\n=== Top atoms contributing at centroid 0 ===')
for rank in range(min(10, len(sorted_idx))):
    idx = int(sorted_idx[rank])
    w = float(sorted_w[rank])
    ae = all_eigs[idx]
    _, evecs = torch.linalg.eigh(cov[idx])
    angle = float(torch.atan2(evecs[1, 1], evecs[0, 1]).item()) * 180 / 3.14159
    print(f'  rank {rank}: atom {idx}, w={w:.4f}, eigs=[{float(ae[0]):.4f}, {float(ae[1]):.4f}], cond={float(ae[1]/ae[0].clamp(min=1e-10)):.2f}, angle={angle:.1f}°')
