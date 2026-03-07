import datetime
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.analyze_metric_full import load_metric_geometry

model_path = Path('outputs/pythae_rhvae_baseline/2026-02-09_11-54-09')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, centroids = load_metric_geometry(model_path, device)

bounds = 6.0
resolution = 180
axis = np.linspace(-bounds, bounds, resolution)
X, Y = np.meshgrid(axis, axis)
grid_2d = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

latent_dim = model.latent_dim
if latent_dim > 2:
    padded = np.zeros((grid_2d.shape[0], latent_dim), dtype=np.float32)
    padded[:, :2] = grid_2d
    grid = padded
else:
    grid = grid_2d

tensor_grid = torch.from_numpy(grid).to(device)

batch = 1024
norms = []
for i in range(0, tensor_grid.shape[0], batch):
    z = tensor_grid[i:i+batch].clone().detach().requires_grad_(True)
    G_inv = model.G_inv(z)
    logdet_inv = torch.linalg.slogdet(G_inv).logabsdet
    obj = 0.5 * logdet_inv
    grad = torch.autograd.grad(obj.sum(), z, create_graph=False)[0]
    grad_2d = grad[:, :2]
    norm = torch.linalg.norm(grad_2d, dim=1)
    norms.append(norm.detach().cpu().numpy())

norms = np.concatenate(norms)
norm_grid = norms.reshape(resolution, resolution)
log_norm_grid = np.log10(norm_grid + 1e-8)

centroids_np = centroids[:, :2].detach().cpu().numpy()

out_base = model_path / 'analysis_results'
out_base.mkdir(parents=True, exist_ok=True)
run_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
out_dir = out_base / f'border_gradcheck_{run_stamp}'
out_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(
    log_norm_grid,
    origin='lower',
    extent=[-bounds, bounds, -bounds, bounds],
    cmap='magma',
)
ax.scatter(
    centroids_np[:, 0],
    centroids_np[:, 1],
    s=16,
    c='cyan',
    edgecolor='white',
    linewidth=0.3,
    alpha=0.85,
)
ax.set_title('log10 ||∇ (0.5 log det G^{-1})||')
ax.set_xlabel('Latent Dim 1')
ax.set_ylabel('Latent Dim 2')
fig.colorbar(im, ax=ax, label='log10 grad norm')
fig.tight_layout()
fig.savefig(out_dir / 'volume_gradmag_log10.png', dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(
    norm_grid,
    origin='lower',
    extent=[-bounds, bounds, -bounds, bounds],
    cmap='viridis',
)
ax.scatter(
    centroids_np[:, 0],
    centroids_np[:, 1],
    s=16,
    c='cyan',
    edgecolor='white',
    linewidth=0.3,
    alpha=0.85,
)
ax.set_title('||∇ (0.5 log det G^{-1})||')
ax.set_xlabel('Latent Dim 1')
ax.set_ylabel('Latent Dim 2')
fig.colorbar(im, ax=ax, label='grad norm')
fig.tight_layout()
fig.savefig(out_dir / 'volume_gradmag_linear.png', dpi=160)
plt.close(fig)

print(out_dir)
