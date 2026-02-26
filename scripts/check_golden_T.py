import torch
import sys
from pathlib import Path

# Add project root to path so we can import src and scripts
project_root = Path("/home/alaforgu/scratch/longitudinal_experiments/AnisoRHVAE").resolve()
sys.path.insert(0, str(project_root))

from src.models.rhvae_geometry import GeometryRHVAE
from scripts.run_pythae_rhvae_baseline import _get_centroids_tensor, _suggest_temperature

m_path = project_root / "outputs" / "metric_core4_sweep" / "2026-02-17_11-07-08"
m = GeometryRHVAE.load_from_folder(str(m_path))
c = _get_centroids_tensor(m)

print("mean_nn for N=800 core4:", _suggest_temperature(c, stat="mean_nn"))
print("median_nn:", _suggest_temperature(c, stat="median_nn"))
print("T actually used:", m.temperature.item())
