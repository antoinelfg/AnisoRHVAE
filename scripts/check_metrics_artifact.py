import torch
import os
import glob
from pathlib import Path

# THE GOLDEN TRUTH: CORE 4c SETTINGS
golden_truth = {
    "kernel_type": "mahalanobis",
    "atom_norm": "trace",
    "atom_power": 1.0959864113997024,
    "kernel_power": 1.0,
    "precision_jitter": 0.01,
    "void_threshold": 1.2,
    "void_weight_threshold": -1.0,
    "void_decay_type": "invquad",
    "void_decay_scale": 8.87131893173153,
    "void_decay_power": 1.7447710342008058,
    "void_decay_softplus_k": 5.0,
    "radial_stretch": 9.252860449508804,
    "transition_steepness": 7.276725930229776,
    "use_attractor": True,
    "attractor_smoothness": "soft",
    "attractor_metric": "mahalanobis",
    "attractor_use_det": True,
    "attractor_gamma": 7.624554217806352,
    "attractor_k_nearest": 1,
    "attractor_bias_energy": 18.0,
    "void_eigshape_mode": "det_preserving_spectral",
    "void_eigshape_alpha_min": 0.8,
    "void_eigshape_power": -1.2,
    "void_eigshape_eig_floor": 1e-8
}

def main():
    base_dir = Path("/home/alaforgu/scratch/longitudinal_experiments/AnisoRHVAE/outputs/low_data_models/ellipses_lowdata_v1_latent2_check/aniso/N050/seed013")
    runs = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
    if not runs:
        print("No runs found!")
        return
    
    latest_run = runs[0]
    metric_path = latest_run / "rhvae_metric.pt"
    if not metric_path.exists():
        print(f"Missing {metric_path}")
        return
    
    print(f"Checking {metric_path}...")
    payload = torch.load(metric_path, map_location="cpu")
    config = payload.get("config", {})
    
    all_passed = True
    for k, v_expected in golden_truth.items():
        v_actual = config.get(k, None)
        if isinstance(v_expected, float):
            match = (v_actual is not None) and (abs(v_actual - v_expected) < 1e-6)
        else:
            match = (v_actual == v_expected)
            
        if not match:
            print(f"CRITICAL FAIL: {k} | Expected: {v_expected} | Actual: {v_actual}")
            all_passed = False
        else:
            print(f"PASS: {k} = {v_actual}")
    
    if all_passed:
        print("All parameters correctly enforced in artifact!")
    else:
        print("Artifact validation failed.")
    
if __name__ == "__main__":
    main()
