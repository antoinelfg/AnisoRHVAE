import torch
import os
from pathlib import Path

def get_latest_run(base_dir: Path):
    if not base_dir.exists():
        return None
    runs = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
    if not runs:
        return None
    return runs[0]

def analyze_metrics(output_root="outputs/low_data_models/ellipses_auto_scaling_v6/aniso"):
    root = Path(output_root)
    results = {}
    
    for n in [50, 100, 1000]:
        n_dir = root / f"N{n:03d}" / "seed013"
        latest_run = get_latest_run(n_dir)
        
        if latest_run is None:
            print(f"[{n}] No run found in {n_dir}")
            continue
            
        metric_file = latest_run / "rhvae_metric.pt"
        model_file = latest_run / "rhvae_model.pt"
        if not metric_file.exists() or not model_file.exists():
            print(f"[{n}] Missing metric/model file in {latest_run}")
            continue
            
        payload = torch.load(metric_file, map_location="cpu", weights_only=False)
        state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
        cfg = payload.get("config", {})
        
        # Temperature dynamically scaled, get it from state_dict.
        t_param = state_dict.get("temperature", None)
        if t_param is not None:
            t = t_param.item()
        else:
            t = cfg.get("temperature", None)

        r0 = cfg.get("void_threshold", None)
        lbd = cfg.get("regularization", None)
        
        # Centroid count from the model file
        centroids = state_dict.get("centroids", None)
        n_centroids = centroids.shape[0] if centroids is not None else "?"
        
        if t is not None and r0 is not None:
            effective_r0 = t * r0
        else:
            effective_r0 = None

        results[n] = {
            "temperature": t,
            "void_threshold_multiplier": r0,
            "effective_r0": effective_r0,
            "regularization": lbd,
            "n_centroids": n_centroids,
        }
        print(f"[{n}] Run: {latest_run.name}, T={t:.4f}, centroids={n_centroids}")
        
    print()
    print(f"{'N':<5} | {'Temp (τ)':<10} | {'r0_mult':<10} | {'Eff r0 (τ*r0)':<14} | {'Reg (λ)':<8} | {'Centroids':<10}")
    print("-" * 75)
    for n in [50, 100, 1000]:
        if n not in results:
            print(f"{n:<5} | PENDING")
            continue
        res = results[n]
        t = res['temperature']
        r0_m = res['void_threshold_multiplier']
        eff_r0 = res['effective_r0']
        lbd = res['regularization']
        nc = res['n_centroids']
        print(f"{n:<5} | {t:<10.4f} | {r0_m:<10.2f} | {eff_r0:<14.4f} | {lbd:<8.4f} | {nc}")

if __name__ == "__main__":
    analyze_metrics()
