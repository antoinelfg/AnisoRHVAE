import torch

def test_metric_conventions():
    # Let G_inv have a large eigenvalue along dimension 0, and small along dimension 1.
    # Imagine dim 0 is "along the manifold" or "radial escape direction"
    g_inv = torch.tensor([[100.0, 0.0], [0.0, 0.01]])
    g = torch.linalg.inv(g_inv)
    
    print(f"G_inv: {g_inv.diag().tolist()}")
    print(f"G: {g.diag().tolist()}")
    
    # 1. Standard Convention: M = G
    # rho ~ N(0, G)
    # v = G_inv @ rho
    cov_rho_std = g
    cov_v_std = g_inv @ cov_rho_std @ g_inv.T
    print("\n--- Standard Convention (M = G) ---")
    print("When G_inv has large eigenvalue 100 along dim 0...")
    print(f"Variance of velocity v: {cov_v_std.diag().tolist()}")
    print(f"-> Moves VERY FAST along dim 0 (var={cov_v_std[0,0].item()})")
    
    # 2. Dual Convention: M = G_inv
    # rho ~ N(0, G_inv)
    # v = G @ rho
    cov_rho_dual = g_inv
    cov_v_dual = g @ cov_rho_dual @ g.T
    print("\n--- Dual Convention (M = G_inv) ---")
    print("When G_inv has large eigenvalue 100 along dim 0...")
    print(f"Variance of velocity v: {cov_v_dual.diag().tolist()}")
    print(f"-> Moves VERY SLOWLY along dim 0 (var={cov_v_dual[0,0].item():.4f})")
    print(f"-> Moves VERY FAST along dim 1 (var={cov_v_dual[1,1].item():.4f})")
    
if __name__ == "__main__":
    test_metric_conventions()
