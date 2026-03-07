import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scripts.analyze_metric_full import load_metric_geometry



def check_alignment(model_path='outputs/pythae_rhvae_baseline/2026-02-09_17-27-43/', bounds=6.0, resolution=100, device='cuda'):
    model_path = Path(model_path)
    model, centroids = load_metric_geometry(model_path, device)
    
    # 1. Création de la grille
    axis = np.linspace(-bounds, bounds, resolution)
    X, Y = np.meshgrid(axis, axis)
    grid_2d = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)
    
    # Gestion dimension latente > 2
    if model.latent_dim > 2:
        z = torch.zeros((len(grid_2d), model.latent_dim), device=device)
        z[:, :2] = torch.from_numpy(grid_2d)
    else:
        z = torch.from_numpy(grid_2d).to(device)
        
    z.requires_grad_(True)
    
    # 2. Calcul du Gradient de Rescue (Force)
    # log det G^-1 = - log det G
    # On veut le gradient de 0.5 * log det G^-1
    G_inv = model.G_inv(z)
    logdet = torch.linalg.slogdet(G_inv).logabsdet
    obj = 0.5 * logdet
    
    # v_rescue = grad(obj)
    grads = torch.autograd.grad(obj.sum(), z)[0]
    # On projette sur 2D pour la visu (si dim > 2, c'est une approximation mais valide pour slice)
    v_rescue = grads[:, :2] 
    norm_rescue = torch.norm(v_rescue, dim=1, keepdim=True) + 1e-8
    v_rescue_dir = v_rescue / norm_rescue
    
    # 3. Calcul de la direction "Cheap" (Inertie)
    # On a besoin de G(z) pour trouver les vecteurs propres. 
    # G = (G^-1)^-1. 
    # Mais attention : Cheap direction = direction de mouvement facile 
    # = Petite valeur propre de G = Grande valeur propre de G^-1.
    # Donc on cherche le vecteur propre associé à la PLUS GRANDE valeur propre de G_inv.
    
    # Décomposition de G_inv (matrice de "mobilité")
    # G_inv est [Batch, D, D]
    L, Q = torch.linalg.eigh(G_inv) 
    # L est trié croissant. La plus grande valeur propre (mobilité max) est la dernière.
    # Le vecteur propre correspondant est la dernière colonne de Q.
    
    e_cheap = Q[:, :, -1] # [Batch, D]
    e_cheap_2d = e_cheap[:, :2]
    norm_cheap = torch.norm(e_cheap_2d, dim=1, keepdim=True) + 1e-8
    e_cheap_dir = e_cheap_2d / norm_cheap
    
    # 4. Calcul de l'alignement (Cosinus Absolu)
    # Absolu car le vecteur propre n'a pas de sens (v ou -v), c'est une ligne directrice.
    dot_prod = (v_rescue_dir * e_cheap_dir).sum(dim=1).abs()
    
    # Reshape pour affichage
    alignment_map = dot_prod.detach().cpu().numpy().reshape(resolution, resolution)
    grad_norm_map = norm_rescue.detach().cpu().numpy().reshape(resolution, resolution)
    
    # Masquer là où le gradient est nul (plateau), car l'alignement n'a pas de sens
    mask_plateau = grad_norm_map < 1e-4
    alignment_map[mask_plateau] = np.nan

    # 5. Plotting
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        alignment_map, 
        origin='lower', 
        extent=[-bounds, bounds, -bounds, bounds], 
        cmap='RdYlGn', # Rouge=0 (Orthogonal), Vert=1 (Aligné)
        vmin=0, vmax=1
    )
    
    # Overlay Centroids
    centroids_np = centroids.detach().cpu().numpy()
    ax.scatter(centroids_np[:, 0], centroids_np[:, 1], c='black', s=20, marker='x', alpha=0.5)
    
    ax.set_title(f"Alignment: Rescue Force vs Cheap Motion\n(1.0 = Perfect, 0.0 = Orthogonal)")
    plt.colorbar(im, ax=ax, label="|Cos(Angle)|")
    
    out_dir = model_path / "analysis_results"
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "rescue_alignment_map.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Alignment map saved to: {save_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        check_alignment(sys.argv[1])
    else:
        print("Usage: python check_alignment.py <model_path>")