#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.utils.low_data_io import (
    load_split_tensor,
    load_subset_indices,
    model_output_dir,
    subset_from_indices,
    write_json,
)
from src.utils.wandb_logging import init_wandb_run, safe_wandb_finish, safe_wandb_log


class VanillaVAE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 512, input_dim: int = 28 * 28):
        super().__init__()
        in_dim = int(input_dim)
        self.input_dim = in_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


@dataclass
class TrainConfig:
    latent_dim: int
    hidden_dim: int
    input_dim: int
    epochs: int
    batch_size: int
    lr: float
    beta_kl: float
    n: int
    seed: int
    dataset_id: str


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta_kl: float) -> tuple[torch.Tensor, float, float]:
    recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total = recon_loss + beta_kl * kl
    return total, float(recon_loss.item()), float(kl.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Vanilla VAE baseline on low-data RotMNIST subset.")
    parser.add_argument("--processed_dir", type=str, default="data/processed/rotmnist/v1")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_kl", type=float, default=1.0)
    parser.add_argument("--output_root", type=str, default="outputs/low_data_models/rotmnist_v1")
    parser.add_argument("--model_id", type=str, default="vanilla_vae")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_name_mode", type=str, default="auto", choices=["timestamp", "auto", "manual"])
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)

    wandb_run = init_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        run_name=args.wandb_run_name,
        name_mode=args.wandb_name_mode,
        mode=args.wandb_mode,
        config=vars(args),
        name_prefix=f"lowdata_{args.model_id}_N{int(args.n):03d}_seed{int(args.seed):03d}",
    )

    try:
        processed_dir = (ROOT / args.processed_dir).resolve()
        dataset_id = str(processed_dir.name)
        metadata_path = processed_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                dataset_id = str(metadata.get("dataset_id", dataset_id))
            except Exception:
                pass
        train_images, train_labels = load_split_tensor(processed_dir, "train")
        val_images, val_labels = load_split_tensor(processed_dir, "val")

        subset_idx = load_subset_indices(processed_dir, args.n, args.seed)
        train_images, train_labels = subset_from_indices(train_images, train_labels, subset_idx)

        x_train = train_images.reshape(train_images.shape[0], -1)
        x_val = val_images.reshape(val_images.shape[0], -1)
        input_dim = int(x_train.shape[1])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VanillaVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_loader = DataLoader(TensorDataset(x_train, train_labels), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, val_labels), batch_size=args.batch_size, shuffle=False)

        history: list[dict[str, float]] = []
        best_val = float("inf")

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            for xb, _ in train_loader:
                xb = xb.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(xb)
                loss, _, _ = _loss(recon, xb, mu, logvar, args.beta_kl)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    recon, mu, logvar = model(xb)
                    loss, _, _ = _loss(recon, xb, mu, logvar, args.beta_kl)
                    val_losses.append(float(loss.item()))

            train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
            val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
            best_val = min(best_val, val_mean)
            history.append({"epoch": epoch, "train_loss": train_mean, "val_loss": val_mean})
            print(f"[train_vanilla_vae] epoch={epoch} train={train_mean:.4f} val={val_mean:.4f}")
            safe_wandb_log(
                wandb_run,
                {
                    "train/loss": train_mean,
                    "val/loss": val_mean,
                    "train/epoch": int(epoch),
                    "meta/n": int(args.n),
                    "meta/seed": int(args.seed),
                    "meta/model_id": str(args.model_id),
                },
                step=epoch,
            )

        out_dir = model_output_dir((ROOT / args.output_root).resolve(), args.model_id, args.n, args.seed)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "input_dim": input_dim,
            },
            out_dir / "model.pt",
        )

        cfg = TrainConfig(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            input_dim=input_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta_kl=args.beta_kl,
            n=args.n,
            seed=args.seed,
            dataset_id=dataset_id,
        )
        write_json(out_dir / "config.json", asdict(cfg))

        torch.save(
            {
                "images": train_images,
                "labels": train_labels,
                "subset_indices": subset_idx,
            },
            out_dir / "train_data.pt",
        )

        metrics = {
            "final_train_loss": history[-1]["train_loss"] if history else float("nan"),
            "final_val_loss": history[-1]["val_loss"] if history else float("nan"),
            "best_val_loss": best_val,
            "history": history,
        }
        write_json(out_dir / "metrics.json", metrics)

        safe_wandb_log(
            wandb_run,
            {
                "final/train_loss": metrics["final_train_loss"],
                "final/val_loss": metrics["final_val_loss"],
                "final/best_val_loss": metrics["best_val_loss"],
                "meta/output_dir": str(out_dir),
            },
        )
        if wandb_run is not None:
            wandb_run.summary["output_dir"] = str(out_dir)
            wandb_run.summary["n"] = int(args.n)
            wandb_run.summary["seed"] = int(args.seed)
            wandb_run.summary["model_id"] = str(args.model_id)

        print(f"[train_vanilla_vae] output_dir={out_dir}")
    finally:
        safe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
