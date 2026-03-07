#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class EBM(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


@dataclass
class EBMTrainConfig:
    n: int
    seed: int
    latent_dim: int
    encoder_hidden_dim: int
    ebm_hidden_dim: int
    ae_epochs: int
    ebm_epochs: int
    batch_size: int
    lr_ae: float
    lr_ebm: float
    beta_conformal: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent EBM conformal baseline on low-data RotMNIST subset.")
    parser.add_argument("--processed_dir", type=str, default="data/processed/rotmnist/v1")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--encoder_hidden_dim", type=int, default=512)
    parser.add_argument("--ebm_hidden_dim", type=int, default=256)
    parser.add_argument("--ae_epochs", type=int, default=15)
    parser.add_argument("--ebm_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_ae", type=float, default=1e-3)
    parser.add_argument("--lr_ebm", type=float, default=1e-3)
    parser.add_argument("--beta_conformal", type=float, default=1.0)
    parser.add_argument("--output_root", type=str, default="outputs/low_data_models/rotmnist_v1")
    parser.add_argument("--model_id", type=str, default="ebm_conformal")
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
        train_images, train_labels = load_split_tensor(processed_dir, "train")
        subset_idx = load_subset_indices(processed_dir, args.n, args.seed)
        train_images, train_labels = subset_from_indices(train_images, train_labels, subset_idx)

        x_train = train_images.reshape(train_images.shape[0], -1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        enc = Encoder(latent_dim=args.latent_dim, hidden_dim=args.encoder_hidden_dim).to(device)
        dec = Decoder(latent_dim=args.latent_dim, hidden_dim=args.encoder_hidden_dim).to(device)

        ae_opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr_ae)
        loader = DataLoader(TensorDataset(x_train, train_labels), batch_size=args.batch_size, shuffle=True)

        ae_history = []
        for epoch in range(1, args.ae_epochs + 1):
            losses = []
            enc.train()
            dec.train()
            for xb, _ in loader:
                xb = xb.to(device)
                ae_opt.zero_grad()
                z = enc(xb)
                recon = dec(z)
                loss = F.binary_cross_entropy(recon, xb, reduction="mean")
                loss.backward()
                ae_opt.step()
                losses.append(float(loss.item()))
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            ae_history.append(mean_loss)
            print(f"[train_ebm_conformal][AE] epoch={epoch} loss={mean_loss:.4f}")
            safe_wandb_log(
                wandb_run,
                {
                    "ae/loss": mean_loss,
                    "ae/epoch": int(epoch),
                    "meta/n": int(args.n),
                    "meta/seed": int(args.seed),
                    "meta/model_id": str(args.model_id),
                },
                step=epoch,
            )

        enc.eval()
        with torch.no_grad():
            z_train = enc(x_train.to(device)).detach()

        ebm = EBM(latent_dim=args.latent_dim, hidden_dim=args.ebm_hidden_dim).to(device)
        ebm_opt = torch.optim.Adam(ebm.parameters(), lr=args.lr_ebm)
        z_loader = DataLoader(TensorDataset(z_train.cpu()), batch_size=args.batch_size, shuffle=True)

        ebm_history = []
        for epoch in range(1, args.ebm_epochs + 1):
            losses = []
            ebm.train()
            for (z_real_cpu,) in z_loader:
                z_real = z_real_cpu.to(device)
                z_fake = torch.randn_like(z_real)
                e_real = ebm(z_real)
                e_fake = ebm(z_fake)
                # Margin-like energy separation objective with L2 stabilization.
                loss = (e_real.mean() - e_fake.mean()) + 1e-4 * ((e_real ** 2).mean() + (e_fake ** 2).mean())
                ebm_opt.zero_grad()
                loss.backward()
                ebm_opt.step()
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses)) if losses else float("nan")
            ebm_history.append(mean_loss)
            print(f"[train_ebm_conformal][EBM] epoch={epoch} loss={mean_loss:.4f}")
            safe_wandb_log(
                wandb_run,
                {
                    "ebm/loss": mean_loss,
                    "ebm/epoch": int(epoch),
                    "meta/n": int(args.n),
                    "meta/seed": int(args.seed),
                    "meta/model_id": str(args.model_id),
                },
                step=args.ae_epochs + epoch,
            )

        out_dir = model_output_dir((ROOT / args.output_root).resolve(), args.model_id, args.n, args.seed)

        torch.save(
            {
                "state_dict": enc.state_dict(),
                "latent_dim": args.latent_dim,
                "hidden_dim": args.encoder_hidden_dim,
            },
            out_dir / "encoder.pt",
        )
        torch.save(
            {
                "state_dict": dec.state_dict(),
                "latent_dim": args.latent_dim,
                "hidden_dim": args.encoder_hidden_dim,
            },
            out_dir / "decoder.pt",
        )
        torch.save(
            {
                "state_dict": ebm.state_dict(),
                "latent_dim": args.latent_dim,
                "hidden_dim": args.ebm_hidden_dim,
            },
            out_dir / "ebm.pt",
        )

        cfg = EBMTrainConfig(
            n=args.n,
            seed=args.seed,
            latent_dim=args.latent_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            ebm_hidden_dim=args.ebm_hidden_dim,
            ae_epochs=args.ae_epochs,
            ebm_epochs=args.ebm_epochs,
            batch_size=args.batch_size,
            lr_ae=args.lr_ae,
            lr_ebm=args.lr_ebm,
            beta_conformal=args.beta_conformal,
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
            "ae_final_loss": ae_history[-1] if ae_history else float("nan"),
            "ebm_final_loss": ebm_history[-1] if ebm_history else float("nan"),
            "ae_history": ae_history,
            "ebm_history": ebm_history,
        }
        write_json(out_dir / "metrics.json", metrics)

        safe_wandb_log(
            wandb_run,
            {
                "final/ae_loss": metrics["ae_final_loss"],
                "final/ebm_loss": metrics["ebm_final_loss"],
                "meta/output_dir": str(out_dir),
                "meta/model_id": str(args.model_id),
            },
        )
        if wandb_run is not None:
            wandb_run.summary["output_dir"] = str(out_dir)
            wandb_run.summary["n"] = int(args.n)
            wandb_run.summary["seed"] = int(args.seed)
            wandb_run.summary["model_id"] = str(args.model_id)

        print(f"[train_ebm_conformal] output_dir={out_dir}")
    finally:
        safe_wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
