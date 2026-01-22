import torch
import lightning as L
from typing import Optional
from torch.utils.data import DataLoader, random_split, Dataset
from omegaconf import DictConfig

from .ellipse_sequences import EllipseSequenceDataset, DeterministicEllipseSequenceDataset


class _SequenceOnlyDataset(Dataset):
    """Wrap a dataset that returns (sequence, label) to expose only the sequence."""

    def __init__(self, base_dataset: Dataset):
        self._base = base_dataset

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        seq = self._base[idx]
        if isinstance(seq, (tuple, list)) and len(seq) > 0:
            seq = seq[0]
        # Return dict with masks to satisfy downstream components expecting seq_mask/pix_mask
        seq_len = seq.shape[0] if hasattr(seq, "shape") else None
        seq_mask = None
        if seq_len is not None:
            seq_mask = torch.ones((seq_len,), dtype=torch.bool)
        pix_mask = torch.ones_like(seq, dtype=torch.bool) if hasattr(torch, "ones_like") else None
        return {"data": seq, "seq_mask": seq_mask, "pix_mask": pix_mask}


class EllipseSequenceDataModule(L.LightningDataModule):
    """Lightning DataModule for procedurally generated ellipse sequences."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.batch_size = config.get("batch_size", 64)
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = config.get("pin_memory", True)
        self.persistent_workers = config.get("persistent_workers", False)
        self.drop_last = bool(config.get("drop_last", False))
        self.seed = int(config.get("seed", 42))
        self.train_ratio = float(config.get("train_ratio", 0.8))
        self.val_ratio = float(config.get("val_ratio", 0.1))
        self.test_ratio = float(config.get("test_ratio", 0.1))
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Optional safety override for environments where CUDA initialization
        # or forked worker + pinned memory causes failures.
        import os
        if os.environ.get("RLVAE_CUDA_SAFE_DATALOADER", "0") == "1":
            print("[DATALOADER SAFE] Enabling safe settings: num_workers=0, pin_memory=False, persistent_workers=False")
            self.num_workers = 0
            self.pin_memory = False
            self.persistent_workers = False

    # ------------------------------------------------------------------
    # Dataset construction helpers
    # ------------------------------------------------------------------
    def _build_dataset(self, data_path: Optional[str] = None) -> EllipseSequenceDataset:
        cfg = self.config
        name = str(cfg.get("name", cfg.get("dataset", "ellipse_sequences"))).lower()
        schedule_type = str(cfg.get("schedule_type", "linear")).lower()
        deterministic_flag = bool(cfg.get("deterministic", False))
        use_deterministic = deterministic_flag or ("deterministic" in name) or (schedule_type.startswith("deterministic"))

        ds_cls = DeterministicEllipseSequenceDataset if use_deterministic else EllipseSequenceDataset
        ds_kwargs = dict(
            num_sequences=int(cfg.get("num_sequences", 1000)),
            seq_len=int(cfg.get("seq_len", 8)),
            image_size=tuple(cfg.get("image_size", (64, 64))),
            min_eccentricity=float(cfg.get("min_eccentricity", 0.0)),
            max_eccentricity=float(cfg.get("max_eccentricity", 0.9)),
            min_radius=int(cfg.get("min_radius", 8)),
            max_radius=int(cfg.get("max_radius", 20)),
            center_jitter=int(cfg.get("center_jitter", 4)),
            antialias=bool(cfg.get("antialias", True)),
            seed=int(cfg.get("seed", 42)),
            fix_center=bool(cfg.get("fix_center", True)),
            fix_theta=bool(cfg.get("fix_theta", True)),
            fix_intensity=bool(cfg.get("fix_intensity", True)),
            keep_major_axis_constant=bool(cfg.get("keep_major_axis_constant", True)),
            keep_area_constant=bool(cfg.get("keep_area_constant", False)),
            outline_only=bool(cfg.get("outline_only", False)),
            outline_width=int(cfg.get("outline_width", 2)),
            # Schedule / sinusoidal options
            schedule_type=str(cfg.get("schedule_type", "linear")),
            sinusoidal_amplitude_range=tuple(cfg.get("sinusoidal_amplitude_range", (0.35, 0.45))),
            sinusoidal_phase_range=tuple(cfg.get("sinusoidal_phase_range", (0.0, 2 * 3.141592653589793))),
            sinusoidal_center=float(cfg.get("sinusoidal_center", 0.45)) if cfg.get("sinusoidal_center", None) is not None else None,
            sinusoidal_cycle=bool(cfg.get("sinusoidal_cycle", False)),
            sinusoidal_frequency=float(cfg.get("sinusoidal_frequency", 1.0)),
            data_path=data_path,
        )
        if ds_cls is DeterministicEllipseSequenceDataset:
            ds_kwargs["deterministic_delta_scale"] = float(cfg.get("deterministic_delta_scale", 0.4))
        return ds_cls(**ds_kwargs)

    # ------------------------------------------------------------------
    # Lightning DataModule interface
    # ------------------------------------------------------------------
    def setup(self, stage: str = None, training_config: DictConfig = None):
        cfg = self.config
        base_path = cfg.get("data_dir", None)
        if base_path is None:
            base_path = cfg.get("data_path", None)
        train_path = cfg.get("train_path", None)
        val_path = cfg.get("val_path", None)
        test_path = cfg.get("test_path", None)

        # Allow users to point to a directory containing train/val/test.pt
        from pathlib import Path

        resolved_train = resolved_val = resolved_test = None
        if base_path is not None:
            base = Path(base_path)
            if base.is_dir():
                resolved_train = base / "train.pt"
                resolved_val = base / "val.pt"
                resolved_test = base / "test.pt"
            else:
                resolved_train = base
                maybe_dir = base.parent
                candidate_val = maybe_dir / "val.pt"
                candidate_test = maybe_dir / "test.pt"
                if candidate_val.exists() and candidate_test.exists():
                    resolved_val = candidate_val
                    resolved_test = candidate_test

        # Explicit override paths take precedence
        if train_path is not None:
            resolved_train = Path(train_path)
        if val_path is not None:
            resolved_val = Path(val_path)
        if test_path is not None:
            resolved_test = Path(test_path)

        use_pre_splits = (
            resolved_train is not None
            and resolved_val is not None
            and resolved_test is not None
            and resolved_train.exists()
            and resolved_val.exists()
            and resolved_test.exists()
        )

        if use_pre_splits:
            train_ds = self._build_dataset(str(resolved_train))
            val_ds = self._build_dataset(str(resolved_val))
            test_ds = self._build_dataset(str(resolved_test))
            base_dataset = train_ds
            self.train_dataset = _SequenceOnlyDataset(train_ds)
            self.val_dataset = _SequenceOnlyDataset(val_ds)
            self.test_dataset = _SequenceOnlyDataset(test_ds)
        else:
            base_dataset = self._build_dataset()
            total = len(base_dataset)

            train_ratio = min(max(self.train_ratio, 0.0), 1.0)
            val_ratio = min(max(self.val_ratio, 0.0), 1.0 - train_ratio)
            test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

            n_train = max(1, int(total * train_ratio))
            n_val = max(1, int(total * val_ratio))
            n_remaining = total - n_train - n_val
            n_test = max(1, n_remaining) if test_ratio > 0.0 else n_remaining
            if n_train + n_val + n_test != total:
                n_test = total - n_train - n_val

            generator = torch.Generator().manual_seed(self.seed)
            subsets = random_split(base_dataset, [n_train, n_val, n_test], generator=generator)
            self.train_dataset = _SequenceOnlyDataset(subsets[0])
            self.val_dataset = _SequenceOnlyDataset(subsets[1])
            self.test_dataset = _SequenceOnlyDataset(subsets[2])

        # Expose basic shape hints back to the config for downstream automation
        self.config.sequence_length = base_dataset.seq_len
        self.config.channels = 1

        if training_config is not None and hasattr(training_config, "data"):
            data_cfg = training_config.data
            self.batch_size = data_cfg.batch_size
            self.num_workers = data_cfg.num_workers
            self.pin_memory = data_cfg.pin_memory
            if hasattr(data_cfg, "drop_last"):
                self.drop_last = bool(data_cfg.drop_last)
        # Re-apply safety override after reading training config
        import os
        if os.environ.get("RLVAE_CUDA_SAFE_DATALOADER", "0") == "1":
            self.num_workers = 0
            self.pin_memory = False
            self.persistent_workers = False

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=self.persistent_workers,
        )
