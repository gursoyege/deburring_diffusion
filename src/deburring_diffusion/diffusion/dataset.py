from collections import defaultdict
import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"


class MotionDataset(Dataset):
    def __init__(self, data_file: Path, device="cuda"):
        """
        Dataset that returns robot motion data.

        Args:
            data_file (str or Path): Path of the file containing data.
            device (str): Device for tensor conversion.
        """
        super().__init__()
        self.device = device
        self.data_file = Path(data_file)

        cache_dir = self.data_file.parent / "processed_datasets"
        cache_dir.mkdir(exist_ok=True)

        file_samples = cache_dir / f"{self.data_file.name}_cache_samples.npy"
        file_q0 = cache_dir / f"{self.data_file.name}_cache_q0.npy"
        file_goal = cache_dir / f"{self.data_file.name}_cache_goal.npy"

        # Load or parse data
        if all(f.exists() for f in [file_samples, file_q0, file_goal]):
            self.samples = torch.tensor(np.load(file_samples), dtype=torch.float32)
            self.q0 = torch.tensor(np.load(file_q0), dtype=torch.float32)
            self.goal = torch.tensor(np.load(file_goal), dtype=torch.float32)
        else:
            with open(data_file, "r") as f:
                data = json.load(f)

            self.samples = torch.tensor(np.asarray([res["trajectory"][1:] for res in data]), dtype=torch.float32)
            self.q0 = torch.tensor(np.asarray([res["trajectory"][0] for res in data]), dtype=torch.float32)
            self.goal = torch.tensor(np.asarray([res["target"] for res in data]), dtype=torch.float32)


            np.save(file_samples, self.samples.numpy())
            np.save(file_q0, self.q0.numpy())
            np.save(file_goal, self.goal.numpy())
        
        # Stats
        self.samples_mean = self.samples.mean(dim=(0, 1), keepdim=True)
        self.samples_std = self.samples.std(dim=(0, 1), keepdim=True)
        self.q0_mean = self.q0.mean(dim=0, keepdim=True)
        self.q0_std = self.q0.std(dim=0, keepdim=True)
        self.goal_mean = self.goal.mean(dim=0, keepdim=True)
        self.goal_std = self.goal.std(dim=0, keepdim=True)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):

        return self.samples[idx], {
            "q0": self.q0[idx],
            "goal": self.goal[idx],
        }


class DataModule(pl.LightningDataModule):
    """Defines the data used for training the diffusion process."""

    def __init__(self, data_file: Path) -> None:
        super().__init__()


        self.dataset = MotionDataset(
           device=device, data_file=data_file
        )

        _, self.seq_length, self.configuration_size = self.dataset.samples.shape

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=16,
            num_workers=4,
            shuffle=True,
        )