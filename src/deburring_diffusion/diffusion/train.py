import torch
from pytorch_lightning.cli import LightningCLI
from deburring_diffusion.diffusion.dataset import DataModule
from deburring_diffusion.diffusion.model import Model


if __name__ == "__main__":
    cli = LightningCLI(
        Model,
        datamodule_class=DataModule,
        trainer_defaults={
            "log_every_n_steps": 1,
            "max_epochs": 3,
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
        },
    )
    
    
    