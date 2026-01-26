from pathlib import Path
import torch

from deburring_diffusion.diffusion.dataset import MotionDataset, DataModule


def test_motion_dataset_init(dummy_motion_data):

    dataset = MotionDataset(dummy_motion_data, device="cpu")

    assert len(dataset) == 10
    assert dataset.samples.ndim == 3
    assert dataset.q0.ndim == 2
    assert dataset.goal.ndim == 2

def test_motion_dataset_shapes(dummy_motion_data):

    dataset = MotionDataset(dummy_motion_data, device="cpu")

    samples, meta = dataset[0]

    # trajectory[1:] so T-1
    assert samples.shape == (4, 7)
    assert meta["q0"].shape == (7,)
    assert meta["goal"].shape == (7,)


def test_motion_dataset_creates_cache(dummy_motion_data):

    dataset = MotionDataset(dummy_motion_data, device="cpu")

    cache_dir = dummy_motion_data.parent / "processed_datasets"
    assert cache_dir.exists()

    expected = [
        f"{dummy_motion_data.name}_cache_samples.npy",
        f"{dummy_motion_data.name}_cache_q0.npy",
        f"{dummy_motion_data.name}_cache_goal.npy",
    ]

    for fname in expected:
        assert (cache_dir / fname).exists()


def test_motion_dataset_cache_consistency(dummy_motion_data):

    ds1 = MotionDataset(dummy_motion_data, device="cpu")
    ds2 = MotionDataset(dummy_motion_data, device="cpu")

    assert torch.allclose(ds1.samples, ds2.samples)
    assert torch.allclose(ds1.q0, ds2.q0)
    assert torch.allclose(ds1.goal, ds2.goal)
    

def test_motion_dataset_stats(dummy_motion_data):

    dataset = MotionDataset(dummy_motion_data, device="cpu")

    assert dataset.samples_mean.shape == (1, 1, 7)
    assert dataset.samples_std.shape == (1, 1, 7)

    assert dataset.q0_mean.shape == (1, 7)
    assert dataset.q0_std.shape == (1, 7)

    assert dataset.goal_mean.shape == (1, 7)
    assert dataset.goal_std.shape == (1, 7)


def test_datamodule_dataloader(dummy_motion_data):

    dm = DataModule(dummy_motion_data)
    loader = dm.train_dataloader()

    batch = next(iter(loader))
    samples, meta = batch

    assert samples.shape[0] == 16 or samples.shape[0] == len(dm.dataset)
    assert samples.ndim == 3
    assert "q0" in meta
    assert "goal" in meta

def test_dataset_shuffle_stability(dummy_motion_data):

    ds1 = MotionDataset(dummy_motion_data, device="cpu")
    ds2 = MotionDataset(dummy_motion_data, device="cpu")

    # Order can differ, but content must match
    assert sorted(ds1.q0.tolist()) == sorted(ds2.q0.tolist())
