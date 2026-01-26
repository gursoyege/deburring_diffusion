import json
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def dummy_motion_data(tmp_path):
    data = []
    N = 10
    T = 5
    nq = 7

    for _ in range(N):
        traj = np.random.randn(T, nq).tolist()
        target = np.random.randn(nq).tolist()
        data.append({
            "trajectory": traj,
            "target": target
        })

    file_path = tmp_path / "data.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    return file_path

