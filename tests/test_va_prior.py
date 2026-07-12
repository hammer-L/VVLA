import h5py
import numpy as np
import torch

from libero.va_prior.data import LiberoActionChunkDataset, compute_action_stats_multi, trajectory_split
from libero.va_prior.model import VAPriorModel


def make_dataset(path, demos=10, length=6):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i in range(demos):
            demo = data.create_group(f"demo_{i}")
            action = np.random.randn(length, 7).astype("float32")
            action[:, -1] = np.where(np.arange(length) < length // 2, -1, 1)
            demo.create_dataset("actions", data=action)
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=np.zeros((length, 16, 16, 3), dtype="uint8"))
            obs.create_dataset("eye_in_hand_rgb", data=np.zeros((length, 16, 16, 3), dtype="uint8"))
            obs.create_dataset("joint_states", data=np.zeros((length, 7), dtype="float32"))
            obs.create_dataset("gripper_states", data=np.zeros((length, 2), dtype="float32"))


def test_trajectory_split_and_padding(tmp_path):
    path = tmp_path / "demo.hdf5"; make_dataset(path)
    split = trajectory_split(path, seed=3)
    assert len(split["train"]) == 8 and len(split["val"]) == 1 and len(split["test"]) == 1
    assert not (set(split["train"]) & set(split["test"]))
    stats = compute_action_stats_multi([path], {str(path): split["train"]})
    ds = LiberoActionChunkDataset(path, split["train"], stats, horizon=10)
    item = ds[len(ds) - 1]
    assert item["continuous"].shape == (10, 6)
    assert item["mask"].sum() == 1


def test_all_heads_candidate_contract():
    batch = {"images": torch.rand(2, 2, 2, 3, 32, 32), "proprio": torch.rand(2, 2, 9),
             "continuous": torch.rand(2, 3, 6), "gripper": torch.randint(0, 2, (2, 3)).float()}
    for head in ("deterministic", "gmm", "flow"):
        model = VAPriorModel(head, "tiny", 9, 6, horizon=3, hidden_dim=32)
        loss, _ = model.loss(batch)
        assert torch.isfinite(loss)
        result = model.candidates(batch, k=4, flow_steps=2)
        assert result["candidate_chunks"].shape == (2, 4 if head != "deterministic" else 1, 3, 7)
        assert torch.allclose(result["prior_weights"].sum(-1), torch.ones(2))
        assert result["candidate_features"].shape[-1] == 32
