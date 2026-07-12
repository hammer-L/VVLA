import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


CAMERA_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
PROPRIO_KEYS = ("joint_states", "gripper_states", "ee_states")


def trajectory_split(hdf5_path, seed=0, ratios=(0.8, 0.1, 0.1)):
    """Return a deterministic trajectory-level split (never a frame split)."""
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
    rng = np.random.RandomState(seed)
    rng.shuffle(demos)
    n = len(demos)
    n_train = max(1, int(n * ratios[0]))
    n_val = max(1, int(n * ratios[1])) if n >= 3 else 0
    n_train = min(n_train, n - n_val - (1 if n >= 2 else 0))
    return {
        "train": demos[:n_train],
        "val": demos[n_train : n_train + n_val],
        "test": demos[n_train + n_val :],
    }


@dataclass
class ActionStats:
    mean: np.ndarray
    std: np.ndarray
    recall_threshold: float

    def save(self, path):
        Path(path).write_text(json.dumps({
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "recall_threshold": self.recall_threshold,
        }, indent=2))

    @classmethod
    def load(cls, path):
        x = json.loads(Path(path).read_text())
        return cls(np.asarray(x["mean"], np.float32), np.asarray(x["std"], np.float32), x["recall_threshold"])


def compute_action_stats(hdf5_path, demos):
    chunks, adjacent_diffs = [], []
    with h5py.File(hdf5_path, "r") as f:
        for demo in demos:
            action = np.asarray(f[f"data/{demo}/actions"][:, :-1], np.float32)
            chunks.append(action)
    actions = np.concatenate(chunks)
    mean = actions.mean(0)
    std = np.maximum(actions.std(0), 1e-3)
    for action in chunks:
        normalized_demo = (action - mean) / std
        if len(action) > 1:
            adjacent_diffs.append(np.linalg.norm(np.diff(normalized_demo, axis=0), axis=-1))
    # A data-derived tolerance: median distance between adjacent demonstrated actions.
    diffs = np.concatenate(adjacent_diffs) if adjacent_diffs else np.asarray([0.25])
    threshold = float(max(np.median(diffs) * 2.0, 0.25))
    return ActionStats(mean.astype(np.float32), std.astype(np.float32), threshold)


class LiberoActionChunkDataset(Dataset):
    def __init__(self, hdf5_path, demos, stats, horizon=10, camera_keys=CAMERA_KEYS,
                 proprio_keys=PROPRIO_KEYS, obs_horizon=2):
        self.path = str(hdf5_path)
        self.demos = list(demos)
        self.stats = stats
        self.horizon = horizon
        self.camera_keys = camera_keys
        self.proprio_keys = proprio_keys
        self.obs_horizon = obs_horizon
        self.index = []
        self._file = None
        with h5py.File(self.path, "r") as f:
            for demo in self.demos:
                n = len(f[f"data/{demo}/actions"])
                self.index.extend((demo, t) for t in range(n))

    def __len__(self):
        return len(self.index)

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def __getitem__(self, idx):
        demo, t = self.index[idx]
        group = self.file[f"data/{demo}"]
        obs = group["obs"]
        times = np.maximum(np.arange(t - self.obs_horizon + 1, t + 1), 0)
        image_sequence, proprio_sequence = [], []
        for obs_t in times:
            images = []
            for key in self.camera_keys:
                x = np.asarray(obs[key][obs_t])
                if x.shape[-1] in (1, 3, 4):
                    x = np.moveaxis(x[..., :3], -1, 0)
                images.append(torch.from_numpy(x.copy()).float().div(255.0))
            proprio = []
            for key in self.proprio_keys:
                if key in obs:
                    proprio.append(np.asarray(obs[key][obs_t], np.float32).reshape(-1))
            image_sequence.append(torch.stack(images))
            proprio_sequence.append(torch.from_numpy(np.concatenate(proprio)))
        actions = np.asarray(group["actions"])
        end = min(t + self.horizon, len(actions))
        chunk = actions[t:end]
        valid = len(chunk)
        if valid < self.horizon:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], self.horizon - valid, axis=0)])
        continuous = (chunk[:, :-1] - self.stats.mean) / self.stats.std
        gripper = (chunk[:, -1] > 0).astype(np.float32)
        progress = t / max(len(actions) - 1, 1)
        event = bool(t > 0 and np.sign(actions[t, -1]) != np.sign(actions[t - 1, -1]))
        return {
            "images": torch.stack(image_sequence),
            "proprio": torch.stack(proprio_sequence),
            "continuous": torch.from_numpy(continuous.astype(np.float32)),
            "gripper": torch.from_numpy(gripper),
            "mask": torch.arange(self.horizon) < valid,
            "progress": torch.tensor(progress, dtype=torch.float32),
            "gripper_event": torch.tensor(event),
            "index": idx,
        }
