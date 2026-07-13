"""Utilities for visualizing sampled action chunks during training."""

import numpy as np


def integrate_action_chunks(chunks):
    """Integrate the first three action DOFs into relative XYZ trajectories."""
    chunks = np.asarray(chunks)
    if chunks.ndim != 3:
        raise ValueError(f"Expected [K, H, D] chunks, got {chunks.shape}")
    if chunks.shape[-1] < 3:
        raise ValueError("At least three continuous action DOFs are required for XYZ trajectories")
    positions = np.cumsum(chunks[..., :3], axis=1)
    origin = np.zeros((chunks.shape[0], 1, 3), dtype=positions.dtype)
    return np.concatenate([origin, positions], axis=1)


def action_distributions(candidate_chunks, prior_weights, target_continuous,
                         target_gripper, mask):
    """Flatten valid predictions and masked targets into comparable per-DOF arrays."""
    candidates = np.asarray(candidate_chunks)
    weights = np.asarray(prior_weights)
    target = np.asarray(target_continuous)
    target_gripper = np.asarray(target_gripper)
    mask = np.asarray(mask, dtype=bool)
    if candidates.ndim != 4:
        raise ValueError(f"Expected [B, K, H, D] candidates, got {candidates.shape}")

    predicted = candidates[weights > 0]
    return {
        "predicted_continuous": predicted[..., :-1].reshape(-1, candidates.shape[-1] - 1),
        "target_continuous": target[mask],
        "predicted_gripper": predicted[..., -1].reshape(-1),
        "target_gripper": (target_gripper[mask] * 2.0 - 1.0).reshape(-1),
    }


def action_chunk_figure(chunks, weights):
    """Create a 3D relative trajectory plot plus gripper-state traces."""
    import matplotlib.pyplot as plt

    chunks = np.asarray(chunks)
    weights = np.asarray(weights)
    valid = weights > 0
    chunks, weights = chunks[valid], weights[valid]
    if not len(chunks):
        raise ValueError("No positive-weight action chunks to visualize")

    if chunks.shape[-1] - 1 >= 3:
        fig = plt.figure(figsize=(12, 5))
        trajectory_ax = fig.add_subplot(1, 2, 1, projection="3d")
        trajectories = integrate_action_chunks(chunks[..., :-1])
        for index, (trajectory, weight) in enumerate(zip(trajectories, weights)):
            trajectory_ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                               marker="o", markersize=2,
                               label=f"chunk {index} (p={weight:.2f})")
        trajectory_ax.scatter([0], [0], [0], color="black", marker="x", s=40)
        trajectory_ax.set_xlabel("relative x")
        trajectory_ax.set_ylabel("relative y")
        trajectory_ax.set_zlabel("relative z")
        trajectory_ax.set_title("Integrated end-effector delta trajectories")
    else:
        fig = plt.figure(figsize=(12, 5))
        trajectory_ax = fig.add_subplot(1, 2, 1)
        for index, (chunk, weight) in enumerate(zip(chunks, weights)):
            for dof in range(chunks.shape[-1] - 1):
                trajectory_ax.plot(chunk[:, dof],
                                   label=f"chunk {index} dof {dof} (p={weight:.2f})")
        trajectory_ax.set_xlabel("horizon step")
        trajectory_ax.set_ylabel("action")
        trajectory_ax.set_title("Action DOF traces (XYZ unavailable)")

    trajectory_ax.legend(fontsize=7)
    gripper_ax = fig.add_subplot(1, 2, 2)
    for index, (chunk, weight) in enumerate(zip(chunks, weights)):
        gripper_ax.step(np.arange(len(chunk)), chunk[:, -1], where="post",
                        label=f"chunk {index} (p={weight:.2f})")
    gripper_ax.set_xlabel("horizon step")
    gripper_ax.set_ylabel("gripper action")
    gripper_ax.set_yticks([-1, 1])
    gripper_ax.set_title("Gripper state")
    gripper_ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
