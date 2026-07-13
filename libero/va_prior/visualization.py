"""Utilities for visualizing sampled action chunks during training."""

import json
from pathlib import Path

import numpy as np


def integrate_action_chunks(chunks, origin=None, position_scale=1.0):
    """Integrate the first three action DOFs into XYZ trajectories."""
    chunks = np.asarray(chunks)
    if chunks.ndim != 3:
        raise ValueError(f"Expected [K, H, D] chunks, got {chunks.shape}")
    if chunks.shape[-1] < 3:
        raise ValueError("At least three continuous action DOFs are required for XYZ trajectories")
    deltas = chunks[..., :3] * np.asarray(position_scale, dtype=chunks.dtype)
    if origin is None:
        origin = np.zeros(3, dtype=chunks.dtype)
    origin = np.asarray(origin, dtype=chunks.dtype).reshape(1, 1, 3)
    origin = np.broadcast_to(origin, (chunks.shape[0], 1, 3))
    positions = origin + np.cumsum(deltas, axis=1)
    return np.concatenate([origin, positions], axis=1)


def scale_controller_actions(actions, controller_geometry):
    """Apply robosuite's clipped affine controller scaling without mutating a simulator."""
    actions = np.asarray(actions)
    control_dim = int(controller_geometry["control_dim"])
    pose = actions[..., :control_dim]
    clipped = np.clip(pose, controller_geometry["input_min"],
                      controller_geometry["input_max"])
    scaled = ((clipped - controller_geometry["input_transform"])
              * controller_geometry["action_scale"]
              + controller_geometry["output_transform"])
    return scaled[..., :3]


def project_world_points(points, world_to_camera, image_height, image_width):
    """Project world XYZ points with robosuite's homogeneous camera matrix."""
    points = np.asarray(points)
    matrix = np.asarray(world_to_camera)
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    projected = homogeneous @ matrix.T
    depth = projected[..., 2]
    pixels = projected[..., :2] / np.where(np.abs(depth[..., None]) > 1e-8,
                                           depth[..., None], np.nan)
    finite = np.isfinite(pixels).all(-1) & np.isfinite(depth) & (depth > 1e-8)
    in_frame = (finite & (pixels[..., 0] >= 0) & (pixels[..., 0] < image_width)
                & (pixels[..., 1] >= 0) & (pixels[..., 1] < image_height))
    return pixels, finite, in_frame


def trajectory_overlay_figure(image, candidate_deltas, weights, target_deltas,
                              target_mask, ee_pos, world_to_camera, title=None):
    """Overlay candidate and ground-truth XYZ trajectories on an agentview image."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image[:3], 0, -1)
    if image.dtype.kind == "f":
        image = np.clip(image, 0, 1)
    height, width = image.shape[:2]
    candidate_deltas = np.asarray(candidate_deltas)
    weights = np.asarray(weights)
    valid_candidates = weights > 0
    candidate_deltas, weights = candidate_deltas[valid_candidates], weights[valid_candidates]
    if not len(candidate_deltas):
        raise ValueError("No positive-weight action chunks to visualize")

    candidate_paths = integrate_action_chunks(candidate_deltas, origin=ee_pos)
    target_mask = np.asarray(target_mask, dtype=bool)
    valid_target = np.asarray(target_deltas)[target_mask]
    if not len(valid_target):
        raise ValueError("Ground-truth action chunk has no valid timesteps")
    target_path = integrate_action_chunks(valid_target[None], origin=ee_pos)[0]
    candidate_pixels, candidate_front, _ = project_world_points(
        candidate_paths, world_to_camera, height, width)
    target_pixels, target_front, _ = project_world_points(
        target_path, world_to_camera, height, width)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image)
    colors = plt.get_cmap("tab10")
    max_weight = max(float(weights.max()), 1e-8)
    for index, (pixels, front, weight) in enumerate(
            zip(candidate_pixels, candidate_front, weights)):
        draw = pixels.copy()
        draw[~front] = np.nan
        alpha = 0.35 + 0.55 * float(weight) / max_weight
        line, = ax.plot(draw[:, 0], draw[:, 1], color=colors(index % 10), linewidth=2,
                        marker="o", markersize=3, alpha=alpha,
                        label=f"candidate {index} (p={weight:.2f})")
        line.set_path_effects([path_effects.Stroke(linewidth=3.5, foreground="black", alpha=.5),
                               path_effects.Normal()])
        visible = np.where(front)[0]
        if len(visible):
            end = pixels[visible[-1]]
            ax.scatter(end[0], end[1], color=colors(index % 10), marker="x", s=35,
                       linewidth=2, zorder=5)

    target_draw = target_pixels.copy()
    target_draw[~target_front] = np.nan
    gt, = ax.plot(target_draw[:, 0], target_draw[:, 1], color="#00ff66", linewidth=3,
                  linestyle="--", marker="s", markersize=4, label="ground truth", zorder=8)
    gt.set_path_effects([path_effects.Stroke(linewidth=5, foreground="black", alpha=.8),
                         path_effects.Normal()])
    if target_front.any():
        target_end = target_pixels[np.where(target_front)[0][-1]]
        ax.scatter(target_end[0], target_end[1], color="#00ff66", marker="X", s=70,
                   edgecolor="black", linewidth=.8, zorder=9)

    if candidate_front[0, 0]:
        start = candidate_pixels[0, 0]
        ax.scatter(start[0], start[1], color="yellow", marker="*", s=150,
                   edgecolor="black", linewidth=1, label="current end effector", zorder=10)
    ax.set_xlim(0, width - 1)
    ax.set_ylim(height - 1, 0)
    ax.set_axis_off()
    ax.set_title(title or "Projected action candidates and ground truth")
    ax.legend(loc="upper left", fontsize=7, framealpha=.75)
    fig.tight_layout(pad=.2)
    return fig


def _decode_attribute(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _resolve_bddl_file(data_group, env_meta, cache_dir):
    configured = (env_meta.get("bddl_file")
                  or env_meta.get("env_kwargs", {}).get("bddl_file_name")
                  or _decode_attribute(data_group.attrs.get("bddl_file_name", "")))
    if configured and Path(configured).exists():
        return Path(configured)
    filename = Path(configured).name if configured else None
    benchmark_root = Path(__file__).resolve().parents[1] / "libero" / "bddl_files"
    matches = list(benchmark_root.rglob(filename)) if filename else []
    if matches:
        return matches[0]
    content = _decode_attribute(data_group.attrs.get("bddl_file_content", ""))
    if not content or not filename:
        raise FileNotFoundError(f"Could not resolve BDDL file {configured!r}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / filename
    if not destination.exists():
        destination.write_text(content)
    return destination


def restore_projection_geometry(dataset_path, demo_id, timestep, image_height,
                                image_width, cache_dir):
    """Restore one demonstration state and extract camera/controller geometry."""
    import h5py
    from robosuite.utils import camera_utils
    from libero.libero.envs import TASK_MAPPING
    from libero.libero.utils import utils as libero_utils

    with h5py.File(dataset_path, "r") as dataset:
        data = dataset["data"]
        env_meta = json.loads(_decode_attribute(data.attrs["env_args"]))
        bddl_file = _resolve_bddl_file(data, env_meta, cache_dir)
        env_kwargs = dict(env_meta["env_kwargs"])
        env_kwargs.update({
            "bddl_file_name": str(bddl_file),
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
        })
        group = data[demo_id]
        model_xml = _decode_attribute(group.attrs["model_file"])
        model_xml = libero_utils.postprocess_model_xml(model_xml, {})
        state = np.asarray(group["states"][int(timestep)])

    env = TASK_MAPPING[env_meta["problem_name"]](**env_kwargs)
    try:
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        camera = camera_utils.get_camera_transform_matrix(
            env.sim, "agentview", image_height, image_width)
        controller = env.robots[0].controller
        control_dim = int(controller.control_dim)
        controller.scale_action(np.zeros(control_dim, dtype=np.float32))
        geometry = {
            "world_to_camera": np.asarray(camera),
            "control_dim": control_dim,
            "input_min": np.asarray(controller.input_min),
            "input_max": np.asarray(controller.input_max),
            "action_scale": np.asarray(controller.action_scale),
            "input_transform": np.asarray(controller.action_input_transform),
            "output_transform": np.asarray(controller.action_output_transform),
        }
    finally:
        env.close()
    return geometry


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
