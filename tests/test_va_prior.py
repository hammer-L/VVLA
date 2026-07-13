import h5py
import numpy as np
import sys
import torch
from types import SimpleNamespace

from libero.va_prior.data import (ActionStats, LiberoActionChunkDataset,
                                  compute_action_stats_multi, trajectory_split)
from libero.va_prior.model import VAPriorModel
from libero.va_prior.visualization import (action_chunk_figure, action_distributions,
                                           integrate_action_chunks, project_world_points,
                                           scale_controller_actions,
                                           trajectory_overlay_figure)
from scripts.va_prior_experiment import log_rollout_to_wandb, validation_candidate_metrics


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
            obs.create_dataset("ee_states", data=np.zeros((length, 6), dtype="float32"))
            obs.create_dataset("ee_pos", data=np.zeros((length, 3), dtype="float32"))


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
        counts = []
        for head_dim in (256, 512, 1024):
            model = VAPriorModel(head, "tiny", 9, 6, horizon=3, hidden_dim=32,
                                 action_head_dim=head_dim)
            loss, _ = model.loss(batch)
            assert torch.isfinite(loss)
            result = model.candidates(batch, k=4, flow_steps=2)
            assert result["candidate_chunks"].shape == (
                2, 4 if head != "deterministic" else 1, 3, 7)
            assert torch.allclose(result["prior_weights"].sum(-1), torch.ones(2))
            assert result["candidate_features"].shape[-1] == 32
            raw = model.candidates(batch, k=4, flow_steps=2, cluster=False)
            expected = 4 if head != "deterministic" else 1
            assert raw["candidate_chunks"].shape == (2, expected, 3, 7)
            assert torch.all(raw["prior_weights"] == 1 / expected)
            counts.append(model.action_head_parameter_count())
        assert counts[0] < counts[1] < counts[2]


def test_base_action_head_keeps_legacy_parameter_shapes():
    legacy = VAPriorModel("flow", "tiny", 9, 6, horizon=3, hidden_dim=32)
    explicit_base = VAPriorModel("flow", "tiny", 9, 6, horizon=3, hidden_dim=32,
                                 action_head_dim=512)
    explicit_base.load_state_dict(legacy.state_dict())
    assert legacy.action_head_dim == explicit_base.action_head_dim == 512


def test_integrate_action_chunks_starts_at_shared_origin():
    chunks = np.asarray([[[1, 0, 0], [0, 2, 0]],
                         [[0, 0, 3], [1, 1, 1]]], dtype=np.float32)
    trajectories = integrate_action_chunks(chunks)
    assert trajectories.shape == (2, 3, 3)
    np.testing.assert_array_equal(trajectories[:, 0], np.zeros((2, 3)))
    np.testing.assert_array_equal(trajectories[0, -1], [1, 2, 0])
    np.testing.assert_array_equal(trajectories[1, -1], [1, 1, 4])


def test_action_distributions_ignore_padding_and_zero_weight_candidates():
    candidates = np.asarray([[[[1, 2, 3, -1], [4, 5, 6, 1]],
                              [[9, 9, 9, 1], [9, 9, 9, 1]]]], dtype=np.float32)
    values = action_distributions(
        candidates, np.asarray([[1, 0]], dtype=np.float32),
        np.asarray([[[10, 11, 12], [20, 21, 22]]], dtype=np.float32),
        np.asarray([[0, 1]], dtype=np.float32), np.asarray([[True, False]]))
    np.testing.assert_array_equal(values["predicted_continuous"], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(values["target_continuous"], [[10, 11, 12]])
    np.testing.assert_array_equal(values["predicted_gripper"], [-1, 1])
    np.testing.assert_array_equal(values["target_gripper"], [-1])


def test_action_chunk_figure_falls_back_below_three_continuous_dofs():
    figure = action_chunk_figure(np.zeros((1, 3, 3), dtype=np.float32), np.ones(1))
    assert len(figure.axes) == 2
    assert "XYZ unavailable" in figure.axes[0].get_title()
    import matplotlib.pyplot as plt
    plt.close(figure)


def test_controller_scaling_projection_and_overlay(tmp_path):
    geometry = {
        "control_dim": 6,
        "input_min": np.full(6, -1, dtype=np.float32),
        "input_max": np.full(6, 1, dtype=np.float32),
        "action_scale": np.full(6, .05, dtype=np.float32),
        "input_transform": np.zeros(6, dtype=np.float32),
        "output_transform": np.zeros(6, dtype=np.float32),
    }
    actions = np.asarray([[[1, 0, 0, 0, 0, 0, -1],
                           [2, 1, 0, 0, 0, 0, 1]]], dtype=np.float32)
    deltas = scale_controller_actions(actions, geometry)
    np.testing.assert_allclose(deltas[0], [[.05, 0, 0], [.05, .05, 0]])

    camera = np.asarray([[100, 0, 50, 0], [0, 100, 50, 0],
                         [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    pixels, front, in_frame = project_world_points(
        np.asarray([[0, 0, 1], [.1, 0, 1]], dtype=np.float32), camera, 100, 100)
    np.testing.assert_allclose(pixels, [[50, 50], [60, 50]])
    assert front.all() and in_frame.all()

    figure = trajectory_overlay_figure(
        np.zeros((100, 100, 3), dtype=np.uint8), deltas, np.ones(1), deltas[0],
        np.asarray([True, True]), np.asarray([0, 0, 1], dtype=np.float32), camera)
    path = tmp_path / "overlay.png"
    figure.savefig(path)
    assert path.stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close(figure)


def test_validation_acc_is_macro_best_of_k_recall():
    class DummyPrior:
        action_mean = torch.zeros(3)
        action_std = torch.ones(3)

        def candidates(self, batch, *_args):
            batch_size, horizon = batch["continuous"].shape[:2]
            chunks = torch.zeros(batch_size, 1, horizon, 4)
            return {"candidate_chunks": chunks, "prior_weights": torch.ones(batch_size, 1)}

    raw = {
        "continuous": torch.tensor([[[0., 0., 0.], [9., 9., 9.]],
                                    [[1., 0., 0.], [9., 9., 9.]]]),
        "mask": torch.tensor([[True, False], [True, False]]),
        "task_id": torch.tensor([0, 1]),
    }
    args = SimpleNamespace(device="cpu", seed=3, val_k=4, val_flow_steps=2,
                           val_cluster_threshold=1.0)
    metrics = validation_candidate_metrics(
        DummyPrior(), [raw], ActionStats(np.zeros(3), np.ones(3), .5), args)
    assert metrics["acc"] == .5
    assert metrics["acc_by_goal"] == {0: 1.0, 1: 0.0}
    assert metrics["best_of_k_ade"] == .5


def test_rollout_wandb_resumes_training_run(monkeypatch, tmp_path):
    events = {}
    fake_wandb = SimpleNamespace()

    def init(**kwargs):
        events["init"] = kwargs
        fake_wandb.run = SimpleNamespace(summary={})

    fake_wandb.init = init
    fake_wandb.log = lambda values: events.setdefault("log", values)
    fake_wandb.finish = lambda: events.setdefault("finished", True)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    args = SimpleNamespace(
        checkpoint=tmp_path / "best.pt", bddl_file=tmp_path / "goal.bddl",
        wandb_project=None, wandb_entity=None, wandb_mode=None, wandb_run_name=None)
    cfg = SimpleNamespace(wandb_run_id="training-run", wandb_project="project",
                          wandb_mode="offline", wandb_entity=None, wandb_run_name="train")
    report = {"success_rate": .25, "success_rate_std_across_init_states": .05,
              "mean_steps": 12.0, "saturation_rate": .1,
              "collision_rate": None, "rollouts": 4}
    log_rollout_to_wandb(report, args, cfg)
    assert events["init"]["id"] == "training-run"
    assert events["init"]["resume"] == "allow"
    assert events["log"]["rollout/acc"] == .25
    assert fake_wandb.run.summary["rollout/by_goal/goal/success_rate"] == .25
    assert events["finished"]
