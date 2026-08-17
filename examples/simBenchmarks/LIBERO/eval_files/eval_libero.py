import dataclasses
import json
import logging
import math
import os
import pathlib
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient
from examples.simBenchmarks.LIBERO.eval_files.classifier_language_rollout import (
    POSITIVE_VARIANTS,
    new_rollout_result,
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"
    rollout_manifest: str = ""  # JSON produced by classifier_language_rollout.py.
    instruction_variant: str = "canonical"
    rollout_phase: str = "libero_original"
    result_json: str = ""
    # The released Qwen2.5 GR00T 4-in-1 data config packs primary then wrist.
    image_views: str = "primary,wrist"
    # Its training config does not opt into LeRobot state packing.
    use_state: bool = False


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        seed=args.seed,
    )

    manifest_entries = []
    if args.rollout_manifest:
        manifest = json.loads(pathlib.Path(args.rollout_manifest).read_text(encoding="utf-8"))
        manifest_entries = [
            row for row in manifest["entries"]
            if row["suite"] == args.task_suite_name and row["variant_id"] == args.instruction_variant
        ]
        if not manifest_entries:
            raise ValueError(
                f"manifest has no {args.task_suite_name}/{args.instruction_variant} entries"
        )
        eval_task_ids = sorted({int(row["task_index"]) for row in manifest_entries})
        manifest_by_task = {}
        for row in manifest_entries:
            manifest_by_task.setdefault(int(row["task_index"]), []).append(row)
    else:
        eval_task_ids = list(range(num_tasks_in_suite))
        manifest_by_task = {}

    # Optional smoke-test cap (still useful for quick verification with -1 = full run).
    if args.max_tasks > 0:
        eval_task_ids = eval_task_ids[: args.max_tasks]
    n_eval_tasks = len(eval_task_ids)
    logging.info(f"Evaluating {n_eval_tasks} of {num_tasks_in_suite} tasks (max_tasks={args.max_tasks})")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    rollout_episodes = []
    for task_id in tqdm.tqdm(eval_task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, canonical_task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_manifest_rows = manifest_by_task.get(task_id, [])
        task_description = (
            task_manifest_rows[0]["instruction"] if task_manifest_rows else canonical_task_description
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        episode_indices = (
            sorted({int(row["initial_state_index"]) for row in task_manifest_rows})
            if task_manifest_rows
            else list(range(args.num_trials_per_task))
        )
        for episode_idx in tqdm.tqdm(episode_indices):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            request_seed = args.seed * 1_000_000 + task_id * 1_000 + episode_idx
            client_model.reset(task_description=task_description, request_seed=request_seed)
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []
            request_diagnostics = {}

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Save preprocessed image for replay video
                replay_images.append(img)

                gripper_q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        gripper_q[:1],
                    )
                ).astype(np.float32)
                if state.shape != (7,):
                    raise ValueError(f"Expected LIBERO 7-D state, got {state.shape}")

                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # Keep camera count and ordering aligned with the checkpoint's training data.
                if args.image_views == "primary":
                    model_images = [observation["observation.primary"][0]]
                elif args.image_views == "primary,wrist":
                    model_images = [
                        observation["observation.primary"][0],
                        observation["observation.wrist_image"][0],
                    ]
                else:
                    raise ValueError(
                        "image_views must be 'primary' or 'primary,wrist', "
                        f"got {args.image_views!r}"
                    )
                example_dict = {
                    "image": model_images,
                    "lang": observation["instruction"][0],
                }
                if args.use_state:
                    example_dict["state"] = observation["observation.state"]

                start_time = time.time()

                response = client_model.step(example=example_dict, step=step)
                if response.get("request_latency_ms") is not None:
                    request_diagnostics.setdefault("client_request_latency_ms", []).append(
                        float(response["request_latency_ms"])
                    )
                diagnostic = response.get("classifier_diagnostics")
                if diagnostic is not None:
                    request_diagnostics[diagnostic.get("seed", len(request_diagnostics))] = diagnostic

                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

            client_latencies = request_diagnostics.pop("client_request_latency_ms", [])
            diagnostics = list(request_diagnostics.values())
            selected_logits = []
            for diagnostic in diagnostics:
                for logits, selected in zip(
                    diagnostic.get("candidate_logits", []), diagnostic.get("selected_indices", [])
                ):
                    selected_logits.append(float(logits[int(selected)]))
            rollout_episodes.append({
                "episode_id": f"{args.task_suite_name}:{task_id}:{episode_idx}",
                "pair_id": f"{args.task_suite_name}:{task_id}:{episode_idx}",
                "suite": args.task_suite_name,
                "task_index": task_id,
                "initial_state_index": episode_idx,
                "variant_id": args.instruction_variant,
                "instruction": task_description,
                "positive_instruction": args.instruction_variant in POSITIVE_VARIANTS,
                "success": bool(done),
                "latency_ms": (
                    float(np.mean([
                        row["inference_latency_ms"] for row in diagnostics if "inference_latency_ms" in row
                    ]))
                    if diagnostics
                    else (float(np.mean(client_latencies)) if client_latencies else None)
                ),
                "classifier_score": float(1.0 / (1.0 + np.exp(-np.mean(selected_logits)))) if selected_logits else None,
                "classifier_diagnostics": diagnostics,
                "action_trajectory": full_actions.tolist(),
            })

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    if args.result_json:
        metadata = client_model._server_metadata
        result = new_rollout_result(
            phase=args.rollout_phase,
            mode=metadata.get("classifier_mode", "off"),
            base_checkpoint=metadata.get("base_ckpt_path", metadata.get("ckpt_path", "")),
            classifier_checkpoint=metadata.get("classifier_ckpt_path"),
            seed=args.seed,
            num_candidates=int(metadata.get("num_candidates", 1)),
            guidance_scale=float(metadata.get("guidance_scale", 0.0)),
            episodes=rollout_episodes,
        )
        result_path = pathlib.Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
