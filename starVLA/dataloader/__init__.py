import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(
    cfg,
    dataset_py="lerobot_datasets_oxe",
    *,
    overlay_split=None,
    overlay_mode=None,
): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
        )
        from starVLA.dataloader.language_overlay import wrap_language_overlay
        vla_dataset = wrap_language_overlay(
            vla_dataset,
            vla_dataset_cfg,
            split=overlay_split,
            mode=overlay_mode,
        )
        num_workers = int(vla_dataset_cfg.get("num_workers", 4))
        dataloader_kwargs = {
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": bool(vla_dataset_cfg.get("pin_memory", True)),
            # shuffle=True
        }
        if num_workers > 0:
            # Overlay epoch rotation lives in dataset state. Non-persistent
            # workers are recreated after set_epoch and observe that state.
            overlay_enabled = bool(vla_dataset_cfg.get("language_overlay_meta", None))
            dataloader_kwargs["persistent_workers"] = False if overlay_enabled else bool(vla_dataset_cfg.get("persistent_workers", True))
            dataloader_kwargs["prefetch_factor"] = int(vla_dataset_cfg.get("prefetch_factor", 2))

        if getattr(vla_dataset, "mode", None) == "contrastive_train":
            from starVLA.dataloader.language_overlay import GroupedDistributedBatchSampler

            dataloader_kwargs["batch_sampler"] = GroupedDistributedBatchSampler(
                vla_dataset,
                batch_size=int(cfg.datasets.vla_data.per_device_batch_size),
                # Accelerate wraps whole batches across ranks after DataLoader
                # construction. Keep the underlying sampler global here to
                # avoid double-sharding while retaining complete triplets.
                num_replicas=1,
                rank=0,
                seed=int(cfg.get("seed", 42)),
            )
        else:
            dataloader_kwargs["batch_size"] = cfg.datasets.vla_data.per_device_batch_size

        vla_train_dataloader = DataLoader(vla_dataset, **dataloader_kwargs)
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
