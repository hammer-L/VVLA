"""Aggregate multi-seed VA-prior metrics and emit the experiment's go/no-go decision."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(paths):
    reports = [json.loads(Path(x).read_text()) for x in paths]
    keys = ("ade", "fde", "best_of_k_ade", "best_of_k_fde", "recall", "diversity", "gripper_f1")
    return {key: {"mean": float(np.mean([x[key] for x in reports])),
                  "std": float(np.std([x[key] for x in reports]))} for key in keys}


def main():
    p = argparse.ArgumentParser()
    for head in ("deterministic", "gmm", "flow"):
        p.add_argument(f"--{head}", nargs="+", required=True, help="metrics.json files, one per seed")
    p.add_argument("--output", required=True); args = p.parse_args()
    result = {head: summarize(getattr(args, head)) for head in ("deterministic", "gmm", "flow")}
    enough_seeds = all(len(getattr(args, x)) >= 3 for x in result)
    flow_beats_det = (result["flow"]["best_of_k_ade"]["mean"] < result["deterministic"]["best_of_k_ade"]["mean"]
                      and result["flow"]["recall"]["mean"] > result["deterministic"]["recall"]["mean"])
    flow_beats_gmm = result["flow"]["recall"]["mean"] > result["gmm"]["recall"]["mean"]
    result["offline_decision"] = {
        "enough_seeds": enough_seeds,
        "flow_beats_deterministic_on_coverage": flow_beats_det,
        "flow_beats_gmm_on_coverage": flow_beats_gmm,
        "proceed_to_language_selector": bool(enough_seeds and flow_beats_det and flow_beats_gmm),
        "note": "Add closed-loop success comparison before the final go/no-go decision."
    }
    Path(args.output).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
