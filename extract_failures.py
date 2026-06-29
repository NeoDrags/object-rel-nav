import os
import sys
import numpy as np
import yaml

sys.path.append(".")

with open("configs/object_react.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

results_dir = (
    f"{CONFIG['path_results']}{CONFIG['task_type']}/{CONFIG['exp_name']}/"
    f"{CONFIG['split']}/{CONFIG['max_start_distance']}"
)
curr_out_dir = f"{CONFIG['out_dir']}"


def is_failed_episode(run_results_dir):
    """Read metadata.txt and return True if the episode failed (i.e. success_status
    does NOT indicate success). Adjust the matching logic below to your actual
    metadata.txt format if this doesn't line up."""

    metadata_path = os.path.join(run_results_dir, "metadata.txt")
    if not os.path.exists(metadata_path):
        return False

    with open(metadata_path, "r") as f:
        content = f.read()

    for line in content.splitlines():
        if line.startswith("success_status"):
            _, _, value = line.partition("=")
            return value.strip().lower() != "success"

    return False


def failed_case_extractor(run_results_dir):
    """Find the sharpest divergence point in the distance-to-goal curve and
    return the agent state, distance, and step index at that point."""
    results = np.load(f"{run_results_dir}/results_dict.npz", allow_pickle=True)
    distances = results["distance_to_goal"]
    agent_states = results["agent_states"]

    d2 = np.diff(distances, n=2)
    min_idx = np.argmax(d2) + 1

    return {
        "episode": run_results_dir,
        "state": agent_states[min_idx],
        "min_dist": distances[min_idx],
        "step": int(min_idx),
    }


def find_failures(current_dir):
    current_dir = os.path.join(results_dir, current_dir)
    failure_list = []

    for root, _, files in os.walk(current_dir):
        if "results_dict.npz" not in files:
            continue
        if not is_failed_episode(root):
            continue
        try:
            failure_list.append(failed_case_extractor(root))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARN] Skipping {root}: {e}")

    if not failure_list:
        print("No failures found!")
        return

    out_path = f"{results_dir}/{curr_out_dir}/failure_state.npy"
    np.save(out_path, failure_list, allow_pickle=True)
    print(f"Saved {len(failure_list)} failure states to {out_path}")

if __name__ == "__main__":
    find_failures(curr_out_dir)
