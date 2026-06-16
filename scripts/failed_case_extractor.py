import numpy as np
import yaml
import habitat

with open("configs/object_react.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

results_dir = f"{CONFIG['path_results']}/{CONFIG['task_type']}"

def failed_case_extractor(run_results_dir):
    results = np.load(f"{run_results_dir}/results_dict.npz", allow_pickle=True)
    distances = results['distance_to_goal']
    agent_states = results['agent_states']

    diffs = np.diff(distances)
    diverge_candidates = np.where((diffs[:-1] < 0) & (diffs[1:] > 0))[0]
    min_idx = diverge_candidates[0] if len(diverge_candidates) > 0 else np.argmin(distances)

    failure_state = agent_states[min_idx]
    np.save(f"{run_results_dir}/failure_state.npy", 
            {"state": failure_state, "min_dist": distances[min_idx], "step": min_idx})