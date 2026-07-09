import os
import argparse
import yaml
import glob
import numpy as np
import matplotlib.pyplot as plt
from argparse import Namespace
import habitat_sim

from libs.experiments import task_setup
from libs.common import utils

def get_args():
    parser = argparse.ArgumentParser(description="Reconstruct and visualize failed Object-React steps.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the individual episode run directory.")
    parser.add_argument("--step", type=int, default=150, help="Step index to reconstruct.")
    parser.add_argument("--save-path", type=str, default=None, help="Save path for generated plot (displays if not specified).")
    return parser.parse_args()

def reconstruct_step(run_dir: str, step_index: int, save_path: str = None):
    # 1. Load run arguments and config
    with open(os.path.join(run_dir, "args.yaml"), "r") as f:
        args_dict = yaml.safe_load(f)
    args = Namespace(**args_dict)
    
    # 2. Extract path scene name and directories
    path_dataset = os.path.dirname(run_dir).split("/results/")[0] + "/data"
    path_scenes_hm3d = os.path.join(path_dataset, "hm3d_v0.2", args.split)
    
    # Resolve the scene file path
    episode_dir_name = os.path.basename(run_dir)
    episode_name = episode_dir_name.split("_")[0]
    
    scene_dir = glob.glob(os.path.join(path_scenes_hm3d, f"*{episode_name}"))[0]
    scene_name_hm3d = glob.glob(os.path.join(scene_dir, "*basis.glb"))[0]
    
    # Determine map directory matching main.py logic
    sh_map = args_dict.get("sim", {}).get("sensor_height_map", 1.31)
    if args.task_type == "via_alt_goal":
        map_dir = f"hm3d_generated/stretch_maps/hm3d_iin_{args.split}/maps_via_alt_goal"
        if sh_map != 1.31:
            map_dir += f"-sh_{sh_map}/"
    else:
        if sh_map != 1.31:
            map_dir = (
                f"hm3d_generated/stretch_maps/hm3d_iin_{args.split}/height-sh_{sh_map}"
            )
        else:
            map_dir = f"hm3d_iin_{args.split}"
            
    path_episode = os.path.join(args.path_dataset, map_dir, episode_dir_name)
    
    # 3. Load model weights (GNM, Segmentors, Matchers)
    print("Preloading models...")
    preload_data = task_setup.preload_models(args)
    
    # 4. Initialize Episode environment
    print("Initializing simulation environment...")
    episode_runner = task_setup.Episode(
        args, path_episode, scene_name_hm3d, run_dir, preload_data
    )
    
    # 5. Load logged AgentStates and teleport the robot
    npz_path = os.path.join(run_dir, "results_dict.npz")
    results = np.load(npz_path, allow_pickle=True)
    agent_states = results["agent_states"]
    
    if step_index >= len(agent_states):
        raise ValueError(f"Step {step_index} is out of bounds (Total logged steps: {len(agent_states)})")
        
    logged_state = agent_states[step_index]
    
    print(f"Teleporting agent to step {step_index} pose...")
    episode_runner.agent.set_state(logged_state)
    
    # 6. Retrieve observations and segment objects
    observations = episode_runner.sim.get_sensor_observations()
    rgb, depth, semantic_instance = utils.split_observations(observations)
    
    if args.infer_depth:
        depth = preload_data["depth_model"].infer(rgb) * 0.44

    # 7. Query Topological Planner and Controller
    print("Generating costmap...")
    episode_runner.get_goal(rgb, depth, semantic_instance)
    
    # For ObjectReact, costmap is goal_mask_vis generated inside ready_goal:
    goal_data = episode_runner.control_input_learnt
    costmap = episode_runner.goal_controller.ready_goal(goal_data)
    goal_mask_vis = episode_runner.goal_controller.goal_mask_vis
    
    # 8. Visualize
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(rgb)
    ax[0].set_title("Agent Observation (RGB)")
    ax[0].axis("off")
    
    ax[1].imshow(goal_mask_vis, cmap="inferno")
    ax[1].set_title("Reconstructed Object Costmap")
    ax[1].axis("off")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    args = get_args()
    reconstruct_step(args.run_dir, args.step, args.save_path)
