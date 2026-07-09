import os
import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt

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
    
    # Safely generate and save latent variance plot if present
    if "latent" in results.files:
        latent_history = results["latent"]
        if latent_history is not None and len(latent_history) > 0 and latent_history[0] is not None:
            # Filter out any None values and convert to a numpy array
            valid_latents = [l for l in latent_history if l is not None]
            if len(valid_latents) > 0:
                latent_arr = np.array(valid_latents) # Shape: (num_steps, 32)
                latent_var = np.var(latent_arr, axis=1) # Variance across dimensions for each step
                
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                # Plot Latent Variance (left axis)
                color_var = "tab:purple"
                ax1.set_xlabel("Step")
                ax1.set_ylabel("Latent Variance", color=color_var)
                ax1.plot(latent_var, color=color_var, linewidth=2, label="Latent Variance")
                ax1.tick_params(axis="y", labelcolor=color_var)
                ax1.grid(True, linestyle="--", alpha=0.6)
                
                # Plot Distance to Goal (right axis)
                ax2 = ax1.twinx()
                color_dist = "tab:blue"
                ax2.set_ylabel("Distance to Goal (m)", color=color_dist)
                ax2.plot(distances[:len(latent_var)], color=color_dist, linewidth=2, linestyle="--", label="Distance to Goal")
                ax2.tick_params(axis="y", labelcolor=color_dist)
                
                # Draw vertical line at failure divergence point
                ax1.axvline(x=min_idx, color="tab:red", linestyle=":", linewidth=2, label=f"Divergence Step ({min_idx})")
                
                # Combine legends from both axes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
                
                plt.title(f"Latent Variance & Distance to Goal Over Steps\n(Failure Divergence Point: Step {min_idx})")
                fig.tight_layout()
                
                out_img_path = os.path.join(run_results_dir, "latent_variance.png")
                plt.savefig(out_img_path, dpi=150, bbox_inches="tight")
                plt.close()

                # --- Generate PCA Plot (PC1 vs PC2 vs PC3) ---
                if len(latent_arr) > 2:
                    mean_val = np.mean(latent_arr, axis=0)
                    centered_data = latent_arr - mean_val
                    U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)
                    pca_proj = centered_data @ Vt[:3].T # (num_steps, 3)
                    
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111, projection="3d")
                    
                    # Plot trajectory line in 3D
                    ax.plot(pca_proj[:, 0], pca_proj[:, 1], pca_proj[:, 2], color="gray", linestyle="-", alpha=0.4, zorder=1)
                    
                    split_idx = min(min_idx, len(latent_arr) - 1)
                    
                    # Plot points before divergence (Green)
                    ax.scatter(pca_proj[:split_idx, 0], pca_proj[:split_idx, 1], pca_proj[:split_idx, 2],
                               color="tab:green", label="Before Divergence", s=40, alpha=0.8, zorder=2)
                    
                    # Plot points after divergence (Red)
                    ax.scatter(pca_proj[split_idx:, 0], pca_proj[split_idx:, 1], pca_proj[split_idx:, 2],
                               color="tab:red", label="After Divergence", s=40, alpha=0.8, zorder=2)
                    
                    # Mark milestones
                    ax.scatter(pca_proj[0, 0], pca_proj[0, 1], pca_proj[0, 2], color="cyan", marker="o", s=120, edgecolors="black", label="Start", zorder=3)
                    ax.scatter(pca_proj[-1, 0], pca_proj[-1, 1], pca_proj[-1, 2], color="orange", marker="X", s=120, edgecolors="black", label="End", zorder=3)
                    ax.scatter(pca_proj[split_idx, 0], pca_proj[split_idx, 1], pca_proj[split_idx, 2], color="yellow", marker="*", s=200, edgecolors="black", label="Divergence Point", zorder=4)
                    
                    ax.set_xlabel("PC 1")
                    ax.set_ylabel("PC 2")
                    ax.set_zlabel("PC 3")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.legend(loc="best")
                    plt.title(f"3D PCA of Latent Space Trajectory\n(Failure Divergence Point: Step {split_idx})")
                    
                    pca_img_path = os.path.join(run_results_dir, "latent_pca.png")
                    plt.savefig(pca_img_path, dpi=150, bbox_inches="tight")
                    plt.close()

    return {
        "episode": run_results_dir,
        "state": agent_states[min_idx],
        "min_dist": distances[min_idx],
        "step": int(min_idx),
    }

def find_failures(current_dir):
    # Scan the entire results directory to update all runs
    scan_dir = results_dir
    failure_list = []

    for root, _, files in os.walk(scan_dir):
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