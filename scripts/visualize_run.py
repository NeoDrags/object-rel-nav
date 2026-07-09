import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description="Post-run visualization of Object-React episodes.")
    parser.add_argument("run_dir", type=str, help="Path to the individual episode run directory (containing results.csv and results_dict.npz).")
    parser.add_argument("--interval", type=int, default=15, help="Step interval at which to plot predicted trajectory rollouts.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output plots (defaults to run_dir).")
    return parser.parse_args()

def main():
    args = get_args()
    run_dir = args.run_dir
    
    if not os.path.exists(run_dir):
        print(f"[Error] Run directory '{run_dir}' does not exist.")
        return

    csv_path = os.path.join(run_dir, "results.csv")
    npz_path = os.path.join(run_dir, "results_dict.npz")
    
    if not os.path.exists(csv_path) or not os.path.exists(npz_path):
        print(f"[Error] Missing results.csv or results_dict.npz in '{run_dir}'.")
        return

    print("Loading log data...")
    # Load csv for global coordinates and controls
    df = pd.read_csv(csv_path)
    
    # Load npz for controller logs (allow_pickle=True since it contains dicts/objects)
    try:
        npz_data = np.load(npz_path, allow_pickle=True)
        controller_logs = npz_data["controller_logs"]
    except Exception as e:
        print(f"[Warning] Could not load npz directly: {e}")
        return

    # Create output dir if specified
    out_dir = args.output_dir if args.output_dir else run_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Generating metrics summary plot...")
    # 1. Plot Metrics Summary
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(df["step"], df["distance_to_goal"], color="tab:blue", linewidth=2, label="Distance to Goal")
    ax1.set_ylabel("Distance to Goal (m)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_title("Episode Run Metrics Over Steps")

    # Add collision markers on the distance plot
    collisions = df[df["collided"] > 0]
    if not collisions.empty:
        ax1.scatter(collisions["step"], collisions["distance_to_goal"], color="tab:red", marker="X", s=80, label="Collision")
    ax1.legend(loc="upper right")

    ax2.plot(df["step"], df["velocity_control"], color="tab:green", linewidth=2, label="Linear Velocity (v)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Linear Velocity Control", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df["step"], df["theta_control"], color="tab:orange", linewidth=1.5, linestyle=":", label="Angular Velocity (w)")
    ax2_twin.set_ylabel("Angular Velocity Control", color="tab:orange")
    ax2_twin.tick_params(axis="y", labelcolor="tab:orange")
    
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    # Combine legends
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    metrics_img_path = os.path.join(out_dir, "metrics_summary.png")
    plt.tight_layout()
    plt.savefig(metrics_img_path, dpi=150)
    plt.close()
    print(f"Saved metrics summary to: {metrics_img_path}")

    print("Generating trajectory and rollout overlay plot...")
    # 2. Plot Trajectory and Rollout Overlay
    plt.figure(figsize=(10, 10))
    
    # Plot agent actual path (X, Z coordinates)
    sc = plt.scatter(df["x"], df["z"], c=df["step"], cmap="viridis", s=15, alpha=0.7, label="Agent Path")
    plt.plot(df["x"], df["z"], color="gray", linestyle="-", alpha=0.3)
    cbar = plt.colorbar(sc)
    cbar.set_label("Step")

    # Mark Start and End Points
    plt.scatter(df["x"].iloc[0], df["z"].iloc[0], color="tab:green", marker="o", s=150, edgecolors="black", zorder=5, label="Start")
    plt.scatter(df["x"].iloc[-1], df["z"].iloc[-1], color="tab:red", marker="*", s=250, edgecolors="black", zorder=5, label="End")

    # Transform and Plot Rollout Waypoints at intervals
    has_rollouts = False
    for idx, row in df.iterrows():
        step = int(row["step"])
        if step % args.interval == 0 and step < len(controller_logs):
            log_entry = controller_logs[step]
            if log_entry is not None and "action_pred" in log_entry:
                action_pred = log_entry["action_pred"] # shape (N_waypoints, 4) -> [dx, dy, sin_theta, cos_theta]
                if action_pred is not None:
                    has_rollouts = True
                    yaw_rad = np.radians(row["yaw"])
                    
                    # Convert relative waypoints (dx = forward, dy = lateral) to global (X, Z) coordinates:
                    local_x = action_pred[:, 0]
                    local_y = action_pred[:, 1]
                    
                    global_x_pred = row["x"] - local_x * np.sin(yaw_rad) - local_y * np.cos(yaw_rad)
                    global_z_pred = row["z"] - local_x * np.cos(yaw_rad) + local_y * np.sin(yaw_rad)
                    
                    # Plot the rollout
                    plt.plot(global_x_pred, global_z_pred, color="tab:red", linestyle="--", alpha=0.6, zorder=3)
                    plt.scatter(global_x_pred, global_z_pred, color="tab:orange", s=15, alpha=0.8, zorder=4)

    plt.xlabel("Global X Coordinate")
    plt.ylabel("Global Z Coordinate")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.axis("equal")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    if has_rollouts:
        from matplotlib.lines import Line2D
        rollout_line = Line2D([0], [0], color="tab:red", linestyle="--", alpha=0.6, label="Predicted Rollout")
        rollout_dot = Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=6, label="Predicted Waypoint")
        handles.extend([rollout_line, rollout_dot])
        labels.extend(["Predicted Rollout", "Predicted Waypoint"])
        
    plt.legend(handles=handles, labels=labels, loc="best")
    plt.title("Agent 2D Trajectory with Model Output Rollouts")

    trajectory_img_path = os.path.join(out_dir, "trajectory_overlay.png")
    plt.savefig(trajectory_img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved trajectory overlay to: {trajectory_img_path}")
    print("Done!")

if __name__ == "__main__":
    main()
