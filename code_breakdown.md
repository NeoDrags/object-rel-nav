# Object-Relative Navigation Codebase Guide

This document provides a structured guide to the key files, classes, and methods across the codebase for developers reading, debugging, or extending the navigation systems.

---

## 1. Main Entrypoint & Run Loop

| File | Key Class / Method | Description |
| :--- | :--- | :--- |
| [main.py](file:///home/neodrags/object-rel-nav/main.py) | `run(args)` | The global simulator environment loop. Spawns tasks, extracts agent camera/semantic observations, processes goal logic, updates controls, and triggers visualizations/logging at each time step. |

---

## 2. Episode & Task Setup

| File | Key Class / Method | Description |
| :--- | :--- | :--- |
| [libs/experiments/task_setup.py](file:///home/neodrags/object-rel-nav/libs/experiments/task_setup.py) | [Episode](file:///home/neodrags/object-rel-nav/libs/experiments/task_setup.py#L427) | High-level orchestrator of individual navigation episodes. |
| | `get_goal(...)` | Calls the topological `Goal_Gen` class to localize visual scene features and compute object distance path lengths. |
| | `get_control_signal(...)` | Queries the chosen controller class (`tango`, `learnt`, or `pixnav`) to obtain linear/angular velocity controls. |
| | `plot(...)` / `init_plotting(...)` | Renders a multi-subplot panel capturing RGB, Depth, Semantic Mask, Selected Goal, and Trajectory logs. |
| [libs/experiments/model_loader.py](file:///home/neodrags/object-rel-nav/libs/experiments/model_loader.py) | `preload_models(...)` | Safely retrieves and instantiates deep learning weights (such as GNM/ObjectReact models or FastSAM segmentors). |

---

## 3. Controllers

### A. ObjectReact (Learning-Based)
| File | Key Class / Method | Description |
| :--- | :--- | :--- |
| [libs/control/objectreact.py](file:///home/neodrags/object-rel-nav/libs/control/objectreact.py) | [ObjRelLearntController](file:///home/neodrags/object-rel-nav/libs/control/objectreact.py#L59) | Wrapper around the GNM learning model for object relative control. |
| | `encode_goal(goal_data)` | Prepares goal embeddings and constructs the `mask_vis` visualization image by assigning normalized path-length intensities to pixel regions. |
| | `predict(rgb, goal_data)` | Evaluates model network outputs to predict forward waypoints, mapping them to steer angles and speeds. |
| | `visualize_prediction(...)` | Generates the 3-panel visualization (Predicted Trajectory, RGB view, and Goal cost-mask overlay). |

### B. Tango (Geometric / Costmap-Based)
| File | Key Class / Method | Description |
| :--- | :--- | :--- |
| [libs/control/tango/tango.py](file:///home/neodrags/object-rel-nav/libs/control/tango/tango.py) | [TangoGoalController](file:///home/neodrags/object-rel-nav/libs/control/tango/tango.py#L18) | Controller using classical path planning over topometric maps. |
| | `control(...)` | Builds safety-eroded costmaps, filters edges via box blur, and performs graph search to get the path. |
| | `add_safety_margin(...)` | Applies morphological erosion to the traversable BEV mask to keep the agent away from obstacles. |
| [libs/control/tango/path_finding/graphs.py](file:///home/neodrags/object-rel-nav/libs/control/tango/path_finding/graphs.py) | [CostMapGrid](file:///home/neodrags/object-rel-nav/libs/control/tango/path_finding/graphs.py#L23) | Models cells/coordinates, scaling movements dynamically. |
| | [CostMapGraphNX](file:///home/neodrags/object-rel-nav/libs/control/tango/path_finding/graphs.py#L64) | Builds a NetworkX representation of the costmap for Dijkstra path searches. |
| [libs/control/tango/path_finding/path_finder.py](file:///home/neodrags/object-rel-nav/libs/control/tango/path_finding/path_finder.py) | [Dijkstra](file:///home/neodrags/object-rel-nav/libs/control/tango/path_finding/path_finder.py#L30) / [AStar](file:///home/neodrags/object-rel-nav/libs/control/tango/path_finding/path_finder.py#L49) | Classic search algorithms implemented over `CostMapGrid`. |

---

## 4. Goal Planning & Topological Tracking

| File | Key Class / Method | Description |
| :--- | :--- | :--- |
| [libs/goal_generator/goal_gen.py](file:///home/neodrags/object-rel-nav/libs/goal_generator/goal_gen.py) | [Goal_Gen](file:///home/neodrags/object-rel-nav/libs/goal_generator/goal_gen.py#L22) | High-level coordinate generator mapping image landmarks to goal directions. |
| | `get_goal_mask(...)` | Runs topological localization, tracking, global path planning, and returns the combined pixel cost mask. |
| [libs/localizer/loc_topo.py](file:///home/neodrags/object-rel-nav/libs/localizer/loc_topo.py) | [LocalizeTopological](file:///home/neodrags/object-rel-nav/libs/localizer/loc_topo.py#L22) | Matches current semantic landmarks and visual nodes with pre-mapped target node descriptors. |
| [libs/planner_global/plan_topo.py](file:///home/neodrags/object-rel-nav/libs/planner_global/plan_topo.py) | [PlanTopological](file:///home/neodrags/object-rel-nav/libs/planner_global/plan_topo.py#L12) | Performs global path queries on top of the environment's landmark/node graph. |
| [libs/tracker/track_topo.py](file:///home/neodrags/object-rel-nav/libs/tracker/track_topo.py) | [TrackTopological](file:///home/neodrags/object-rel-nav/libs/tracker/track_topo.py#L15) | Filters and tracks topological history to avoid false positives and stabilize predictions. |

---

## 5. Visualizer & Data Utilities

| File | Key Class / Method | Description |
| :--- | :--- | :--- |
| [libs/common/utils_visualize.py](file:///home/neodrags/object-rel-nav/libs/common/utils_visualize.py) | `plot_sensors(...)` | Plots RGB images, Depth maps, and Semantic instances inside matplotlib windows. |
| | `plot_path_points(...)` | Superimposes path trajectories onto BEV grid charts. |
| [libs/common/utils.py](file:///home/neodrags/object-rel-nav/libs/common/utils.py) | `normalize_pls(...)` | Normalizes raw geodesic path lengths to scale goals between 0 and 100. |
