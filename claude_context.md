# Claude Context — 2026-04-27

## Session Summary

Pre-experiment session to audit, clean up, and push all code before the QBot lab
run. Three repos/locations were in play; all are now in sync and pushed.

---

## Repo Map

| Repo | URL | Purpose |
|---|---|---|
| `jonathanung/follow-ahead-project` | github.com/jonathanung/follow-ahead-project | ROS2 package — the thing that actually runs |
| `Vermillion-1/qbot_ws` | github.com/Vermillion-1/qbot_ws | Outer workspace — setup scripts, install tooling, host machine config |

### Branches

**`jonathanung/follow-ahead-project`**
- `main` — stable, has everything up to PR #6 merged
- `sim-fixes` — active development branch; currently ahead of main with:
  - Simulation infrastructure (fake_odom, fake_vicon, sim.launch.py, robot.launch.py)
  - Updated main.py, planner.py, params
  - experiment_data/ (CSVs + plots from dev machine runs)
  - scripts/ (plot_results.py, run_experiment.py)
  - .devcontainer/ (Docker setup for Ubuntu 22.04)
  - Makefile, setup/cyclonedds.xml, setup/container_setup.sh

**`Vermillion-1/qbot_ws`**
- `main` — original workspace state
- `qbot_ws_pre_experiment` — snapshot with: fixed cyclonedds.xml, fixed
  host_install.sh, added zed-ros2-wrapper to ros2.repos, Makefile, how_to_run.md,
  CHANGELOG.md

---

## Three Sources of Truth (resolved)

Before this session, local files in `src/follow-ahead-project/` had 7 uncommitted
changes not pushed anywhere. Now resolved:

| Location | State |
|---|---|
| `src/follow-ahead-project/` | Committed + pushed to `sim-fixes` (f129b91) |
| `old_qbot_ws/qbot_ws/src/follow-ahead-project/` | Archived copy, at dd6256b — behind sim-fixes, safe to ignore |
| `origin/sim-fixes` | Up to date, f129b91 |

---

## What Changed in the Package (sim-fixes vs main)

See `follow_ahead_package_changelog.md` in `qbot_ws/` root for full detail.
Short version:

- **New:** `fake_odom.py`, `sim.launch.py`, `robot.launch.py`, `test_cases.yaml`,
  `main_params.yaml`, `sim.rviz`
- **Modified:** `main.py` (workspace root auto-discovery, params from YAML,
  relative action Twist), `fake_vicon.py` (Gazebo optional, stationary + zigzag
  modes, static robot fallback), `planner.py` (use_stay flag), `setup.py`
  (registers new files with colcon)
- **Deleted:** `llm/FOLLOW_AHEAD_ROS2_PLAN.md`

---

## QBot Deployment Plan (for tomorrow)

**What's working:** teleop, qbot_driver, all dependencies installed by teammate.

**Clone on QBot:**
```bash
git clone -b sim-fixes https://github.com/jonathanung/follow-ahead-project.git
cd follow-ahead-project
pip install -r requirements.txt
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select follow
source install/setup.bash
```

**Launch:**
```bash
# With Vicon (real experiment):
ros2 launch follow robot.launch.py
# Set Vicon server IP in follow/params/vicon_params.yaml first

# Without Vicon (smoke test):
ros2 run follow fake_vicon &
ros2 run follow main --ros-args --params-file follow/params/main_params.yaml -p sim:=false
```

**Key check before running:**
```bash
python3 -c "import torch; print(torch.__version__)"
```
PyTorch on Jetson needs the Jetson-specific wheel, not the standard pip one. If
teleop setup already installed PyTorch, this is fine. If it errors, install the
NVIDIA Jetson wheel before proceeding.

**Model files needed (both tracked in git, will clone automatically):**
- `lstm-fc/outputs/hypertune_v3/best_model.pt`
- `RL_sim/models/a2c_follow_ahead.zip`

**Vicon params to update:**
- `follow/params/vicon_params.yaml` → set `hostname` to Vicon PC's actual IP

---

## DevContainer (Ubuntu 22.04 dev machines)

Clone `sim-fixes`, open in VS Code → "Reopen in Container". The container:
- Base image: `althack/ros2:humble-dev`
- Installs: nav2, rmw-cyclonedds-cpp, torch, SB3, gymnasium, scipy, matplotlib
- Auto-builds the `follow` package on container create
- Sets `CYCLONEDDS_URI` and `RMW_IMPLEMENTATION` automatically

Use `make sim TC=circle` to run simulation, `make robot` for real robot.

---

## Outstanding

- `sim-fixes` not yet merged into `main` on `jonathanung/follow-ahead-project`
  — do this after the experiment when the branch is stable
- `Vermillion-1/qbot_ws` not shared with Jonathan yet — share if he needs the
  host install scripts
- One CSV from 2026-04-27 has no plot (`circle_20260427_043059.csv`) — likely
  an interrupted run
