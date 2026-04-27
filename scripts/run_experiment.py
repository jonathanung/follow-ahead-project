#!/usr/bin/env python3
"""
run_experiment.py — Launch a follow-ahead sim test case, wait, then plot results.

Usage:
    python3 run_experiment.py circle 60          # run 'circle' for 60 s then plot
    python3 run_experiment.py --all 60           # run all 5 cases sequentially
    python3 run_experiment.py circle 60 --no-plot  # skip auto-plotting

The script sources ROS2 + workspace setup automatically.
"""

import argparse
import glob
import os
import signal
import subprocess
import sys
import time

TEST_CASES   = ['circle', 'stationary', 'square', 'oscillate', 'zigzag',
                'gentle_arc', 'approach_and_hold', 'gentle_zigzag']
LOG_DIR      = os.path.expanduser('~/follow_data')
SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOT_SCRIPT  = os.path.join(SCRIPTS_DIR, 'plot_results.py')

# Adjust these paths if your workspace is somewhere else
WS_ROOT      = os.path.abspath(os.path.join(SCRIPTS_DIR, '..', '..', '..', '..', '..'))
ROS_SETUP    = '/opt/ros/humble/setup.bash'
WS_SETUP     = os.path.join(WS_ROOT, 'install', 'setup.bash')


def _source_cmd() -> str:
    """Shell snippet to source ROS2 + workspace before any ros2 command."""
    return f'source {ROS_SETUP} && source {WS_SETUP} 2>/dev/null'


def run_case(test_case: str, duration: int, plot: bool) -> str | None:
    """Launch the sim for `duration` seconds, kill it, return path to newest CSV."""
    print(f'\n{"="*60}')
    print(f'  Running: {test_case}  ({duration} s)')
    print(f'{"="*60}')

    cmd = (
        f'{_source_cmd()} && '
        f'ros2 launch follow sim.launch.py test_case:={test_case}'
    )

    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable='/bin/bash',
        preexec_fn=os.setsid,   # process group so we can kill children too
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print('\n[interrupted] killing sim…')

    # Kill entire process group (ros2 launch spawns child nodes)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=8)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

    # Give the node a moment to flush the CSV on shutdown
    time.sleep(2)

    # Find the newest CSV for this test case
    pattern = os.path.join(LOG_DIR, f'{test_case}_*.csv')
    csvs    = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not csvs:
        print(f'[warn] No CSV found for {test_case} in {LOG_DIR}')
        return None

    newest = csvs[-1]
    print(f'[ok] CSV → {newest}')

    if plot:
        _plot(newest)

    return newest


def _plot(csv_path: str):
    cmd = [sys.executable, PLOT_SCRIPT, csv_path, '--summary']
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'[warn] plot_results.py exited with code {result.returncode}')


def main():
    parser = argparse.ArgumentParser(
        description='Run follow-ahead sim test case(s) and plot results.')
    parser.add_argument('test_case', nargs='?',
                        choices=TEST_CASES + ['--all'],
                        help='Test case to run, or omit with --all')
    parser.add_argument('duration', type=int, default=60,
                        help='Simulation duration in seconds (default 60)')
    parser.add_argument('--all', dest='run_all', action='store_true',
                        help='Run all 5 test cases sequentially')
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                        help='Skip auto-plotting after each run')
    args = parser.parse_args()

    if not os.path.exists(WS_SETUP):
        print(f'[error] Workspace setup not found: {WS_SETUP}')
        print('        Build the workspace first:  colcon build --symlink-install')
        sys.exit(1)

    os.makedirs(LOG_DIR, exist_ok=True)

    cases = TEST_CASES if args.run_all or args.test_case == '--all' else [args.test_case]

    if not cases or cases == [None]:
        parser.print_help()
        sys.exit(1)

    collected = []
    for case in cases:
        csv = run_case(case, args.duration, plot=args.plot)
        if csv:
            collected.append(csv)

    # Print combined summary table if multiple cases ran
    if len(collected) > 1:
        print('\n[generating combined summary plot…]')
        cmd = [sys.executable, PLOT_SCRIPT, '--summary'] + collected
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
