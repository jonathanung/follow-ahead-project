#!/usr/bin/env python3
"""
run_experiment.py — Launch follow-ahead sim test case(s), optionally multiple runs
                    with randomised robot start poses, then plot results.

Usage:
    # Single run
    python3 run_experiment.py circle 60

    # Multi-run (10 runs, matching paper protocol)
    python3 run_experiment.py circle 60 --runs 10

    # All test cases, 10 runs each
    python3 run_experiment.py --all 60 --runs 10

    # Skip auto-plotting
    python3 run_experiment.py circle 60 --runs 10 --no-plot

Multi-run protocol (matches Leisiazar et al. IEEE RA-L 2025, Section IV-A):
    Run 1   : nominal start pose from test_cases.yaml (no perturbation)
    Runs 2-N: nominal pose + uniform random perturbation
                 Δx, Δy ∈ [-0.4, 0.4] m
                 Δθ     ∈ [-0.26, 0.26] rad  (±15°)
    CSVs are named  <test_case>_run<N>_<timestamp>.csv
    Aggregation is handled by plot_results.py --multi-run
"""

import argparse
import glob
import os
import random
import signal
import subprocess
import sys
import time

TEST_CASES   = ['circle', 'stationary', 'square', 'oscillate', 'zigzag',
                'gentle_arc', 'approach_and_hold', 'gentle_zigzag']
LOG_DIR      = os.path.expanduser('~/follow_data')
SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOT_SCRIPT  = os.path.join(SCRIPTS_DIR, 'plot_results.py')

WS_ROOT   = os.path.abspath(os.path.join(SCRIPTS_DIR, '..', '..', '..'))
ROS_SETUP = '/opt/ros/humble/setup.bash'
WS_SETUP  = os.path.join(WS_ROOT, 'install', 'setup.bash')

# Perturbation bounds for multi-run (matching paper's "randomised initial robot pose")
PERTURB_XY_M   = 0.4    # ± metres
PERTURB_THETA_R = 0.26  # ± radians (~±15°)


def _source_cmd() -> str:
    return f'source {ROS_SETUP} && source {WS_SETUP} 2>/dev/null'


def run_single(test_case: str, duration: int, run_idx: int,
               dx: float = 0.0, dy: float = 0.0, dtheta: float = 0.0,
               map_name: str = 'cropped', log_dir: str = None) -> str | None:
    """
    Launch sim for one run, wait `duration` seconds, kill, return CSV path.

    run_idx : 1-based run number (used in CSV name and log tag)
    dx/dy/dtheta : pose perturbation applied on top of test_cases.yaml nominal pose
    """
    tag = f'run {run_idx}' + (f'  Δ=({dx:+.2f},{dy:+.2f},{dtheta:+.2f})' if run_idx > 1 else '  (nominal)')
    print(f'\n{"─"*60}')
    print(f'  {test_case}  |  {tag}  |  {duration}s')
    print(f'{"─"*60}')

    # CSV name encodes run number so plot_results can group them
    env_test_case = f'{test_case}_run{run_idx:02d}'

    extra_args = ''
    if run_idx > 1:
        extra_args = (
            f' robot_start_x:={dx}'
            f' robot_start_y:={dy}'
            f' robot_start_theta:={dtheta}'
        )

    cmd = (
        f'{_source_cmd()} && '
        f'ros2 launch follow sim.launch.py test_case:={test_case} map:={map_name}{extra_args}'
    )

    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable='/bin/bash',
        preexec_fn=os.setsid,
        env={**os.environ, 'FOLLOW_TEST_CASE': env_test_case},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        print('\n[interrupted] killing sim…')

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=8)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

    time.sleep(2)  # let node flush CSV on shutdown

    _log_dir = log_dir or LOG_DIR
    pattern = os.path.join(_log_dir, f'{env_test_case}_*.csv')
    csvs    = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not csvs:
        print(f'[warn] no CSV found for {env_test_case} in {_log_dir}')
        return None

    newest = csvs[-1]
    print(f'[ok] {newest}')
    return newest


def run_case(test_case: str, duration: int, n_runs: int, plot: bool,
             map_name: str = 'cropped', log_dir: str = None) -> list[str]:
    """Run `test_case` for `n_runs` repetitions. Returns list of CSV paths."""
    csvs = []
    for i in range(1, n_runs + 1):
        if i == 1:
            dx, dy, dtheta = 0.0, 0.0, 0.0
        else:
            dx     = random.uniform(-PERTURB_XY_M,    PERTURB_XY_M)
            dy     = random.uniform(-PERTURB_XY_M,    PERTURB_XY_M)
            dtheta = random.uniform(-PERTURB_THETA_R, PERTURB_THETA_R)

        csv = run_single(test_case, duration, run_idx=i,
                         dx=dx, dy=dy, dtheta=dtheta, map_name=map_name,
                         log_dir=log_dir)
        if csv:
            csvs.append(csv)

    if plot and csvs:
        _plot_multi(test_case, csvs, n_runs)

    return csvs


def _plot_multi(test_case: str, csvs: list[str], n_runs: int):
    """Invoke plot_results.py in multi-run mode for a single test case."""
    flag = '--multi-run' if n_runs > 1 else '--summary'
    cmd  = [sys.executable, PLOT_SCRIPT, flag, '--'] + csvs
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'[warn] plot_results.py exited {result.returncode}')


def main():
    parser = argparse.ArgumentParser(
        description='Run follow-ahead sim test case(s) and plot results.')
    parser.add_argument('test_case', nargs='?',
                        help='Test case name, or omit with --all')
    parser.add_argument('duration', type=int, nargs='?', default=60,
                        help='Simulation duration per run in seconds (default: 60)')
    parser.add_argument('--duration', dest='duration_kw', type=int, default=None,
                        help='Simulation duration per run in seconds (keyword form)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs per test case (default: 1; paper uses 10)')
    parser.add_argument('--all', dest='run_all', action='store_true',
                        help='Run all test cases sequentially')
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                        help='Skip auto-plotting after each test case')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible perturbations')
    parser.add_argument('--map', default='cropped',
                        help='Map to use: cropped | my_room | open (default: cropped)')
    parser.add_argument('--log-dir', default=None,
                        help='Directory for CSV output (default: ~/follow_data)')
    args = parser.parse_args()

    # --duration keyword takes priority over positional; allows --all --duration 75
    if args.duration_kw is not None:
        args.duration = args.duration_kw

    if args.seed is not None:
        random.seed(args.seed)

    if not os.path.exists(WS_SETUP):
        print(f'[error] workspace setup not found: {WS_SETUP}')
        sys.exit(1)

    log_dir = os.path.expanduser(args.log_dir) if args.log_dir else LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    cases = TEST_CASES if args.run_all else ([args.test_case] if args.test_case else None)
    if not cases:
        parser.print_help()
        sys.exit(1)

    print(f'\nProtocol: {len(cases)} case(s) × {args.runs} run(s) × {args.duration}s')
    print(f'Output dir: {log_dir}')
    print(f'Total wall-time estimate: ~{len(cases) * args.runs * (args.duration + 15) // 60} min\n')

    all_csvs = []
    for case in cases:
        csvs = run_case(case, args.duration, n_runs=args.runs, plot=args.plot,
                        map_name=args.map, log_dir=log_dir)
        all_csvs.extend(csvs)

    # Cross-case comparison after --all runs
    if args.run_all and all_csvs and args.plot:
        print('\n[generating cross-case comparison…]')
        flag = '--multi-run' if args.runs > 1 else '--summary'
        subprocess.run([sys.executable, PLOT_SCRIPT, flag,
                        '--paper-compare', '--'] + all_csvs)


if __name__ == '__main__':
    main()
