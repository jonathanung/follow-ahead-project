"""Generate synthetic walking trajectory data in Human3.6M format.

Creates data_3d_h36m.npz with realistic 2D walking trajectories including
turns, straight walking, and curved paths. The data format matches what
TrajectoryDataset expects: positions_3d[subject][motion] -> (T, J, 3)
where we only use joint 0 (hip) and first 2 dimensions (x, y).
"""

import numpy as np
from pathlib import Path


def generate_walking_trajectory(
    n_frames: int = 3000,
    freq: int = 50,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """Generate a single realistic walking trajectory.

    Simulates a person walking with:
    - Base forward velocity with natural variation
    - Random gentle turns (curvature changes)
    - Occasional sharper turns
    - Small step-to-step noise

    Returns:
        Array of shape (n_frames, num_joints, 3) mimicking H36M format.
        Only joint 0 (hip) x,y coordinates are meaningful.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Walking parameters
    base_speed = 0.02 + rng.uniform(-0.005, 0.005)  # meters per frame at 50Hz
    dt = 1.0 / freq

    # Generate heading angle over time using a random walk with occasional turns
    heading = np.zeros(n_frames)
    heading[0] = rng.uniform(-np.pi, np.pi)

    # Create curvature profile: mostly straight with occasional turns
    curvature = np.zeros(n_frames)

    # Add gentle continuous curvature variation
    for _ in range(rng.randint(5, 15)):
        center = rng.randint(0, n_frames)
        width = rng.randint(50, 300)
        amplitude = rng.uniform(-0.003, 0.003)
        curvature += amplitude * np.exp(-0.5 * ((np.arange(n_frames) - center) / width) ** 2)

    # Add occasional sharper turns
    n_turns = rng.randint(3, 10)
    for _ in range(n_turns):
        center = rng.randint(100, n_frames - 100)
        width = rng.randint(20, 80)
        amplitude = rng.uniform(-0.02, 0.02)
        curvature += amplitude * np.exp(-0.5 * ((np.arange(n_frames) - center) / width) ** 2)

    # Integrate curvature to get heading
    for i in range(1, n_frames):
        heading[i] = heading[i - 1] + curvature[i]

    # Generate positions by integrating velocity
    positions_2d = np.zeros((n_frames, 2))
    speed = base_speed + rng.normal(0, 0.002, n_frames)
    speed = np.clip(speed, 0.005, 0.04)

    for i in range(1, n_frames):
        dx = speed[i] * np.cos(heading[i])
        dy = speed[i] * np.sin(heading[i])
        positions_2d[i] = positions_2d[i - 1] + np.array([dx, dy])

    # Add small noise to simulate natural gait variation
    noise = rng.normal(0, 0.001, positions_2d.shape)
    positions_2d += noise

    # Pack into H36M format: (n_frames, num_joints, 3)
    # Use 17 joints like H36M, but only joint 0 matters
    num_joints = 17
    positions_3d = np.zeros((n_frames, num_joints, 3))
    positions_3d[:, 0, 0] = positions_2d[:, 0]  # x
    positions_3d[:, 0, 1] = positions_2d[:, 1]  # y
    positions_3d[:, 0, 2] = 0.0  # z (not used)

    # Fill other joints with offsets from hip (not used but realistic)
    for j in range(1, num_joints):
        offset = rng.uniform(-0.5, 0.5, 3)
        positions_3d[:, j, :] = positions_3d[:, 0, :] + offset

    return positions_3d


def generate_dataset(
    output_path: str = "data_3d_h36m.npz",
    subjects: list = None,
    motions: list = None,
    n_frames_per_sequence: int = 3000,
    n_sequences_per_subject: int = 2,
    seed: int = 42,
):
    """Generate full synthetic dataset in H36M format.

    Args:
        output_path: Where to save the npz file.
        subjects: List of subject IDs.
        motions: List of motion types.
        n_frames_per_sequence: Frames per walking sequence.
        n_sequences_per_subject: Walking sequences per subject/motion.
        seed: Random seed.
    """
    if subjects is None:
        subjects = ["S1", "S5", "S6", "S8", "S9", "S11"]
    if motions is None:
        motions = ["Walking"]

    rng = np.random.RandomState(seed)

    positions_3d = {}
    for subject in subjects:
        positions_3d[subject] = {}
        for motion in motions:
            # Concatenate multiple sequences for longer data
            all_frames = []
            for seq_idx in range(n_sequences_per_subject):
                sub_seed = rng.randint(0, 100000)
                sub_rng = np.random.RandomState(sub_seed)
                traj = generate_walking_trajectory(
                    n_frames=n_frames_per_sequence,
                    freq=50,
                    rng=sub_rng,
                )
                all_frames.append(traj)

            positions_3d[subject][motion] = np.concatenate(all_frames, axis=0)
            print(
                f"  {subject}/{motion}: "
                f"{positions_3d[subject][motion].shape[0]} frames"
            )

    np.savez_compressed(output_path, positions_3d=positions_3d)
    print(f"\nSaved to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    print("Generating synthetic Human3.6M walking data...")
    generate_dataset()
