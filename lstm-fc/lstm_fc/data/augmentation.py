import numpy as np
from typing import List


def augment_sequence(sequence: np.ndarray) -> List[np.ndarray]:
    """Apply all augmentations to a single trajectory sequence.

    Generates augmented copies via:
    1. Original (identity)
    2. Time-reversal
    3. Axis mirrors: [-1,1], [1,-1], [-1,-1]
    4. Rotations at 20 evenly spaced angles in [-pi, pi)
    5. Time-reversed + rotated

    Args:
        sequence: Trajectory of shape (seq_length, 2).

    Returns:
        List of augmented sequences, each with same shape as input.
    """
    augmented = []

    # Original
    augmented.append(sequence.copy())

    # Time-reversal
    augmented.append(np.flip(sequence, axis=0).copy())

    # Axis mirrors
    for mirror in [[-1, 1], [1, -1], [-1, -1]]:
        augmented.append(sequence * mirror)

    # Rotations + flipped rotations
    for theta in np.arange(-np.pi, np.pi, np.pi / 10):
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        augmented.append(sequence @ rot)
        augmented.append(np.flip(sequence, axis=0).copy() @ rot)

    return augmented
