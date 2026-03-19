import numpy as np


def generate_target(
    sequence: np.ndarray, input_length: int, tanh_power: float = 0.2
) -> np.ndarray:
    """Generate soft probability labels from a trajectory sequence.

    Computes the angular difference between current heading (last two input
    points) and future heading (last input point to future point). Converts
    this angle to soft [left, straight, right] probabilities using tanh.

    Args:
        sequence: Full trajectory of shape (seq_length, 2).
        input_length: Number of points used as input (future point is last).
        tanh_power: Exponent controlling label sharpness (lower = softer).

    Returns:
        Array of shape (3,) with [p_left, p_straight, p_right].
    """
    p_current = sequence[input_length - 1]
    p_previous = sequence[input_length - 2]
    p_future = sequence[-1]

    current_theta = np.arctan2(
        p_current[1] - p_previous[1], p_current[0] - p_previous[0]
    )
    future_theta = np.arctan2(
        p_future[1] - p_current[1], p_future[0] - p_current[0]
    )

    alpha = future_theta - current_theta

    f_left = max(np.tanh(alpha), 0.0) ** tanh_power
    f_right = max(np.tanh(-alpha), 0.0) ** tanh_power
    f_straight = 1.0 - f_left - f_right

    return np.array([f_left, f_straight, f_right], dtype=np.float32)
