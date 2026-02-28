from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    input_size: int = 2       # 2D trajectory points (x, y)
    hidden_size: int = 64
    output_size: int = 3      # left, straight, right
    num_layers: int = 1


@dataclass
class DataConfig:
    data_path: str = "data_3d_h36m.npz"
    subjects: List[str] = field(
        default_factory=lambda: ["S1", "S5", "S6", "S8", "S9", "S11"]
    )
    motion: str = "Walking"
    source_freq: int = 50     # Hz, original Human3.6M frequency
    target_freq: int = 5      # Hz, downsampled frequency
    seq_length: int = 15      # Total sequence points (3 seconds at 5Hz)
    input_length: int = 14    # Points used as model input
    tanh_power: float = 0.2   # Controls label softness
    train_ratio: float = 0.9
    seed: int = 42


@dataclass
class TrainingConfig:
    num_epochs: int = 100_000
    batch_size: int = 64
    learning_rate: float = 0.01
    scheduler_step: int = 100_000
    scheduler_gamma: float = 0.5
    log_interval: int = 100
    checkpoint_interval: int = 10_000
    output_dir: str = "outputs"
    device: str = "auto"      # "auto", "cuda", "cpu"
    seed: int = 42
