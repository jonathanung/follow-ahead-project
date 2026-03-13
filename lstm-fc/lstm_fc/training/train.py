"""Training entry point. Run: python -m lstm_fc.training.train [--options]"""

import argparse
import logging
import sys

from ..config import ModelConfig, TrainingConfig, DataConfig
from .trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM-FC human action predictor"
    )

    # Data args
    parser.add_argument(
        "--data-path",
        type=str,
        default="data_3d_h36m.npz",
        help="Path to Human3.6M npz file",
    )
    parser.add_argument("--target-freq", type=int, default=5)
    parser.add_argument("--seq-length", type=int, default=15)
    parser.add_argument("--input-length", type=int, default=14)
    parser.add_argument("--train-ratio", type=float, default=0.9)

    # Model args
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training args
    parser.add_argument("--num-epochs", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.006)
    parser.add_argument("--scheduler-type", type=str, default="onecycle",
                        choices=["step", "cosine", "onecycle"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)

    return parser.parse_args()


def main():
    args = parse_args()

    log_handlers = [logging.StreamHandler(sys.stdout)]
    try:
        from pathlib import Path

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        log_handlers.append(
            logging.FileHandler(f"{args.output_dir}/training.log", mode="w")
        )
    except OSError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=log_handlers,
    )

    data_config = DataConfig(
        data_path=args.data_path,
        target_freq=args.target_freq,
        seq_length=args.seq_length,
        input_length=args.input_length,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        scheduler_type=args.scheduler_type,
        device=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
    )

    trainer = Trainer(model_config, training_config, data_config)
    model_path = trainer.train()
    print(f"\nTraining complete. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
