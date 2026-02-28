import torch
import torch.nn as nn

from .config import ModelConfig


class LSTMModel2D(nn.Module):
    """LSTM with fully-connected output for 2D trajectory classification.

    Takes a sequence of 2D points and predicts probability distribution
    over actions: left, straight, right.

    Architecture: LSTM -> FC -> Softmax
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(config.hidden_size, config.output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, 2).

        Returns:
            Probabilities of shape (batch_size, 3) for [left, straight, right].
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out
