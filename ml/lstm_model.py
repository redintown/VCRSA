import torch
import torch.nn as nn


class MobilityLSTM(nn.Module):

    def __init__(
        self,
        input_size=8,
        hidden_size=64,
        num_layers=2,
        output_size=1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size,
        )

    def forward(self, x):

        # x shape:
        # [batch, sequence_length, features]

        output, (hidden, cell) = self.lstm(x)

        # Use the final timestep output
        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        # Return [batch]
        return prediction.squeeze(1)
