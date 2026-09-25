import sys
from pathlib import Path

import joblib
import numpy as np
import torch

# Allow importing lstm_model.py from ml/
ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

from lstm_model import MobilityLSTM


MODEL_PATH = ML_DIR / "checkpoints" / "best_highway_lstm.pt"
SCALER_PATH = ML_DIR / "highway_lstm_scaler.pkl"


def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MobilityLSTM(
        input_size=8,
        hidden_size=64,
        num_layers=2,
        output_size=1,
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    scaler = joblib.load(SCALER_PATH)

    return model, scaler, device


def predict_residency(sequence):
    """
    sequence shape:
        (10, 8)

    Feature order:
        x
        y
        speed
        acceleration
        heading
        lane_id
        direction
        zone_id
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if sequence.shape != (10, 8):
        raise ValueError(
            f"Expected input shape (10, 8), "
            f"got {sequence.shape}"
        )

    model, scaler, device = load_model()

    # StandardScaler expects 2D input.
    scaled = scaler.transform(
        sequence
    )

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor)

    return float(
        prediction.item()
    )


if __name__ == "__main__":

    # ------------------------------------------------------------
    # Simple smoke test using synthetic but valid input shape.
    # ------------------------------------------------------------

    test_sequence = np.array(
        [
            [
                1000.0 + i * 10.0,
                0.0,
                27.78,
                0.0,
                90.0,
                0.0,
                1.0,
                1.0,
            ]
            for i in range(10)
        ],
        dtype=np.float32,
    )

    predicted_time = predict_residency(
        test_sequence
    )

    print(
        "LSTM inference successful."
    )

    print(
        f"Predicted remaining zone time: "
        f"{predicted_time:.3f} seconds"
    )
