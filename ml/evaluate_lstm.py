import csv
import pickle
from pathlib import Path

import numpy as np
import torch

from lstm_model import MobilityLSTM


BASE_DIR = Path("sumo/datasets/highway/processed")

TEST_FILE = BASE_DIR / "test.csv"
SCALER_FILE = Path("ml/highway_lstm_scaler.pkl")
MODEL_FILE = Path("ml/checkpoints/best_highway_lstm.pt")


SEQUENCE_LENGTH = 10
NUM_FEATURES = 8


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_test_data():

    with TEST_FILE.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def create_sequence(row):

    sequence = []

    for t in range(SEQUENCE_LENGTH):

        values = [
            float(row[f"t{t}_x"]),
            float(row[f"t{t}_y"]),
            float(row[f"t{t}_speed"]),
            float(row[f"t{t}_acceleration"]),
            float(row[f"t{t}_heading"]),
            float(row[f"t{t}_lane_id"]),
            float(row[f"t{t}_direction"]),
            float(row[f"t{t}_zone_id"]),
        ]

        sequence.append(values)

    return np.array(
        sequence,
        dtype=np.float32
    )


def main():

    print("=" * 60)
    print("LSTM Prediction Sanity Check")
    print("=" * 60)

    print()
    print("Device:", DEVICE)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

    model = MobilityLSTM(
        input_size=NUM_FEATURES,
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        output_size=1,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # -----------------------------------------------------
    # Load scaler
    # -----------------------------------------------------

    with SCALER_FILE.open("rb") as f:
        scaler = pickle.load(f)

    # -----------------------------------------------------
    # Load test samples
    # -----------------------------------------------------

    rows = load_test_data()

    print(
        "Test samples:",
        len(rows)
    )

    print()
    print("-" * 60)

    # -----------------------------------------------------
    # Predict first 10 samples
    # -----------------------------------------------------

    errors = []

    for index, row in enumerate(rows[:10]):

        sequence = create_sequence(row)

        original_shape = sequence.shape

        sequence_2d = sequence.reshape(
            -1,
            NUM_FEATURES
        )

        sequence_scaled = scaler.transform(
            sequence_2d
        )

        sequence_scaled = sequence_scaled.reshape(
            original_shape
        )

        X = torch.tensor(
            sequence_scaled,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            prediction = model(X)

        predicted = prediction.item()
        actual = float(row["target"])

        error = abs(
            predicted - actual
        )

        errors.append(error)

        print(
            f"Sample {index + 1:02d} | "
            f"Vehicle: {row['vehicle_id']} | "
            f"Zone: {row['zone_id']} | "
            f"Actual: {actual:.2f}s | "
            f"Predicted: {predicted:.2f}s | "
            f"Error: {error:.2f}s"
        )

    print("-" * 60)

    print()
    print(
        "Mean absolute error of displayed samples:",
        f"{np.mean(errors):.4f}s"
    )

    print()
    print("Sanity check complete.")


if __name__ == "__main__":
    main()
