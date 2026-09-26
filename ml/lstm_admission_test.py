import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

sys.path.insert(0, str(ML_DIR))

from lstm_model import MobilityLSTM
from admission_policy import classify_admission


CSV_PATH = (
    PROJECT_DIR
    / "sumo"
    / "datasets"
    / "highway"
    / "raw"
    / "highway_mobility.csv"
)

MODEL_PATH = (
    ML_DIR
    / "checkpoints"
    / "best_highway_lstm.pt"
)

SCALER_PATH = (
    ML_DIR
    / "highway_lstm_scaler.pkl"
)


# ------------------------------------------------------------
# Admission thresholds
#
# These are configurable experiment parameters.
# ------------------------------------------------------------

X_THRESHOLD = 25.0
TMIN_THRESHOLD = 10.0


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

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler, device


def find_sequence(vehicle_df):

    vehicle_df = vehicle_df.sort_values(
        "timestamp"
    )

    for zone_id, zone_df in vehicle_df.groupby(
        "zone_id"
    ):

        zone_df = zone_df.sort_values(
            "timestamp"
        )

        timestamps = (
            zone_df["timestamp"]
            .to_numpy()
        )

        for i in range(
            len(timestamps) - 9
        ):

            window = timestamps[
                i:i + 10
            ]

            if np.allclose(
                np.diff(window),
                1.0
            ):

                return zone_df.iloc[
                    i:i + 10
                ].copy()

    return None


def prepare_features(sequence):

    direction_numeric = (
        sequence["direction"]
        .map(
            {
                "EAST": 1.0,
                "WEST": -1.0,
            }
        )
        .astype(float)
    )

    lane_numeric = (
        sequence["lane_id"]
        .str.extract(
            r"_(\d+)$"
        )[0]
        .astype(float)
    )

    features = np.column_stack(
        [
            sequence["x"].to_numpy(),
            sequence["y"].to_numpy(),
            sequence["speed"].to_numpy(),
            sequence["acceleration"].to_numpy(),
            sequence["heading"].to_numpy(),
            lane_numeric.to_numpy(),
            direction_numeric.to_numpy(),
            sequence["zone_id"].to_numpy(),
        ]
    ).astype(np.float32)

    return features


def main():

    print("Loading real mobility dataset...")

    df = pd.read_csv(CSV_PATH)

    print(
        f"Mobility records: {len(df)}"
    )

    # --------------------------------------------------------
    # Select first vehicle with valid sequence
    # --------------------------------------------------------

    selected_vehicle = None
    selected_sequence = None

    for vehicle_id in df["vehicle_id"].unique():

        vehicle_df = df[
            df["vehicle_id"] == vehicle_id
        ]

        sequence = find_sequence(
            vehicle_df
        )

        if sequence is not None:

            selected_vehicle = vehicle_id
            selected_sequence = sequence

            break

    if selected_sequence is None:

        raise RuntimeError(
            "No valid 10-second sequence found."
        )

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    features = prepare_features(
        selected_sequence
    )

    model, scaler, device = load_model()

    scaled = scaler.transform(
        features
    )

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    # --------------------------------------------------------
    # LSTM inference
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            tensor
        )

    predicted_residency = float(
        prediction.item()
    )

    # Avoid negative predicted lifetime
    predicted_residency = max(
        predicted_residency,
        0.0
    )

    # --------------------------------------------------------
    # Admission decision
    # --------------------------------------------------------

    admission = classify_admission(
        predicted_residency,
        X_THRESHOLD,
        TMIN_THRESHOLD,
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("        LSTM + PREDICTIVE ADMISSION")
    print("=" * 65)

    print(
        f"Vehicle              : "
        f"{selected_vehicle}"
    )

    print(
        f"Time window           : "
        f"{selected_sequence['timestamp'].iloc[0]:.0f}s"
        f" → "
        f"{selected_sequence['timestamp'].iloc[-1]:.0f}s"
    )

    print(
        f"Zone                  : "
        f"{selected_sequence['zone_id'].iloc[-1]}"
    )

    print(
        f"Direction             : "
        f"{selected_sequence['direction'].iloc[-1]}"
    )

    print(
        f"Predicted residency   : "
        f"{predicted_residency:.3f}s"
    )

    print(
        f"Full threshold (X)    : "
        f"{X_THRESHOLD:.1f}s"
    )

    print(
        f"Minimum threshold     : "
        f"{TMIN_THRESHOLD:.1f}s"
    )

    print()
    print(
        f"ADMISSION DECISION    : "
        f"{admission.value}"
    )

    print(
        f"Device                : "
        f"{device}"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()
