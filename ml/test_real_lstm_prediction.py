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


def main():

    print("Loading real mobility dataset...")

    df = pd.read_csv(CSV_PATH)

    print(
        f"Mobility records: {len(df)}"
    )

    # ------------------------------------------------------------
    # Select one vehicle
    # ------------------------------------------------------------

    vehicle_id = df["vehicle_id"].iloc[0]

    vehicle_df = df[
        df["vehicle_id"] == vehicle_id
    ].copy()

    vehicle_df = vehicle_df.sort_values(
        "timestamp"
    )

    print(
        f"Selected vehicle: {vehicle_id}"
    )

    print(
        f"Vehicle records: {len(vehicle_df)}"
    )

    # ------------------------------------------------------------
    # Find a valid 10-second sequence
    # ------------------------------------------------------------

    selected_sequence = None

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
                i : i + 10
            ]

            differences = np.diff(
                window
            )

            if np.allclose(
                differences,
                1.0
            ):

                selected_sequence = (
                    zone_df.iloc[
                        i : i + 10
                    ].copy()
                )

                break

        if selected_sequence is not None:
            break

    if selected_sequence is None:

        raise RuntimeError(
            "Could not find a valid "
            "10-second sequence."
        )

    # ------------------------------------------------------------
    # Prepare features
    # ------------------------------------------------------------

    direction_numeric = (
        selected_sequence[
            "direction"
        ]
        .map(
            {
                "EAST": 1.0,
                "WEST": -1.0,
            }
        )
        .astype(float)
    )

    lane_numeric = (
        selected_sequence[
            "lane_id"
        ]
        .str.extract(
            r"_(\d+)$"
        )[0]
        .astype(float)
    )

    features = np.column_stack(
        [
            selected_sequence["x"].to_numpy(),
            selected_sequence["y"].to_numpy(),
            selected_sequence["speed"].to_numpy(),
            selected_sequence[
                "acceleration"
            ].to_numpy(),
            selected_sequence["heading"].to_numpy(),
            lane_numeric.to_numpy(),
            direction_numeric.to_numpy(),
            selected_sequence["zone_id"].to_numpy(),
        ]
    ).astype(np.float32)

    # ------------------------------------------------------------
    # Load model + scaler
    # ------------------------------------------------------------

    model, scaler, device = (
        load_model()
    )

    scaled = scaler.transform(
        features
    )

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            tensor
        )

    predicted_time = float(
        prediction.item()
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("        REAL LSTM PREDICTION")
    print("=" * 60)

    print(
        f"Vehicle       : {vehicle_id}"
    )

    print(
        f"Time window   : "
        f"{selected_sequence['timestamp'].iloc[0]:.0f}"
        f"s → "
        f"{selected_sequence['timestamp'].iloc[-1]:.0f}"
        f"s"
    )

    print(
        f"Zone          : "
        f"{selected_sequence['zone_id'].iloc[-1]}"
    )

    print(
        f"Direction     : "
        f"{selected_sequence['direction'].iloc[-1]}"
    )

    print(
        f"Current speed : "
        f"{selected_sequence['speed'].iloc[-1]:.2f} m/s"
    )

    print(
        f"Predicted remaining zone time: "
        f"{predicted_time:.3f} seconds"
    )

    print(
        f"Device        : {device}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
