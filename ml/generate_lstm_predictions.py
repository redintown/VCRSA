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

OUTPUT_PATH = (
    PROJECT_DIR
    / "sumo"
    / "datasets"
    / "highway"
    / "processed"
    / "highway_lstm_predictions.csv"
)


def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
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

    return np.column_stack(
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


def main():

    print("Loading mobility dataset...")

    df = pd.read_csv(CSV_PATH)

    model, scaler, device = load_model()

    predictions = []

    print(
        f"Vehicles: {df['vehicle_id'].nunique()}"
    )

    # ------------------------------------------------------------
    # Process each vehicle independently
    # ------------------------------------------------------------

    for vehicle_id, vehicle_df in df.groupby(
        "vehicle_id"
    ):

        vehicle_df = vehicle_df.sort_values(
            "timestamp"
        )

        # --------------------------------------------------------
        # Process each zone separately
        # --------------------------------------------------------

        for zone_id, zone_df in vehicle_df.groupby(
            "zone_id"
        ):

            zone_df = zone_df.sort_values(
                "timestamp"
            ).reset_index(drop=True)

            timestamps = zone_df[
                "timestamp"
            ].to_numpy()

            # Need exactly 10 consecutive 1-second samples.
            for i in range(
                len(zone_df) - 9
            ):

                window = timestamps[
                    i:i + 10
                ]

                if not np.allclose(
                    np.diff(window),
                    1.0
                ):
                    continue

                sequence = zone_df.iloc[
                    i:i + 10
                ]

                features = prepare_features(
                    sequence
                )

                scaled = scaler.transform(
                    features
                )

                tensor = torch.tensor(
                    scaled,
                    dtype=torch.float32,
                ).unsqueeze(0).to(device)

                with torch.no_grad():

                    prediction = model(
                        tensor
                    )

                predicted_time = max(
                    0.0,
                    float(
                        prediction.item()
                    )
                )

                predictions.append(
                    {
                        "vehicle_id":
                            vehicle_id,

                        "timestamp":
                            float(
                                sequence[
                                    "timestamp"
                                ].iloc[-1]
                            ),

                        "zone_id":
                            int(zone_id),

                        "direction":
                            sequence[
                                "direction"
                            ].iloc[-1],

                        "predicted_residency":
                            predicted_time,
                    }
                )

    result = pd.DataFrame(
        predictions
    )

    result = result.sort_values(
        [
            "timestamp",
            "vehicle_id",
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 60)
    print("LSTM PREDICTION EXPORT")
    print("=" * 60)

    print(
        f"Prediction rows : {len(result)}"
    )

    print(
        f"Vehicles        : "
        f"{result['vehicle_id'].nunique()}"
    )

    print(
        f"Output          : {OUTPUT_PATH}"
    )

    print(
        f"Device           : {device}"
    )

    print()

    print(
        result.head(10).to_string(
            index=False
        )
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
