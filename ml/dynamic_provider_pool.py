import sys
from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

sys.path.insert(0, str(ML_DIR))

from lstm_model import MobilityLSTM


# ============================================================
# Paths
# ============================================================

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


# ============================================================
# Configuration
# ============================================================

MAX_VEHICLES = 50

RESOURCE_TYPES = [
    "CPU",
    "MEMORY",
    "STORAGE",
]


# ============================================================
# Resource profile
# ============================================================

def resource_profile(vehicle_index):

    resource_class = vehicle_index % 3

    if resource_class == 0:

        return {
            "CPU": 4.0,
            "MEMORY": 8.0,
            "STORAGE": 128.0,
        }

    elif resource_class == 1:

        return {
            "CPU": 8.0,
            "MEMORY": 16.0,
            "STORAGE": 256.0,
        }

    else:

        return {
            "CPU": 12.0,
            "MEMORY": 32.0,
            "STORAGE": 512.0,
        }


# ============================================================
# LSTM loader
# ============================================================

def load_lstm():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
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
        model.load_state_dict(
            checkpoint
        )

    model.to(device)
    model.eval()

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler, device


# ============================================================
# Sequence extraction
# ============================================================

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

        timestamps = zone_df[
            "timestamp"
        ].to_numpy()

        if len(timestamps) < 10:
            continue

        for i in range(
            len(timestamps) - 9
        ):

            window = timestamps[
                i:i + 10
            ]

            if np.allclose(
                np.diff(window),
                1.0,
            ):

                return zone_df.iloc[
                    i:i + 10
                ].copy()

    return None


# ============================================================
# Feature preparation
# ============================================================

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
            sequence[
                "acceleration"
            ].to_numpy(),
            sequence["heading"].to_numpy(),
            lane_numeric.to_numpy(),
            direction_numeric.to_numpy(),
            sequence["zone_id"].to_numpy(),
        ]
    ).astype(np.float32)


# ============================================================
# LSTM prediction
# ============================================================

def predict_residency(
    sequence,
    model,
    scaler,
    device,
):

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

    return max(
        float(prediction.item()),
        0.0,
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading SUMO mobility dataset..."
    )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Mobility records: {len(df)}"
    )

    print(
        "Loading LSTM model..."
    )

    model, scaler, device = (
        load_lstm()
    )

    vehicle_ids = (
        df["vehicle_id"]
        .drop_duplicates()
        .tolist()
    )

    vehicle_ids = vehicle_ids[
        :MAX_VEHICLES
    ]

    print(
        f"Vehicles selected: "
        f"{len(vehicle_ids)}"
    )

    providers = []

    print()
    print("=" * 90)
    print(
        "                 DYNAMIC PROVIDER POOL"
    )
    print("=" * 90)

    print(
        f"{'Vehicle':<24}"
        f"{'Zone':<7}"
        f"{'Direction':<10}"
        f"{'CPU':<8}"
        f"{'Memory':<10}"
        f"{'Storage':<10}"
        f"{'Residency':<12}"
    )

    print("-" * 90)

    for vehicle_index, vehicle_id in enumerate(
        vehicle_ids
    ):

        vehicle_df = df[
            df["vehicle_id"]
            == vehicle_id
        ]

        sequence = find_sequence(
            vehicle_df
        )

        if sequence is None:
            continue

        predicted_residency = (
            predict_residency(
                sequence,
                model,
                scaler,
                device,
            )
        )

        latest = sequence.iloc[-1]

        zone = int(
            latest["zone_id"]
        )

        direction = (
            latest["direction"]
        )

        resources = resource_profile(
            vehicle_index
        )

        print(
            f"{vehicle_id:<24}"
            f"{zone:<7}"
            f"{direction:<10}"
            f"{resources['CPU']:<8.1f}"
            f"{resources['MEMORY']:<10.1f}"
            f"{resources['STORAGE']:<10.1f}"
            f"{predicted_residency:<12.2f}"
        )

        for resource_type in RESOURCE_TYPES:

            providers.append(
                {
                    "vehicle_id": vehicle_id,
                    "zone_id": zone,
                    "direction": direction,
                    "resource_type": resource_type,
                    "available_amount":
                        resources[
                            resource_type
                        ],
                    "predicted_residency":
                        predicted_residency,
                }
            )

    print("=" * 90)

    print()
    print(
        f"Provider records generated: "
        f"{len(providers)}"
    )

    print(
        f"Expected maximum: "
        f"{len(vehicle_ids) * 3}"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    valid = (
        len(providers)
        == len(vehicle_ids) * 3
    )

    residency_values = [
        p["predicted_residency"]
        for p in providers
    ]

    print()
    print("=" * 90)
    print(
        "                 VALIDATION"
    )
    print("=" * 90)

    print(
        "Provider generation : "
        + ("PASS" if valid else "CHECK")
    )

    print(
        "LSTM predictions    : "
        + (
            "PASS"
            if all(
                value >= 0
                for value in residency_values
            )
            else "CHECK"
        )
    )

    print(
        "Dynamic provider pool: "
        + (
            "PASS"
            if valid
            else "CHECK"
        )
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
