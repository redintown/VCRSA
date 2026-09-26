import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

# Allow importing lstm_model.py from ml/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_model import MobilityLSTM


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SEQUENCE_FILE = os.path.join(
    BASE_DIR,
    "sumo",
    "datasets",
    "highway",
    "processed",
    "highway_lstm_sequences.csv",
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "ml",
    "checkpoints",
    "best_highway_lstm.pt",
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "ml",
    "highway_lstm_scaler.pkl",
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "sumo",
    "datasets",
    "highway",
    "processed",
    "highway_lstm_predictions.csv",
)


# ============================================================
# Configuration
# ============================================================

INPUT_SIZE = 8
HIDDEN_SIZE = 64
NUM_LAYERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("LSTM PREDICTION TRACE GENERATION")
    print("=" * 70)

    print()
    print("Sequence file:")
    print(SEQUENCE_FILE)

    print()
    print("Model:")
    print(MODEL_FILE)

    print()
    print("Scaler:")
    print(SCALER_FILE)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        SEQUENCE_FILE,
        MODEL_FILE,
        SCALER_FILE,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # --------------------------------------------------------
    # Load sequence dataset
    # --------------------------------------------------------

    print()
    print("Loading LSTM sequences...")

    df = pd.read_csv(SEQUENCE_FILE)

    print("Sequences:", len(df))

    # --------------------------------------------------------
    # Identify feature columns
    # --------------------------------------------------------

    feature_names = [
        "x",
        "y",
        "speed",
        "acceleration",
        "heading",
        "lane_id",
        "direction",
        "zone_id",
    ]

    sequence_feature_columns = []

    for timestep in range(10):

        for feature in feature_names:

            column = f"t{timestep}_{feature}"

            if column not in df.columns:
                raise ValueError(
                    f"Missing sequence column: {column}"
                )

            sequence_feature_columns.append(column)

    # --------------------------------------------------------
    # Extract metadata
    # --------------------------------------------------------

    metadata_columns = [
        "vehicle_id",
        "end_timestamp",
        "zone_id",
    ]

    for column in metadata_columns:

        if column not in df.columns:
            raise ValueError(
                f"Missing metadata column: {column}"
            )

    # --------------------------------------------------------
    # Build X
    # --------------------------------------------------------

    X_flat = df[
        sequence_feature_columns
    ].values.astype(np.float32)

    X = X_flat.reshape(
        len(df),
        10,
        INPUT_SIZE
    )

    print(
        "Input shape:",
        X.shape
    )

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    print()
    print("Loading training scaler...")

    scaler = joblib.load(
        SCALER_FILE
    )

    # --------------------------------------------------------
    # Scale exactly like training
    # --------------------------------------------------------

    X_2d = X.reshape(
        -1,
        INPUT_SIZE
    )

    X_scaled_2d = scaler.transform(
        X_2d
    )

    X_scaled = X_scaled_2d.reshape(
        len(df),
        10,
        INPUT_SIZE
    ).astype(np.float32)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading trained LSTM...")

    model = MobilityLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

    # Handle either raw state_dict or checkpoint dict.
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:

            # The saved object itself may be a state_dict.
            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    model.eval()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print("Running LSTM inference...")

    X_tensor = torch.from_numpy(
        X_scaled
    ).to(DEVICE)

    predictions = []

    batch_size = 512

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(X_tensor),
            )

            batch = X_tensor[
                start:end
            ]

            output = model(
                batch
            )

            predictions.extend(
                output.detach()
                .cpu()
                .numpy()
                .tolist()
            )

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Safety clipping
    # --------------------------------------------------------

    predictions = np.maximum(
        predictions,
        0.0,
    )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    output = pd.DataFrame(
    {
        "timestamp": df["end_timestamp"].values,
        "vehicle_id": df["vehicle_id"].values,
        "zone_id": df["zone_id"].values,
        "predicted_residency": predictions,
    }
)

    # Sort chronologically
    output = output.sort_values(
        [
            "timestamp",
            "vehicle_id",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREDICTION TRACE GENERATED")
    print("=" * 70)

    print(
        "Output:",
        OUTPUT_FILE
    )

    print(
        "Prediction records:",
        len(output)
    )

    print(
        "Vehicles:",
        output["vehicle_id"].nunique()
    )

    print(
        "Prediction min:",
        f"{predictions.min():.4f}s"
    )

    print(
        "Prediction max:",
        f"{predictions.max():.4f}s"
    )

    print(
        "Prediction mean:",
        f"{predictions.mean():.4f}s"
    )

    print()
    print("First 10 predictions:")
    print()

    print(
        output.head(10).to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
