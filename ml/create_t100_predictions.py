import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT = (
    PROJECT_DIR
    / "sumo"
    / "datasets"
    / "highway"
    / "processed"
    / "highway_lstm_predictions.csv"
)

OUTPUT = (
    PROJECT_DIR
    / "sumo"
    / "datasets"
    / "highway"
    / "processed"
    / "highway_lstm_predictions_t100.csv"
)

TARGET_TIME = 100.0

df = pd.read_csv(INPUT)

snapshot = df[
    df["timestamp"].eq(TARGET_TIME)
].copy()

snapshot = snapshot.sort_values(
    "vehicle_id"
)

snapshot = snapshot[
    [
        "vehicle_id",
        "timestamp",
        "zone_id",
        "direction",
        "predicted_residency",
    ]
]

snapshot.to_csv(
    OUTPUT,
    index=False
)

print("=" * 60)
print("T=100 LSTM PREDICTION SNAPSHOT")
print("=" * 60)

print(f"Rows     : {len(snapshot)}")
print(
    f"Vehicles : "
    f"{snapshot['vehicle_id'].nunique()}"
)
print(f"Output   : {OUTPUT}")

print()
print(snapshot.to_string(index=False))

print()
print("Zone distribution:")
print(
    snapshot["zone_id"]
    .value_counts()
    .sort_index()
)

print("=" * 60)
