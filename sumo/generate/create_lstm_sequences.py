import csv
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path(
    "sumo/datasets/highway/processed/highway_lstm_targets.csv"
)

OUTPUT_FILE = Path(
    "sumo/datasets/highway/processed/highway_lstm_sequences.csv"
)

SEQUENCE_LENGTH = 10


FEATURES = [
    "x",
    "y",
    "speed",
    "acceleration",
    "heading",
    "lane_id",
    "direction",
    "zone_id",
]


def load_data():
    vehicles = defaultdict(list)

    with INPUT_FILE.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row["timestamp"] = float(row["timestamp"])
            row["x"] = float(row["x"])
            row["y"] = float(row["y"])
            row["speed"] = float(row["speed"])
            row["acceleration"] = float(row["acceleration"])
            row["heading"] = float(row["heading"])
            row["zone_id"] = int(row["zone_id"])
            row["remaining_zone_time"] = float(
                row["remaining_zone_time"]
            )

            vehicles[row["vehicle_id"]].append(row)

    return vehicles


def create_sequences(vehicles):
    sequences = []

    for vehicle_id, records in vehicles.items():

        records.sort(key=lambda r: r["timestamp"])

        for i in range(SEQUENCE_LENGTH - 1, len(records)):

            window = records[
                i - SEQUENCE_LENGTH + 1 : i + 1
            ]

            # Make sure all observations are consecutive 1-second samples.
            timestamps = [r["timestamp"] for r in window]

            consecutive = all(
                timestamps[j] - timestamps[j - 1] == 1.0
                for j in range(1, len(timestamps))
            )

            if not consecutive:
                continue

            # A sequence should represent the same zone.
            zones = {r["zone_id"] for r in window}

            if len(zones) != 1:
                continue

            target = window[-1]["remaining_zone_time"]

            feature_values = []

            for record in window:

                lane_id = record["lane_id"]

                # Convert SUMO lane IDs such as e2_east_1
                # into a numeric lane index.
                try:
                    lane_index = int(lane_id.split("_")[-1])
                except ValueError:
                    lane_index = -1

                direction_value = (
                    1 if record["direction"] == "EAST" else -1
                )

                feature_values.append([
                    record["x"],
                    record["y"],
                    record["speed"],
                    record["acceleration"],
                    record["heading"],
                    lane_index,
                    direction_value,
                    record["zone_id"],
                ])

            sequences.append({
                "vehicle_id": vehicle_id,
                "end_timestamp": window[-1]["timestamp"],
                "zone_id": window[-1]["zone_id"],
                "target": target,
                "features": feature_values,
            })

    return sequences


def save_sequences(sequences):

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "vehicle_id",
        "end_timestamp",
        "zone_id",
        "target",
    ]

    for timestep in range(SEQUENCE_LENGTH):
        for feature in FEATURES:
            fieldnames.append(
                f"t{timestep}_{feature}"
            )

    with OUTPUT_FILE.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for sequence in sequences:

            row = {
                "vehicle_id": sequence["vehicle_id"],
                "end_timestamp": sequence["end_timestamp"],
                "zone_id": sequence["zone_id"],
                "target": sequence["target"],
            }

            for timestep, values in enumerate(
                sequence["features"]
            ):

                for feature, value in zip(
                    FEATURES,
                    values
                ):
                    row[
                        f"t{timestep}_{feature}"
                    ] = value

            writer.writerow(row)


def main():

    print("Loading target dataset...")

    vehicles = load_data()

    print(
        f"Vehicles loaded: {len(vehicles)}"
    )

    print(
        f"Creating {SEQUENCE_LENGTH}-second sequences..."
    )

    sequences = create_sequences(vehicles)

    print(
        f"Generated sequences: {len(sequences)}"
    )

    save_sequences(sequences)

    print()
    print("Sequence dataset created successfully.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
