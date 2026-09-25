import csv
import random
from pathlib import Path


INPUT_FILE = Path(
    "sumo/datasets/highway/processed/highway_lstm_sequences.csv"
)

TRAIN_FILE = Path(
    "sumo/datasets/highway/processed/train.csv"
)

VAL_FILE = Path(
    "sumo/datasets/highway/processed/validation.csv"
)

TEST_FILE = Path(
    "sumo/datasets/highway/processed/test.csv"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def main():

    print("Loading sequence dataset...")

    with INPUT_FILE.open(newline="") as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        rows = list(reader)

    vehicles = sorted(
        set(row["vehicle_id"] for row in rows)
    )

    print(f"Total sequences: {len(rows)}")
    print(f"Total vehicles: {len(vehicles)}")

    # Reproducible shuffle
    random.seed(RANDOM_SEED)
    random.shuffle(vehicles)

    total_vehicles = len(vehicles)

    train_count = int(
        total_vehicles * TRAIN_RATIO
    )

    val_count = int(
        total_vehicles * VAL_RATIO
    )

    train_vehicles = set(
        vehicles[:train_count]
    )

    validation_vehicles = set(
        vehicles[
            train_count:
            train_count + val_count
        ]
    )

    test_vehicles = set(
        vehicles[
            train_count + val_count:
        ]
    )

    # Safety checks
    assert train_vehicles.isdisjoint(
        validation_vehicles
    )

    assert train_vehicles.isdisjoint(
        test_vehicles
    )

    assert validation_vehicles.isdisjoint(
        test_vehicles
    )

    assert (
        len(train_vehicles)
        + len(validation_vehicles)
        + len(test_vehicles)
        == total_vehicles
    )

    train_rows = []
    validation_rows = []
    test_rows = []

    for row in rows:

        vehicle = row["vehicle_id"]

        if vehicle in train_vehicles:
            train_rows.append(row)

        elif vehicle in validation_vehicles:
            validation_rows.append(row)

        elif vehicle in test_vehicles:
            test_rows.append(row)

    def save_file(path, data):

        with path.open("w", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(data)

    save_file(TRAIN_FILE, train_rows)
    save_file(VAL_FILE, validation_rows)
    save_file(TEST_FILE, test_rows)

    print()
    print("Vehicle-level split completed.")
    print()

    print("Vehicles:")
    print(f"  Train:      {len(train_vehicles)}")
    print(f"  Validation: {len(validation_vehicles)}")
    print(f"  Test:       {len(test_vehicles)}")

    print()
    print("Sequences:")
    print(f"  Train:      {len(train_rows)}")
    print(f"  Validation: {len(validation_rows)}")
    print(f"  Test:       {len(test_rows)}")

    print()
    print("Output files:")
    print(f"  {TRAIN_FILE}")
    print(f"  {VAL_FILE}")
    print(f"  {TEST_FILE}")


if __name__ == "__main__":
    main()
