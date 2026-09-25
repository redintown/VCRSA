import csv
from collections import defaultdict
from pathlib import Path


INPUT_FILE = Path("sumo/datasets/highway/raw/highway_mobility.csv")
OUTPUT_FILE = Path("sumo/datasets/highway/processed/highway_lstm_targets.csv")


def load_mobility_data():
    """
    Load mobility records grouped by vehicle.

    Each record contains:
    timestamp, vehicle_id, x, y, speed, acceleration,
    heading, lane_id, road_id, direction, zone_id
    """

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

            vehicles[row["vehicle_id"]].append(row)

    return vehicles


def create_targets(vehicles):
    """
    Target definition:

    remaining_zone_time =
        time until the vehicle leaves its current zone.

    Example:

        Zone 2: t = 71 ... 108

        At t=80 -> target = 28 sec
        At t=90 -> target = 18 sec
        At t=100 -> target = 8 sec

    Samples at the final observation of a zone are excluded
    because there is no future observation from which to determine
    remaining residence time.
    """

    output_rows = []

    for vehicle_id, records in vehicles.items():

        # SUMO records should already be chronological,
        # but sort explicitly to guarantee correct ordering.
        records.sort(key=lambda r: r["timestamp"])

        n = len(records)

        for i in range(n):

            current = records[i]
            current_zone = current["zone_id"]
            current_time = current["timestamp"]

            # Find the first later observation where the vehicle
            # enters a different zone.
            exit_time = None

            for j in range(i + 1, n):

                future = records[j]

                if future["zone_id"] != current_zone:
                    exit_time = future["timestamp"]
                    break

            # No zone transition found.
            # This usually means the vehicle has reached the end
            # of its recorded trajectory.
            if exit_time is None:
                continue

            remaining_time = exit_time - current_time

            # Ignore invalid/non-positive targets.
            if remaining_time <= 0:
                continue

            output_rows.append({
                "timestamp": current_time,
                "vehicle_id": vehicle_id,
                "x": current["x"],
                "y": current["y"],
                "speed": current["speed"],
                "acceleration": current["acceleration"],
                "heading": current["heading"],
                "lane_id": current["lane_id"],
                "direction": current["direction"],
                "zone_id": current_zone,
                "remaining_zone_time": remaining_time
            })

    return output_rows


def save_targets(rows):

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "vehicle_id",
        "x",
        "y",
        "speed",
        "acceleration",
        "heading",
        "lane_id",
        "direction",
        "zone_id",
        "remaining_zone_time"
    ]

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)


def main():

    print("Loading mobility dataset...")

    vehicles = load_mobility_data()

    print(f"Vehicles loaded: {len(vehicles)}")

    print("Creating remaining-zone-time targets...")

    rows = create_targets(vehicles)

    print(f"Generated samples: {len(rows)}")

    save_targets(rows)

    print()
    print("Target dataset created successfully.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
