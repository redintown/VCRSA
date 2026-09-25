import csv
import os
import traci


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "sumo/scenarios/highway/highway.sumocfg"
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "sumo/datasets/highway/raw/highway_mobility.csv"
)

STEP_LENGTH = 1.0


# ============================================================
# Zone mapping
# ============================================================

ZONE_MAP = {
    "e0_east": 0,
    "e0_west": 0,

    "e1_east": 1,
    "e1_west": 1,

    "e2_east": 2,
    "e2_west": 2,

    "e3_east": 3,
    "e3_west": 3,

    "e4_east": 4,
    "e4_west": 4,
}


# ============================================================
# Direction mapping
# ============================================================

DIRECTION_MAP = {
    "e0_east": "EAST",
    "e1_east": "EAST",
    "e2_east": "EAST",
    "e3_east": "EAST",
    "e4_east": "EAST",

    "e4_west": "WEST",
    "e3_west": "WEST",
    "e2_west": "WEST",
    "e1_west": "WEST",
    "e0_west": "WEST",
}


# ============================================================
# Start SUMO
# ============================================================

print("Starting SUMO...")

traci.start([
    "sumo",
    "-c",
    SUMO_CONFIG,
    "--step-length",
    str(STEP_LENGTH),
])

print("SUMO connected through TraCI.")


# ============================================================
# Prepare output directory
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_CSV),
    exist_ok=True
)


# ============================================================
# Generate CSV
# ============================================================

row_count = 0

with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "timestamp",
        "vehicle_id",
        "x",
        "y",
        "speed",
        "acceleration",
        "heading",
        "lane_id",
        "road_id",
        "direction",
        "zone_id",
    ])

    # --------------------------------------------------------
    # Simulation loop
    # --------------------------------------------------------

    while traci.simulation.getMinExpectedNumber() > 0:

        traci.simulationStep()

        timestamp = traci.simulation.getTime()

        vehicle_ids = traci.vehicle.getIDList()

        for vehicle_id in vehicle_ids:

            x, y = traci.vehicle.getPosition(
                vehicle_id
            )

            speed = traci.vehicle.getSpeed(
                vehicle_id
            )

            acceleration = traci.vehicle.getAcceleration(
                vehicle_id
            )

            heading = traci.vehicle.getAngle(
                vehicle_id
            )

            lane_id = traci.vehicle.getLaneID(
                vehicle_id
            )

            road_id = traci.vehicle.getRoadID(
                vehicle_id
            )
            
            if road_id not in ZONE_MAP:
                continue
            direction = DIRECTION_MAP.get(
                road_id,
                "UNKNOWN"
            )

            zone_id = ZONE_MAP.get(
                road_id,
                -1
            )

            writer.writerow([
                timestamp,
                vehicle_id,
                round(x, 4),
                round(y, 4),
                round(speed, 4),
                round(acceleration, 4),
                round(heading, 4),
                lane_id,
                road_id,
                direction,
                zone_id,
            ])

            row_count += 1


# ============================================================
# Close TraCI
# ============================================================

traci.close()


# ============================================================
# Summary
# ============================================================

print()
print("==============================================")
print("Mobility dataset generation completed")
print("==============================================")
print(f"Rows generated : {row_count}")
print(f"Output file    : {OUTPUT_CSV}")
print("==============================================")
