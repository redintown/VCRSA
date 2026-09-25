import csv
import os
import sys
import traci


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SUMO_CONFIG = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../scenarios/highway/highway.sumocfg"
    )
)

OUTPUT_CSV = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../datasets/highway/raw/highway_mobility.csv"
    )
)

STEP_LENGTH = 1.0


# --------------------------------------------------
# Start SUMO
# --------------------------------------------------

sumo_binary = "sumo"

traci.start([
    sumo_binary,
    "-c",
    SUMO_CONFIG,
    "--step-length",
    str(STEP_LENGTH),
])


# --------------------------------------------------
# Prepare output directory
# --------------------------------------------------

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)


# --------------------------------------------------
# CSV writer
# --------------------------------------------------

with open(OUTPUT_CSV, "w", newline="") as csv_file:

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
    ])

    # --------------------------------------------------
    # Simulation loop
    # --------------------------------------------------

    while traci.simulation.getMinExpectedNumber() > 0:

        traci.simulationStep()

        timestamp = traci.simulation.getTime()

        vehicle_ids = traci.vehicle.getIDList()

        for vehicle_id in vehicle_ids:

            x, y = traci.vehicle.getPosition(vehicle_id)

            speed = traci.vehicle.getSpeed(vehicle_id)

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

            direction = traci.vehicle.getDrivingDistance2D(
                vehicle_id,
                x,
                y,
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
            ])


# --------------------------------------------------
# Close TraCI
# --------------------------------------------------

traci.close()

print()
print("========================================")
print("Mobility dataset generation completed")
print("========================================")
print(f"Output: {OUTPUT_CSV}")
