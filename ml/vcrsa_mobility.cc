#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

using namespace ns3;

struct MobilityRecord
{
    double timestamp;
    std::string vehicleId;
    double x;
    double y;
    double speed;
    double acceleration;
    double heading;
    std::string laneId;
    std::string roadId;
    std::string direction;
    int zoneId;
};


static std::vector<MobilityRecord>
LoadMobilityCsv(const std::string& filename)
{
    std::vector<MobilityRecord> records;

    std::ifstream file(filename);

    if (!file.is_open())
    {
        NS_FATAL_ERROR(
            "Cannot open mobility CSV: " << filename
        );
    }

    std::string line;

    // Skip CSV header
    std::getline(file, line);

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream ss(line);
        std::string field;

        std::vector<std::string> fields;

        while (std::getline(ss, field, ','))
        {
            fields.push_back(field);
        }

        if (fields.size() != 11)
        {
            continue;
        }

        MobilityRecord record;

        record.timestamp =
            std::stod(fields[0]);

        record.vehicleId =
            fields[1];

        record.x =
            std::stod(fields[2]);

        record.y =
            std::stod(fields[3]);

        record.speed =
            std::stod(fields[4]);

        record.acceleration =
            std::stod(fields[5]);

        record.heading =
            std::stod(fields[6]);

        record.laneId =
            fields[7];

        record.roadId =
            fields[8];

        record.direction =
            fields[9];

        record.zoneId =
            std::stoi(fields[10]);

        records.push_back(record);
    }

    return records;
}


int
main(int argc, char* argv[])
{
    CommandLine cmd(__FILE__);

    std::string mobilityFile =
        "../sumo/datasets/highway/raw/highway_mobility.csv";

    cmd.AddValue(
        "mobilityFile",
        "Path to SUMO mobility CSV",
        mobilityFile
    );

    cmd.Parse(argc, argv);

    std::cout
        << "Loading mobility dataset..."
        << std::endl;

    auto records =
        LoadMobilityCsv(mobilityFile);

    std::cout
        << "Total mobility records: "
        << records.size()
        << std::endl;

    if (records.empty())
    {
        NS_FATAL_ERROR(
            "Mobility dataset is empty."
        );
    }

    std::cout
        << std::endl
        << "First mobility record:"
        << std::endl;

    const auto& first = records.front();

    std::cout
        << "  Time:       "
        << first.timestamp
        << " s"
        << std::endl;

    std::cout
        << "  Vehicle:    "
        << first.vehicleId
        << std::endl;

    std::cout
        << "  Position:   ("
        << first.x
        << ", "
        << first.y
        << ")"
        << std::endl;

    std::cout
        << "  Speed:      "
        << first.speed
        << " m/s"
        << std::endl;

    std::cout
        << "  Acceleration: "
        << first.acceleration
        << " m/s^2"
        << std::endl;

    std::cout
        << "  Heading:    "
        << first.heading
        << " degrees"
        << std::endl;

    std::cout
        << "  Lane:       "
        << first.laneId
        << std::endl;

    std::cout
        << "  Road:       "
        << first.roadId
        << std::endl;

    std::cout
        << "  Direction:  "
        << first.direction
        << std::endl;

    std::cout
        << "  Zone:       "
        << first.zoneId
        << std::endl;

    std::cout
        << std::endl
        << "CSV mobility reader test successful."
        << std::endl;

    return 0;
}
