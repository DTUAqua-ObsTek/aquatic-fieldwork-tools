from utils import find_files
import argparse
from datetime import datetime
from enum import Enum, auto


class MeasurementType(Enum):
    spread = auto()
    height = auto()
    depth = auto()
    roll = auto()
    pitch = auto()
    battery_status = auto()
    roll_px_trawleye = auto()
    pitch_px_trawleye = auto()
    depth_px_trawleye = auto()
    temperature_px_trawleye = auto()
    battery_status_px_trawleye = auto()
    height_px_trawleye = auto()
    geometry_differential_px_trawleye = auto()
    catch_px_trawleye = auto()
    clearance_px_trawleye = auto()
    opening_px_trawleye = auto()
    geometry = auto()
    geometry_differential = auto()
    temperature = auto()
    catch = auto()
    rip = auto()
    compass = auto()
    water_flow_x = auto()
    water_flow_y = auto()
    water_flow_cell = auto()


def main(args: argparse.Namespace):
    sensors = []
    paths = find_files(args.folders, ".txt", recursive=args.recursive)
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                if line.startswith("$PSIMTV80"):
                    fields = line.split(",")
                    timestamp = datetime.strptime("-".join(fields[1:3]), "%H%M%S-%y%m%d")
                    measurement_type = MeasurementType(int(fields[15]))
                    measurement_from_location = fields[16]
                    measurement_to_location = fields[17]
                    measurement = float(fields[18])
                    if measurement_from_location not in sensors:
                        sensors.append(measurement_from_location)
                    if measurement_to_location not in sensors:
                        sensors.append(measurement_to_location)
                    # Door spread measurement
                    if measurement_from_location.endswith("1") and measurement_to_location.endswith("2"):
                        print(line)
                    # depth measurement
                    if measurement_type == MeasurementType.depth and not measurement_from_location.endswith("7"):
                        print(line)
                    # height measurement
                    if measurement_type == MeasurementType.height:
                        print(line)
                    if measurement_type == MeasurementType.geometry:
                        print(line)
                    if measurement_from_location == "b7" or measurement_to_location == "b7":
                        print(line)
    print(sensors)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+", help="Space delimited list of paths you want to search for log data.")
    parser.add_argument("recursive", action="store_true", help="Search directories recursively.")
    args = parser.parse_args()
    main(args)
