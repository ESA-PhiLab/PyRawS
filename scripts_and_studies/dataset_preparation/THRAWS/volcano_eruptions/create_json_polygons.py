import geopandas
from shapely.geometry import Polygon
from create_volcano_csv import parse_csv
import argparse
import os
from glob import glob


def create_output_dir(out_dir):
    print("Creating the directory: ", out_dir)
    if os.path.isdir(out_dir):
        print("Cleaning the ", out_dir, "directory.")
        for file_to_remove in sorted(glob(os.path.join(out_dir, "*"))):
            os.remove(out_dir)
    else:
        os.mkdir(out_dir)


def parse_coordinate(coordinate_str):
    lon = float(
        coordinate_str[coordinate_str.find("[") + 1 : coordinate_str.find(",") - 1]
    )
    lat = float(
        coordinate_str[coordinate_str.find(",") + 1 : coordinate_str.find("]") - 1]
    )
    return (lon, lat)


def get_polygon(eruption):
    coord_sw = parse_coordinate(eruption["Pol-SW"])
    coord_nw = parse_coordinate(eruption["Pol-NW"])
    coord_ne = parse_coordinate(eruption["Pol-NE"])
    coord_se = parse_coordinate(eruption["Pol-SE"])
    return Polygon([coord_sw, coord_nw, coord_ne, coord_se])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eruption_cvs",
        type=str,
        help="Eruption list CSV.",
        default="eruption_selected.csv",
    )
    parser.add_argument(
        "--json_dir_path",
        type=str,
        help="Path to the directory where JSON files are stored.",
        default="./json",
    )
    pargs = parser.parse_args()
    csv_name = pargs.eruption_cvs
    csv_content = parse_csv(csv_name)
    json_dir = pargs.json_dir_path

    create_output_dir(json_dir)

    for eruption in csv_content:
        polygon = get_polygon(eruption)
        json_content = geopandas.GeoSeries([polygon]).to_json()

        with open(
            os.path.join(
                json_dir, eruption["Name"].replace(" ", "_").replace("/", "_") + ".json"
            ),
            "w",
        ) as f:
            f.write(json_content)


if __name__ == "__main__":
    main()
