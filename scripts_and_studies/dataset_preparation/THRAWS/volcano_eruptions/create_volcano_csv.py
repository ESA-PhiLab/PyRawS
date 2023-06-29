import csv
from datetime import date, timedelta
from termcolor import colored
import geopy.distance
import numpy as np
import argparse


sentinel2_starting_date = date(2015, 6, 23)


def parse_csv(csv_name):
    """Parse a CSV file containing volcano eruption list.

    Returns list of eruptions.

    :csv_name: CSV name.
    :return: (list of tests. Each test is a dictionary.)
    """
    eruptions_list = []
    try:
        with open(csv_name) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                eruptions_list.append(row)
    except:  # noqa: E722
        return ValueError("Impossible to parse CSV file: ", csv_name)

    return eruptions_list


def get_s2_date(eruption, date_type="Start"):
    s2_year = eruption[date_type + " Year"]
    s2_month = eruption[date_type + " Month"]
    s2_day = eruption[date_type + " Day"]
    if eruption[date_type + " Year Uncertainty"] == "":
        s2_year_uncertainty = 0
    else:
        s2_year_uncertainty = int(eruption[date_type + " Year Uncertainty"])

    if eruption[date_type + " Day Uncertainty"] == "":
        s2_day_uncertainty = 0
    else:
        s2_day_uncertainty = int(eruption[date_type + " Day Uncertainty"])

    s2_date = date(int(s2_year), int(s2_month), int(s2_day))
    s2_date -= timedelta(days=s2_year_uncertainty * 365 + s2_day_uncertainty)

    return s2_date


def parse_eruption(eruption):
    volcano_name = eruption["Volcano Name"]
    start_date = get_s2_date(eruption, date_type="Start")
    end_date = get_s2_date(eruption, date_type="End")
    discard = False

    if (sentinel2_starting_date >= start_date) and (
        sentinel2_starting_date >= end_date
    ):  # I check both to be sure there are not errors in the excel.
        discard = True

    coordinates = {
        "lat": float(eruption["Latitude"]),
        "lon": float(eruption["Longitude"]),
    }

    return [volcano_name, start_date, end_date, coordinates, discard]


def create_rectangular_polygon(
    center_coordinates, center_distance_x, center_distance_y
):
    center_distance = np.sqrt(
        center_distance_x * center_distance_x + center_distance_y * center_distance_y
    )
    bearing_angle_0 = 90 - np.arcsin(center_distance_y / center_distance) * 180 / np.pi
    print(center_distance)

    bearing_angles = {
        "SW": 180 + bearing_angle_0,
        "NW": 360 - bearing_angle_0,
        "NE": bearing_angle_0,
        "SE": 180 - bearing_angle_0,
    }
    print(bearing_angles)
    polygon_dict = {"SW": 0, "NW": 0, "NE": 0, "SE": 0}  # South-West, North-West, ...
    for bearing_keys in list(bearing_angles.keys()):
        polygon_vertex = geopy.distance.distance(meters=center_distance).destination(
            (center_coordinates["lat"], center_coordinates["lon"]),
            bearing=bearing_angles[bearing_keys],
        )
        polygon_dict[bearing_keys] = [polygon_vertex[1], polygon_vertex[0]]
    return polygon_dict


def create_polygon(center_coordinates, center_distance):
    bearing_angles = {"SW": 225.0, "NW": 135.0, "NE": 45.0, "SE": 315.0}

    polygon_dict = {"SW": 0, "NW": 0, "NE": 0, "SE": 0}  # South-West, North-West, ...
    for bearing_keys in list(bearing_angles.keys()):
        polygon_vertex = geopy.distance.distance(meters=center_distance).destination(
            (center_coordinates["lat"], center_coordinates["lon"]),
            bearing=bearing_angles[bearing_keys],
        )
        polygon_dict[bearing_keys] = [polygon_vertex[1], polygon_vertex[0]]
    return polygon_dict


def get_eruptions_info_list(
    eruption_list_csv_name,
    n_max_volcano,
    target_csv_filename="eruption_selected.csv",
    eruption_day_length=2,
    polygon_semidiagonal_km=7.0,
):
    eruptions_list = parse_csv(eruption_list_csv_name)

    volcanos_count_dict = {}
    eruptions_info_list = []
    skipped_eruptions_list = []
    for eruption in eruptions_list:
        volcano_name = eruption["Volcano Name"]
        if not (volcano_name in list(volcanos_count_dict.keys())):
            volcanos_count_dict[volcano_name] = 0

        if volcanos_count_dict[volcano_name] < n_max_volcano:
            eruption_info = parse_eruption(eruption)
            if eruption_info[-1]:
                print(
                    "Skipping volcano eruption: "
                    + colored(volcano_name, "red")
                    + "for unacceptable dates Start date("
                    + colored(eruption_info[-4], "blue")
                    + "), End date("
                    + colored(eruption_info[-3], "green")
                    + ")."
                )
                skipped_eruptions_list.append(eruption_info[:-1])
            else:
                eruptions_info_list.append(eruption_info[:-1])
                volcanos_count_dict[volcano_name] += 1
        else:
            eruption_info = parse_eruption(eruption)
            skipped_eruptions_list.append(eruption_info[:-1])
            print(
                "Skipping volcano eruption: "
                + colored(volcano_name, "red")
                + "for passing the number of eruptions for volcano ("
                + colored(n_max_volcano, "blue")
                + "."
            )

    fieldnames = [
        "Name",
        "Starting date",
        "End date",
        "Pol-SW",
        "Pol-NW",
        "Pol-NE",
        "Pol-SE",
    ]

    volcanos_id_cnt = dict(
        zip(
            list(volcanos_count_dict.keys()),
            [0 for n in range(len(list(volcanos_count_dict.keys())))],
        )
    )

    with open(target_csv_filename, mode="w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for eruption_info in eruptions_info_list:
            volcano_name = eruption_info[0]
            volcano_id = volcano_name + "_" + str(volcanos_id_cnt[volcano_name])
            volcanos_id_cnt[volcano_name] += 1
            start_date = eruption_info[1]
            end_date = eruption_info[2]
            center_coordinates = eruption_info[3]

            if start_date >= sentinel2_starting_date:
                end_date = start_date + timedelta(days=eruption_day_length)
            else:
                start_date = end_date - timedelta(days=eruption_day_length)

            start_date = (
                str(start_date.year)
                + "-"
                + str(start_date.month)
                + "-"
                + str(start_date.day)
                + "T00:00:00"
            )
            end_date = (
                str(end_date.year)
                + "-"
                + str(end_date.month)
                + "-"
                + str(end_date.day)
                + "T23:59:59"
            )
            polygon = create_polygon(center_coordinates, polygon_semidiagonal_km * 1000)
            eruption_record = [
                volcano_id,
                start_date,
                end_date,
                polygon["SW"],
                polygon["NW"],
                polygon["NE"],
                polygon["SE"],
            ]

            writer.writerow(dict(zip(fieldnames, eruption_record)))

    print("Number of accepted eruptions: " + colored(len(eruptions_info_list), "green"))
    print("Number of skipped eruptions: " + colored(len(skipped_eruptions_list), "red"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eruption_cvs",
        type=str,
        help="Eruption list CSV.",
        default="eruption_list.csv",
    )
    parser.add_argument(
        "--n_max_volcano",
        type=int,
        help="Maximum number of eruption per volcano.",
        default=5,
    )
    parser.add_argument(
        "--eruption_day_length",
        type=int,
        help="Number of days margin per eruption.",
        default=2,
    )
    parser.add_argument(
        "--pol_distance_km",
        type=float,
        help="Distance of a vertex from the polygon center in km (semidiagonal).",
        default=7.0,
    )
    pargs = parser.parse_args()
    eruption_list_csv_name = pargs.eruption_cvs
    n_max_volcano = pargs.n_max_volcano
    eruption_day_length = pargs.eruption_day_length
    polygon_semidiagonal_km = pargs.pol_distance_km
    get_eruptions_info_list(
        eruption_list_csv_name,
        n_max_volcano,
        target_csv_filename="eruption_selected.csv",
        eruption_day_length=eruption_day_length,
        polygon_semidiagonal_km=polygon_semidiagonal_km,
    )


if __name__ == "__main__":
    main()
