import geopandas as gpd
import geopy
from ..raw.raw_event import Raw_event
from shapely.geometry import Polygon
import numpy as np


def get_granules_polys(requested_bands, event_name, device):
    """Returns the polygons over bands coordinates and correspondent indices.

    Args:
        requested_bands (list): requested bands.
        event_name (st): event ID.
        device (torch.device): device.

    Returns:
        dict: {granule_name : polygon}
        dict: {granule_name (e.g., granule_0_STACKED_T_granule_1) : index of granules stacked along track}
    """
    raw_event = Raw_event(device=device)
    raw_event.from_database(event_name, requested_bands, verbose=False)
    stackable_granules, stackable_couples = raw_event.stack_granules_couples()
    names = [x.get_granule_info()[0] for x in stackable_granules]
    coords = [
        x.get_bands_coordinates(latlon_format=False)[requested_bands[0]]
        for x in stackable_granules
    ]
    polys = [Polygon(x) for x in coords]
    poly_dict = dict(zip(names, polys))
    polys_index = dict(zip(names, stackable_couples))
    return poly_dict, polys_index


def create_polygon_from_coordinates(
    lat, lon, center_distance_x=5000, center_distance_y=14000, json_name=None
):
    """Creates a rectangular polygon from coordinates (lat, lon). If json_name is not None, the polygon is saved as JSON file.

    Args:
        lat (float): center latitude.
        lon (float): center longitude.
        center_distance_x (int, optional): Horizontal distance from the center in m. Defaults to 5000.
        center_distance_y (int, optional): Vertical distance from the center in m. Defaults to 14000.
        json_name (str, optional): output JSON file name. If None, no file is created. Defaults to None.

    Returns:
        dict: polygon dict {"SW" : (lat, lon), "NW" : (lat, lon), "NE" : (lat, lon), "SE" : (lat, lon)}
    """
    center_coordinates = {"lat": lat, "lon": lon}
    polygon_dict = create_rectangular_polygon(
        center_coordinates, center_distance_x, center_distance_y
    )
    polygon = Polygon(
        [polygon_dict["SW"], polygon_dict["NW"], polygon_dict["NE"], polygon_dict["SE"]]
    )
    json_content = gpd.GeoSeries([polygon]).to_json()

    if json_name is not None:
        with open(json_name, "w") as f:
            f.write(json_content)
            f.write("\n\n")
            f.write(str(polygon_dict["SW"]))
            f.write("\n\n")
            f.write(str(polygon_dict["NW"]))
            f.write("\n\n")
            f.write(str(polygon_dict["NE"]))
            f.write("\n\n")
            f.write(str(polygon_dict["SE"]))

    return polygon


def create_rectangular_polygon(
    center_coordinates, center_distance_x, center_distance_y
):
    """Creates a rectangular polygon from coordinates (lat, lon).
    Args:
        center_coordinates (dict): center coordinates {"lat" : lat, "lon" : lon}
        center_distance_x (int, optional): Horizontal distance from the center in m. Defaults to 5000.
        center_distance_y (int, optional): Vertical distance from the center in m. Defaults to 14000.

    Returns:
        dict: polygon dict {"SW" : (lat, lon), "NW" : (lat, lon), "NE" : (lat, lon), "SE" : (lat, lon)}
    """

    center_distance = np.sqrt(
        center_distance_x * center_distance_x + center_distance_y * center_distance_y
    )
    bearing_angle_0 = 90 - np.arcsin(center_distance_y / center_distance) * 180 / np.pi
    # print(center_distance)

    bearing_angles = {
        "SW": 180 + bearing_angle_0,
        "NW": 360 - bearing_angle_0,
        "NE": bearing_angle_0,
        "SE": 180 - bearing_angle_0,
    }
    # print(bearing_angles)
    polygon_dict = {"SW": 0, "NW": 0, "NE": 0, "SE": 0}  # South-West, North-West, ...
    for bearing_keys in list(bearing_angles.keys()):
        polygon_vertex = geopy.distance.distance(meters=center_distance).destination(
            (center_coordinates["lat"], center_coordinates["lon"]),
            bearing=bearing_angles[bearing_keys],
        )
        polygon_dict[bearing_keys] = [polygon_vertex[1], polygon_vertex[0]]
    return polygon_dict
