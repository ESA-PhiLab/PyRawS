import os
import torch
from glob import glob
from xml.dom import minidom
from termcolor import colored
from .database_utils import get_raw_shift_lut, get_event_info, get_event_granule_bb_dict
from .constants import BANDS_RAW_SHAPE_DICT, BAND_SPATIAL_RESOLUTION_DICT
import numpy as np
import geopy.distance
from tifffile import imread
from tqdm import tqdm
from rasterio.control import GroundControlPoint as GCP
from rasterio.transform import from_gcps
import rasterio


def get_bands_shift(
    bands_list, satellite, detector_number, downsampling=True, cfg_file_dict=None
):
    """It returns the number of backward shift pixels of the various bands with respect to the first band in the list.
    Negative shift means the bands shall translate forward.

    Args:
        bands_list (list): list of bands
        satellite (str, optional): "S2A" or "S2B" respectively for "Sentinel-2A" data and "Sentinel-2B" data.
        detector_number (int): Detectorn number.
        downsampling (boolean, optional): if True, shift values for downsampled bands of the chosen satellite are used.
                                          Otherwise, values for upsampled bands are used. Defaults to True.
        cfg_file_dict (dict, optional): dictionary containing paths to the different pyraws directories.
                                        If None, internal CSV database will be parsed. Defaults to None.

    Returns:
        list: list of relative pixel shift compared to the first band.
    """
    lut_df = get_raw_shift_lut(satellite, detector_number, downsampling, cfg_file_dict)
    lut_df_keys = list(lut_df.keys())
    b_m = bands_list[0]
    if b_m == "B02":
        b_m_index = -1
    else:
        b_m_index = lut_df_keys.index(b_m)

    bands_shift_0 = []
    BAND_SPATIAL_RESOLUTION_ACROSS_TRACK_DICT = dict(
        zip(
            list(BAND_SPATIAL_RESOLUTION_DICT.keys()),
            [
                20
                if BAND_SPATIAL_RESOLUTION_DICT[band_name] == 60
                else BAND_SPATIAL_RESOLUTION_DICT[band_name]
                for band_name in list(BAND_SPATIAL_RESOLUTION_DICT.keys())
            ],
        )
    )

    for b_s in bands_list[1:]:
        if b_s == "B02":
            b_s_index = -1
        else:
            b_s_index = lut_df_keys.index(b_s)

        if b_m_index < b_s_index:
            s_along_track_ms = np.array(
                [
                    BAND_SPATIAL_RESOLUTION_DICT[lut_df_keys[k + 1]]
                    / BAND_SPATIAL_RESOLUTION_DICT[b_s]
                    * lut_df[lut_df_keys[k + 1]][0]
                    for k in range(b_m_index, b_s_index)
                ]
            )
            s_across_track_ms = np.array(
                [
                    BAND_SPATIAL_RESOLUTION_ACROSS_TRACK_DICT[lut_df_keys[k + 1]]
                    / BAND_SPATIAL_RESOLUTION_ACROSS_TRACK_DICT[b_s]
                    * lut_df[lut_df_keys[k + 1]][1]
                    for k in range(b_m_index, b_s_index)
                ]
            )
        else:
            s_along_track_ms = np.array(
                [
                    -BAND_SPATIAL_RESOLUTION_DICT[lut_df_keys[k + 1]]
                    / BAND_SPATIAL_RESOLUTION_DICT[b_s]
                    * lut_df[lut_df_keys[k + 1]][0]
                    for k in range(b_s_index, b_m_index)
                ]
            )
            s_across_track_ms = np.array(
                [
                    -BAND_SPATIAL_RESOLUTION_ACROSS_TRACK_DICT[lut_df_keys[k + 1]]
                    / BAND_SPATIAL_RESOLUTION_ACROSS_TRACK_DICT[b_s]
                    * lut_df[lut_df_keys[k + 1]][1]
                    for k in range(b_s_index, b_m_index)
                ]
            )

        bands_shift_0.append([s_along_track_ms.sum(), s_across_track_ms.sum()])

    return bands_shift_0


def get_granule_px_length(
    n_stacked_granules, satellite, detector_number, cropped_pixels_along=None
):
    """Returns the length of a granule in px
    Args:
        n_stacked_granules (int): number of stacked granules.
        satellite (str, optional): "S2A" or "S2B" respectively for "Sentinel-2A" data and "Sentinel-2B" data.
        detector_number (int): Detectorn number.
        cropped_pixels_along (int): Number of cropped pixels along the along-track direction

    Returns:
        int: Length of a granule in px.
    """
    b09_size = BANDS_RAW_SHAPE_DICT["B09"][0]

    if cropped_pixels_along is not None:
        b09_size -= cropped_pixels_along

    b_02_b09_shift = abs(
        get_bands_shift(["B02", "B09"], satellite, detector_number, downsampling=False)[
            0
        ][0]
    )
    return int(
        np.round(
            (n_stacked_granules * b09_size + b_02_b09_shift)
            * BAND_SPATIAL_RESOLUTION_DICT["B09"]
            / BAND_SPATIAL_RESOLUTION_DICT["B02"]
        )
    )


def read_Raw_granule(
    granule_path, bands_list, verbose=True, device=torch.device("cpu")
):
    """Read specific bands of an Raw Sentine2 granule, specified in "bands_list".
    The image contains several granules  at "dir_path".

    Args:
        granule_path (str): Sentinel 2 Raw granule path.
        bands_list (list): bands list.
        verbose (bool, optional): if True, if True, verbose mode is used. Defaults to True.
        device (torch.device, optional): torch device. Defaults to torch.device("cpu").
    Raises:
        ValueError: Impossible to open the images with the requested bands.

    Returns:
        list: it includes G granules, each of them is a list including  torch.tensor for each Sentinel 2A image band.
        list: metadata including original polygon coordinates.
        list: metadata including polygon cloud cover percentage.
    """

    metadata_xml_path = os.path.join(granule_path, "Inventory_Metadata.xml")
    granule_path = os.path.join(granule_path, "TIF")
    try:
        bands_img_paths = sorted(glob(os.path.join(granule_path, "*")))
        band_name_file_dict = dict(
            zip(bands_list, bands_list)
        )  # This dictionary is to match the desired band with the file. We initialized with bands_list also as
        # value because they will be fixed in the next for loop.

        for name in bands_img_paths:
            band_number = name[name.find("_B") + 1 : name.find(".tif")]
            if name[name.find(".tif") + 1 :] == "tif" and band_number in bands_list:
                band_name_file_dict[
                    name[name.find("_B") + 1 : name.find(".tif")]
                ] = name

        sentinel_raw_granule = []
        if verbose:
            for band in tqdm(bands_list, desc="Parsing sentinel bands"):
                print("Taking band: " + colored(band, "green"))
                band_k = imread(band_name_file_dict[band])[:, :, 0]
                sentinel_raw_granule.append(
                    torch.from_numpy(band_k.astype(np.float32)).to(device)
                )
        else:
            for band in bands_list:
                band_k = imread(band_name_file_dict[band])[:, :, 0]
                sentinel_raw_granule.append(
                    torch.from_numpy(band_k.astype(np.float32)).to(device)
                )
    except:  # noqa: E722
        raise ValueError(
            colored("Error. ", "red")
            + " impossible to open: "
            + colored(granule_path, "blue")
            + " with the requested bands."
        )

    try:
        # Parsing XML metadata
        xml_content = minidom.parse(metadata_xml_path)
        polygon_content = xml_content.getElementsByTagName("Geographic_Localization")
        polygon_coords_children = polygon_content[0].getElementsByTagName("Geo_Pnt")

        polygon_coordinates_list = []

        for point in polygon_coords_children[
            :-1
        ]:  # Last one is excluded to avoid repetition of the first point.
            latitude = float(point.getElementsByTagName("LATITUDE")[0].firstChild.data)
            longitude = float(
                point.getElementsByTagName("LONGITUDE")[0].firstChild.data
            )
            polygon_coordinates_list.append([latitude, longitude])

        cloud_percentage = float(
            xml_content.getElementsByTagName("CloudPercentage")[0].firstChild.data
        )
    except:  # noqa: E722
        raise ValueError(
            colored("Error. ", "red")
            + " impossible to read: "
            + colored(xml_content, "blue")
            + " Raw granule metatada."
        )

    return sentinel_raw_granule, polygon_coordinates_list, cloud_percentage


def read_Raw_event_from_path(
    dir_path, bands_list, verbose=True, device=torch.device("cpu")
):
    """Read specific bands of an Raw Sentine2 event, specified in "bands_list".
    The image contains several granules  at "dir_path".

    Args:
        dir_path (str): Sentinel 2 Raw image path.
        bands_list (list): bands list.
        verbose (bool, optional): if True, if True, verbose mode is used. Defaults to True.
        device (torch.device, optional): torch device. Defaults to torch.device("cpu").
    Raises:
        ValueError: Impossible to open the events with the requested bands.

    Returns:
        list: list of G granules, each of them is a list of the requested Sentinel 2A bands.
        list: list of granules paths.
        list: list of granules polygon coordinates.
        list: list of cloud coverage percentages.
    """

    try:
        granules_path = sorted(glob(os.path.join(dir_path, "*")))
        granules_list = []
        granules_polygons_coordinates_list = []
        cloud_percentages_list = []
        if verbose:
            for granule_path in tqdm(granules_path, desc="Parsing granules..."):
                granule, polygon_coordinates, cloud_percentage = read_Raw_granule(
                    granule_path, bands_list, verbose, device
                )
                granules_list.append(granule)
                granules_polygons_coordinates_list.append(polygon_coordinates)
                cloud_percentages_list.append(cloud_percentage)
        else:
            for granule_path in granules_path:
                granule, polygon_coordinates, cloud_percentage = read_Raw_granule(
                    granule_path, bands_list, verbose, device
                )
                granules_list.append(granule)
                granules_polygons_coordinates_list.append(polygon_coordinates)
                cloud_percentages_list.append(cloud_percentage)
        return (
            granules_list,
            granules_path,
            granules_polygons_coordinates_list,
            cloud_percentages_list,
        )

    except:  # noqa: E722
        raise ValueError(
            colored("Error. ", "red")
            + " impossible to open: "
            + colored(dir_path, "blue")
            + " with the requested bands."
        )


def find_granules_names(granules_paths):
    """Extract name of granules from granules path.

    Args:
        granules_paths (list): list of paths to the granules.

    Returns:
        list: list of granule names.
    """
    granule_names = []
    for granule_path in granules_paths:
        first_name_char_pos = -granule_path[::-1].find(os.sep)
        granule_names.append(granule_path[first_name_char_pos:])
    return granule_names


def read_Raw_event_from_database(
    id_event,
    bands_list,
    cfg_file_dict=None,
    id_raw_l1_dict=None,
    database="THRAWS",
    verbose=True,
    device=torch.device("cpu"),
):
    """Read specific bands of the Raw Sentinel-2 event ""id_event"", specified in "bands_list".

    Args:
        id_event (str): event ID.
        bands_list (list): bands list.
        cfg_file_dict (dict, optional): dictionary containing paths to the different pyraws directories.
                                        If None, internal CSV database will be parsed.
        id_raw_l1_dict (dict, optional): id-raw-l1 dictionary. If None, internal CSV database will be parsed.
        database (string, optional): database name. Defaults to "THRAWS".
        verbose (bool, optional): if True, if True, verbose mode is used. Defaults to True.
        device (torch.device, optional): torch device. Defaults to torch.device("cpu").

    Raises:
        ValueError: impossible to find information on the database.

    Returns:
        torch.tensor: tensor containing TOA values of the requested Sentinel 2A image bands.
        string: expected class name.
        list: list of granules names.
        list: list of raw data useful granules.
        list: list of complementary raw data granules (to coregister without 0).
        list: list of polygon coordinates for each granule.
        list: list of cloud coverage percentage for each granule.
        dict: {useful granule : bbox}.
    """
    try:
        (
            raw_dir_path,
            _,
            _,
            expected_class,
            raw_useful_granules,
            raw_complementary_granules,
            _,
            _,
        ) = get_event_info(id_event, cfg_file_dict, id_raw_l1_dict, database=database)
        print(raw_dir_path)
    except:  # noqa: E722
        raise ValueError(
            "Impossible to find information on event: "
            + colored(id_event, "blue")
            + ". Check it is included in the database."
        )

    (
        sentinel_raw_img,
        granules_paths,
        granules_polygons_coordinates_list,
        cloud_percentages_list,
    ) = read_Raw_event_from_path(raw_dir_path, bands_list, verbose, device)
    granule_names = find_granules_names(granules_paths)
    useful_granules_bb_dict = get_event_granule_bb_dict(
        id_event, database=database, cfg_file_dict=cfg_file_dict
    )
    return (
        sentinel_raw_img,
        expected_class,
        granule_names,
        raw_useful_granules,
        raw_complementary_granules,
        granules_polygons_coordinates_list,
        cloud_percentages_list,
        useful_granules_bb_dict,
    )


def shift_point_coordinates(point_coordinates, point_distance_y):
    """Vertical shift of point coordinates.

    Args:
        point_coordinates (list): [lan, lon] - point coordinates.
        point_distance_y (int): vertical shift. Use negative for shift towards north.

    Returns:
        list: coordinates of the shifted point.
    """
    if point_distance_y > 0:
        polygon_vertex = geopy.distance.distance(meters=point_distance_y).destination(
            (point_coordinates[0], point_coordinates[1]), bearing=180
        )
    else:
        polygon_vertex = geopy.distance.distance(meters=-point_distance_y).destination(
            (point_coordinates[0], point_coordinates[1]), bearing=0
        )
    return [polygon_vertex[0], polygon_vertex[1]]


def swap_latlon(poly):
    """Function that swaps latitude and logitude values
    Args:
        poly (list): list of points coordinates
    Returns:
        poly (list): list of points coordinates swapped.
    """
    poly = [[x[1], x[0]] for x in poly]
    return poly


def export_band_to_tif(band, coords, save_path):
    """Export band to TIF.

    Args:
        band (torch.tensor): band to save.
        coords (list): list of bands coordinates [UL, BL, BR, UR]. Each point is (LON, LAT).
        save_path (str): save path.
    """
    height, width = band[:, :].shape

    # UL                       #BL                         #BR                            #UR
    gcps = [
        GCP(0, 0, *coords[0]),
        GCP(height, 0, *coords[1]),
        GCP(height, width, *coords[2]),
        GCP(0, width, *coords[3]),
    ]
    transform = from_gcps(gcps)

    kwargs = {
        "crs": {"init": "EPSG:4326"},
        "transform": transform,
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint16",
    }

    with rasterio.open(save_path, "w", **kwargs) as dst:
        dst.write(band.detach().cpu().numpy().astype(rasterio.uint16), 1)
