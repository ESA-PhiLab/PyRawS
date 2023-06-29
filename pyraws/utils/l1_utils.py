from glob import glob
import numpy as np
import os
import rasterio
import rasterio.warp
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.control import GroundControlPoint as GCP
from rasterio.transform import from_gcps
import rasterio
from skimage.measure import label, regionprops
from termcolor import colored
import torch
from tqdm import tqdm
from ..utils.constants import S2_DEFAULT_QUANTIFICATION_VALUE
from ..utils.database_utils import get_event_info
from xml.dom import minidom


def get_event_bounding_box(event_hotmap, coords_dict):
    """Returns the bounding box and the coordinates of the bounding box top-left,
    bottom-right corners for each clust of 1 in the event_hotmap.

    Args:
        event_hotmap (torch.tensor): event hotmap. Pixels = 1 indicate event.
        coords_dict (dict): {"lat" : lat, lon : "lon"}, containing coordinates for each pixel in the hotmap.

    Returns:
        skimage.bb: bounding box
        list: list of coordinates [lat, lon] of top-left,
             bottom-right coordinates for each cluster of events in the hotmap.
    """
    mask = event_hotmap.numpy()
    lbl = label(mask)
    props = regionprops(lbl)
    event_bbox_coordinates_list = []
    for prop in props:
        lat = np.array(coords_dict["lat"])
        lon = np.array(coords_dict["lon"])

        bbox_coordinates = [
            [lat[prop.bbox[0], prop.bbox[1]], lon[prop.bbox[0], prop.bbox[1]]],
            [lat[prop.bbox[2], prop.bbox[3]], lon[prop.bbox[2], prop.bbox[3]]],
        ]
        event_bbox_coordinates_list.append(bbox_coordinates)

    return props, event_bbox_coordinates_list


def get_l1C_image_default_path(
    id_event,
    cfg_file_dict=None,
    id_raw_l1_dict=None,
    database="THRAWS",
    device=torch.device("cpu"),
):
    """Returns th default path to the L1C cropped image including the ""id_event_name"" tif files from database.

    Args:
        id_event (str): event ID.
        cfg_file_dict (dict, optional): dictionary containing paths to the different pyraws directories.
                                       If None, internal CSV database will be parsed.
        id_raw_l1_dict (dict, optional): id-raw-l1 dictionary.
                                       If None, internal CSV database will be parsed.
        database (string, optional): database name. Defaults to "THRAWS".

    Returns:
        string: default path to the L1C cropped image tif files.
    """
    _, _, l1c_post_processed_path, _, _, _, _, _ = get_event_info(
        id_event, cfg_file_dict, id_raw_l1_dict, database=database
    )
    return l1c_post_processed_path


def get_reprojected_bounds(rasterio_file):
    """Returns the 4 corners coordinates by reprojecting files.

    Args:
        rasterio_file (rasterio): rasterio file container.

    Returns:
        list: list of coordinates.
        dict: key params

    """
    dstCrs = {"init": "EPSG:4326"}
    srcRst = rasterio_file
    # calculate transform array and shape of reprojected raster
    transform, width, height = calculate_default_transform(
        srcRst.crs, dstCrs, srcRst.width, srcRst.height, *srcRst.bounds
    )

    # working of the meta for the destination raster
    kwargs = srcRst.meta.copy()
    kwargs.update(
        {"crs": dstCrs, "transform": transform, "width": width, "height": height}
    )
    # open destination raster
    with MemoryFile() as memfile:
        with memfile.open(**kwargs) as dstRst:
            # reproject and save raster band data
            reproject(
                source=rasterio.band(srcRst, 1),
                destination=rasterio.band(dstRst, 1),
                # src_transform=srcRst.transform,
                src_crs=srcRst.crs,
                # dst_transform=transform,
                dst_crs=dstCrs,
                resampling=Resampling.nearest,
            )

        with memfile.open() as reprojected_raster:
            left = reprojected_raster.bounds[0]
            bottom = reprojected_raster.bounds[1]
            right = reprojected_raster.bounds[2]
            top = reprojected_raster.bounds[3]

    return [[top, left], [bottom, left], [bottom, right], [top, right]], kwargs


def reproject_raster(rasterio_file, output_filename):
    """Returns the reprojected raster.

    Args:
        rasterio_file (rasterio): rasterio file container.
        output_filename (str): output_filename of the reprojected raster.

    Returns:
        raster: reprojected raster

    """
    dstCrs = {"init": "EPSG:4326"}
    srcRst = rasterio_file
    # calculate transform array and shape of reprojected raster
    transform, width, height = calculate_default_transform(
        srcRst.crs, dstCrs, srcRst.width, srcRst.height, *srcRst.bounds
    )

    # working of the meta for the destination raster
    kwargs = srcRst.meta.copy()
    kwargs.update(
        {"crs": dstCrs, "transform": transform, "width": width, "height": height}
    )
    # open destination raster
    with rasterio.open(output_filename, "w+", **kwargs) as dstRst:
        # reproject and save raster band data
        reproject(
            source=rasterio.band(srcRst, 1),
            destination=rasterio.band(dstRst, 1),
            # src_transform=srcRst.transform,
            src_crs=srcRst.crs,
            # dst_transform=transform,
            dst_crs=dstCrs,
            resampling=Resampling.nearest,
        )

    return rasterio.open(output_filename, "r")


def read_L1C_tile_from_path(
    tile_path,
    bands_list,
    reproject_bounds=True,
    verbose=True,
    device=torch.device("cpu"),
):
    """Read specific bands of an L1C Sentine2 tile,
    specified in "bands_list". The tile, located at "tile_path",
    is divided by a factor specified in the auxiliary file "auxiliary_file_path" to transform it from DN to TOA reflectance.

    Args:
        tile_path (str): Sentinel 2 image path.
        bands_list (list): bands list.
        reproject_bounds (bool, optional): if True, bounds are reprojected to EPGS:4326.
        verbose (bool, optional): if True, if True, verbose mode is used. Defaults to True.
        device (torch.device, optional): torch device. Defaults to torch.device("cpu").
    Raises:
        ValueError: Impossible to open auxiliary file.
        ValueError: Impossible to open the images with the requested bands.

    Returns:
        list: list of the requested Sentinel 2A image bands. Each band is a torch.tensor containing TOA values.
        list: list of tile's corners' coordinates extracted from metadata.
        list: list of tile's footprint's coordinates extracted from metadata.
        dict: bands_filename_dict
    """

    auxiliary_file_path = os.path.join(tile_path, "MTD_MSIL1C.xml")
    try:
        xml_content = minidom.parse(auxiliary_file_path)
        quantification_value = float(
            xml_content.getElementsByTagName("QUANTIFICATION_VALUE")[0].firstChild.data
        )
        tile_footprint_coordinates = (
            xml_content.getElementsByTagName("n1:Geometric_Info")[0]
            .getElementsByTagName("Product_Footprint")[0]
            .getElementsByTagName("Global_Footprint")[0]
            .getElementsByTagName("EXT_POS_LIST")[0]
            .firstChild.data
        )
        tile_footprint_coordinates = tile_footprint_coordinates.split(" ")
        tile_footprint_coordinates = [
            [x, y]
            for (x, y) in zip(
                tile_footprint_coordinates[::2], tile_footprint_coordinates[1::2]
            )
        ]
        tile_footprint_coordinates = tile_footprint_coordinates[
            :-1
        ]  # Excluding the last couple since it is the repetition of the first one.
    except:  # noqa: E722
        print(
            colored("Warning: ", "red")
            + "Metadata missing. "
            + "It is impossible to estimate the footprint coordinates. "
            + "However, georeferencing of the tile is still possible."
        )
        tile_footprint_coordinates = [["NA", "NA"] for i in range(4)]
        quantification_value = S2_DEFAULT_QUANTIFICATION_VALUE

    try:
        granule_name_path = sorted(glob(os.path.join(tile_path, "GRANULE", "*")))[0]

        bands_img_paths = sorted(glob(os.path.join(granule_name_path, "IMG_DATA", "*")))

        band_name_file_dict = dict(
            zip(bands_list, bands_list)
        )  # This dictionary is to match the desired band with the file.
        # We initialized with bands_list also as value because they will be fixed in the next for loop.
        for name in bands_img_paths:
            if name[-3:] == "jp2" and name[-7:-4] in bands_list:
                band_name_file_dict[name[-7:-4]] = name
        n = 0
        sentinel_img = []

        if verbose:
            for band in tqdm(bands_list, desc="Parsing sentinel bands"):
                print("Taking band: " + colored(band, "green"))
                band_k_raster = rasterio.open(band_name_file_dict[band])

                if reproject_bounds:
                    if n == 0:
                        tile_coordinates, _ = get_reprojected_bounds(band_k_raster)
                else:
                    left = band_k_raster.bounds[0]
                    bottom = band_k_raster.bounds[1]
                    right = band_k_raster.bounds[2]
                    top = band_k_raster.bounds[3]
                    tile_coordinates = [
                        [top, left],
                        [bottom, left],
                        [bottom, right],
                        [top, right],
                    ]

                band_k = band_k_raster.read(1)
                band_k_raster.close()

                fullres = band_k[:]

                band_k_torch = torch.from_numpy(
                    fullres.astype(np.float32) / float(quantification_value)
                )
                if device == torch.device("cuda"):
                    band_k_torch.to("cuda")
                sentinel_img.append(band_k_torch)
                n += 1
        else:
            for band in bands_list:
                band_k_raster = rasterio.open(band_name_file_dict[band])
                if reproject_bounds:
                    if n == 0:
                        tile_coordinates, _ = get_reprojected_bounds(band_k_raster)
                else:
                    left = band_k_raster.bounds[0]
                    bottom = band_k_raster.bounds[1]
                    right = band_k_raster.bounds[2]
                    top = band_k_raster.bounds[3]
                    tile_coordinates = [
                        [top, left],
                        [bottom, left],
                        [bottom, right],
                        [top, right],
                    ]

                band_k = band_k_raster.read(1)
                band_k_raster.close()

                fullres = band_k[:]

                band_k_torch = torch.from_numpy(
                    fullres.astype(np.float32) / float(quantification_value)
                )
                if device == torch.device("cuda"):
                    band_k_torch.to("cuda")
                sentinel_img.append(band_k_torch)
                n += 1
    except:  # noqa: E722
        raise ValueError(
            colored("Error. ", "red")
            + " impossible to open: "
            + colored(tile_path, "blue")
            + " with the requested bands."
        )

    return (
        sentinel_img,
        tile_coordinates,
        tile_footprint_coordinates,
        band_name_file_dict,
    )


def read_L1C_event_from_path(
    l1c_image_path,
    bands_list,
    reproject_bounds=True,
    verbose=True,
    device=torch.device("cpu"),
):
    """Read specific bands of an L1C Sentinel 2 event, specified in "bands_list" from img_name.
    Every tile of the event is represented as TOA reflectance.

    Args:
        id_event (str): event ID.
        bands_list (list): bands list.
        reproject_bounds (bool, optional): if True, bounds are reprojected to EPGS:4326.
        verbose (bool, optional): if True, if True, verbose mode is used. Defaults to True.
        device (torch.device, optional): torch device. Defaults to torch.device("cpu").

    Raises:
        ValueError: impossible to find information on the database.

    Returns:
        dictionary: dictionary containing [tile_name, tile]. Each tile is list of tensors,
                    each of them is made of TOA values of the requested Sentinel 2A image bands.
        dictionary: dictionary containing [tile_name, coordinates], where coordinates are the corners
                    coordinates of each tile composing the requested image.
        dictionary: dictionary containing [tile_name, coordinates], where coordinates are the footprint's
                    coordinates of each tile composing the requested image.
        dictionary: dictionary containing [tile_name, band_filename_dict], where band_filename_dict is the
                    dictionary associating to every band the original jp2 file.
    """

    tiles_paths = sorted(glob(os.path.join(l1c_image_path, "*")))

    # tiles names
    tiles_names = [
        tile_path[-tile_path[::-1].find(os.sep) :] for tile_path in tiles_paths
    ]

    sentinel_img = dict(zip(tiles_names, [0 for n in range(len(tiles_names))]))
    tiles_coordinates_dict = dict(
        zip(tiles_names, [0 for n in range(len(tiles_names))])
    )
    tiles_footprint_coordinates_dict = dict(
        zip(tiles_names, [0 for n in range(len(tiles_names))])
    )
    bands_filename_dict_dict = dict(
        zip(tiles_names, [0 for n in range(len(tiles_names))])
    )
    if verbose:
        for tile_name, tile_path in tqdm(
            zip(tiles_names, tiles_paths), desc="Parsing tiles..."
        ):
            (
                tile_bands,
                tile_coordinates,
                tile_footprint_coordinates,
                bands_filename_dict,
            ) = read_L1C_tile_from_path(
                tile_path, bands_list, reproject_bounds, verbose, device
            )
            sentinel_img[tile_name] = tile_bands
            tiles_coordinates_dict[tile_name] = tile_coordinates
            tiles_footprint_coordinates_dict[tile_name] = tile_footprint_coordinates
            bands_filename_dict_dict[tile_name] = bands_filename_dict
    else:
        for tile_name, tile_path in zip(tiles_names, tiles_paths):
            (
                tile_bands,
                tile_coordinates,
                tile_footprint_coordinates,
                bands_filename_dict,
            ) = read_L1C_tile_from_path(
                tile_path, bands_list, reproject_bounds, verbose, device
            )
            sentinel_img[tile_name] = tile_bands
            tiles_coordinates_dict[tile_name] = tile_coordinates
            tiles_footprint_coordinates_dict[tile_name] = tile_footprint_coordinates
            bands_filename_dict_dict[tile_name] = bands_filename_dict
    return (
        sentinel_img,
        tiles_coordinates_dict,
        tiles_footprint_coordinates_dict,
        bands_filename_dict_dict,
    )


def read_L1C_event_from_database(
    id_event,
    bands_list,
    cfg_file_dict=None,
    id_raw_l1_dict=None,
    database="THRAWS",
    reproject_bounds=True,
    verbose=True,
    device=torch.device("cpu"),
):
    """Read specific bands of an L1C Sentinel 2 event, specified in "bands_list" from img_name.
    Every tile of the event is represented as TOA reflectance.

    Args:
        id_event (str): event ID.
        bands_list (list): bands list.
        cfg_file_dict (dict, optional): dictionary containing paths to the different pyraws directories.
                                        If None, internal CSV database will be parsed.
        id_raw_l1_dict (dict, optional): id-raw-l1 dictionary. If None, internal CSV database will be parsed.
        database (string, optional): database name. Defaults to "THRAWS".
        reproject_bounds (bool, optional): if True, bounds are reprojected to EPGS:4326.
        verbose (bool, optional): if True, if True, verbose mode is used. Defaults to True.
        device (torch.device, optional): torch device. Defaults to torch.device("cpu").

    Raises:
        ValueError: impossible to find information on the database.

    Returns:
        dictionary: dictionary containing [tile_name, tile].
                    Each tile is list of tensors, each of them is made of TOA values of the requested Sentinel 2A image bands.
        dictionary: dictionary containing [tile_name, coordinates], where coordinates are the corners coordinates of each tile
                    composing the requested image.
        dictionary: dictionary containing [tile_name, coordinates], where coordinates are the footprint's coordinates
                    of each tile composing the requested image.
        dictionary: dictionary containing [tile_name, band_filename_dict], where band_filename_dict is the dictionary
                    associating to every band the original jp2 file.
        string: expected class name.
    """
    try:
        _, l1c_image_path, _, expected_class, _, _, _, _ = get_event_info(
            id_event, cfg_file_dict, id_raw_l1_dict, database=database
        )
    except:  # noqa: E722
        raise ValueError(
            "Impossible to find information on image: "
            + colored(id_event, "blue")
            + ". Check it is included in the database."
        )

    (
        sentinel_img,
        tiles_coordinates_dict,
        tiles_footprint_coordinates_dict,
        bands_filename_dict_dict,
    ) = read_L1C_event_from_path(
        l1c_image_path, bands_list, reproject_bounds, verbose, device=device
    )
    return (
        sentinel_img,
        tiles_coordinates_dict,
        tiles_footprint_coordinates_dict,
        bands_filename_dict_dict,
        expected_class,
    )


def read_L1C_image_from_tif(
    id_event,
    out_name_ending=None,
    cfg_file_dict=None,
    id_raw_l1_dict=None,
    database="THRAWS",
    device=torch.device("cpu"),
):
    """Read an L1C Sentine2 image from a cropped TIF. The image is represented as TOA reflectance.
    The image is post processed.

    Args:
        id_event (str): event ID.
        out_name_ending (src, optional): optional ending for the output name. Defaults to None.
        cfg_file_dict (dict, optional): dictionary containing paths to the different pyraws directories.
                                        If None, internal CSV database will be parsed.
        id_raw_l1_dict (dict, optional): id-raw-l1 dictionary. If None, internal CSV database will be parsed.
        database (string, optional): database name. Defaults to "THRAWS".
        device (torch.device, optional): torch device. Defaults to torch.device("cpu").

    Raises:
        ValueError: impossible to find information on the database.

    Returns:
        dictionary: dictionary containing every tile composing the requested image.
                    Each tensor is made of TOA values of the requested Sentinel 2A image bands.
        dictionary: dictionary containing lat and lon for every point.
        string: expected class name.
    """
    try:
        _, _, l1c_post_processed_path, expected_class, _, _, _, _ = get_event_info(
            id_event, cfg_file_dict, id_raw_l1_dict, database=database
        )
    except:  # noqa: E722
        raise ValueError(
            "Impossible to find information on image: "
            + colored(id_event, "blue")
            + ". Check it is included in the database."
        )

    if out_name_ending is not None:
        l1c_post_processed_path = (
            l1c_post_processed_path + "_" + out_name_ending + ".tif"
        )
    else:
        l1c_post_processed_path = l1c_post_processed_path + ".tif"

    with rasterio.open(l1c_post_processed_path) as raster:
        img_np = raster.read()
        sentinel_img = torch.from_numpy(img_np.astype(np.float32))
        height = sentinel_img.shape[1]
        width = sentinel_img.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(raster.transform, rows, cols)
        lons = np.array(ys)
        lats = np.array(xs)
        coords_dict = {"lat": lats, "lon": lons}

    if device == torch.device("cuda"):
        sentinel_img = sentinel_img.cuda()

    sentinel_img = sentinel_img.permute(1, 2, 0) / S2_DEFAULT_QUANTIFICATION_VALUE
    return sentinel_img, coords_dict, expected_class


def swap_latlon(poly):
    """Function that swaps latitude and logitude values
    Args:
        poly (list): list of points coordinates
    Returns:
        poly (list): list of points coordinates swapped.
    """
    poly = [[x[1], x[0]] for x in poly]
    return poly


def export_band_to_tif(band, crs, coords, save_path):
    """Export band to TIF.

    Args:
        band (torch.tensor): band to save.
        crs (string): epcs.
        coords (list): list of bands coordinates [UL, BL, BR, UR]. Each point is (LON, LAT).
        save_path (_type_): _description_
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
        "crs": {"init": crs},
        "transform": transform,
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint16",
    }

    with rasterio.open(save_path, "w", **kwargs) as dst:
        dst.write(band.detach().cpu().numpy().astype(rasterio.uint16), 1)
