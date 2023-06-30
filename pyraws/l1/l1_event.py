from .l1_tile import L1C_tile
from ..utils.date_utils import get_timestamp
from ..utils.constants import BAND_SPATIAL_RESOLUTION_DICT
from ..utils.l1_utils import (
    get_l1C_image_default_path,
    read_L1C_event_from_database,
    read_L1C_event_from_path,
    reproject_raster,
)
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
from shapely.geometry import Polygon
from rasterio.mask import mask
from rasterio.merge import merge
from termcolor import colored
import torch
from tqdm import tqdm


class L1C_event:
    __device = None
    __bands_names = []
    __tiles_collection = []
    __event_name = []

    def __init__(
        self,
        tiles_collection=None,
        bands_names=None,
        event_class=None,
        event_name=None,
        device=torch.device("cpu"),
    ):
        """Creates an L1C image from a tiles collection and band_names.
        It is possible to associate an image class to the image.

        Args:
            tiles_collection (list, optional): list of L1C_tile. Defaults to None.
            bands_names (list, optional): list of band names. Defaults to None.
            event_class (string, optional): class name. Defaults to None.
            event_name (str, optional): event name. Defaults to None.
            device (torch.device, optional): device for each L0 tile in the image. Defaults to torch.device("cpu").
        """
        if bands_names is None:
            self.__bands_names = []
        else:
            self.__bands_names = bands_names

        self.__event_class = event_class
        if tiles_collection is None:
            self.__tiles_collection = []
            self.n_tiles = 0
        else:
            for n in range(len(tiles_collection)):
                self.__tiles_collection.append(
                    torch.device(tiles_collection[n], device=device)
                )
            self.n_tiles = len(self.__tiles_collection)

        self.n_tiles = 0

        self.__event_name = event_name

    def get_bands_list(self):
        """Returns the list of bands of every L1C_tile object in the collection.

        Returns:
            list: band names.
        """
        return self.__bands_names

    def get_event_class(self):
        """Get event class.

        Returns:
            dict: Returns {useful granules : bounding box dictionary}
        """
        return self.__event_class

    def from_path(self, l1c_dir_path, bands_list, reproject_bounds=True, verbose=True):
        """Read specific bands of the Sentinel-2 L0 event located at ""l0_dir_path"".

        Args:
            l1c_dir_path (str): path to the l0 event dir.
            bands list. If None, all bands are used and sorted according to the datasheet order. Defaults to None.
            reproject_bounds (bool, optional): if True, bounds are reprojected to EPGS:4326.
            verbose (bool, optional): if True, if True, verbose mode is used. Defaults to True.
        """
        if reproject_bounds:
            crs = "EPSG:4326:"
        else:
            crs = "EPSG:32633:"

        if bands_list is None:
            bands_list = list(BAND_SPATIAL_RESOLUTION_DICT.keys())
        else:
            bands_list = bands_list

        try:
            (
                tiles_collection_dict,
                tiles_coordinates_dict,
                footprint_coordinates_dict,
                bands_file_name_dict_dict,
            ) = read_L1C_event_from_path(
                l1c_dir_path,
                bands_list,
                reproject_bounds,
                verbose,
                device=self.__device,
            )
        except:  # noqa: E722
            raise ValueError(
                "Impossible to open the L1C file at: "
                + colored(l1c_dir_path, "red")
                + "."
            )

        self.__event_name = l1c_dir_path.split(os.sep)[-1]
        tiles_names = list(tiles_collection_dict.keys())
        self.__bands_names = bands_list
        self.__event_class = None

        for tile_name in tiles_names:
            tile = tiles_collection_dict[tile_name]
            new_tile = L1C_tile(device=self.__device)
            new_tile.create_tile(
                bands_list,
                tile,
                tile_name,
                tiles_coordinates_dict[tile_name],
                footprint_coordinates_dict[tile_name],
                bands_file_name_dict_dict[tile_name],
                crs,
            )
            self.__tiles_collection.append(new_tile)

        self.n_tiles = len(self.__tiles_collection)

    def from_database(
        self,
        id_event,
        bands_list=None,
        cfg_file_dict=None,
        id_l0_l1_dict=None,
        reproject_bounds=True,
        verbose=True,
        database="THRAWS",
    ):
        """Read specific bands of the L1 Sentine2 image ""id_event"", specified in "bands_list", from database.

        Args:
            id_event (str): event ID.
            bands_list (list, optional): bands list. If None, all bands are used and sorted according to the datasheet order.
                                         Defaults to None.
            cfg_file_dict (dict, optional): dictionary containing paths to the different end2end directories.
                                            If None, internal CSV database will be parsed.
            id_l0_l1_dict (dict, optional): id-l0-l1 dictionary. If None, internal CSV database will be parsed.
            reproject_bounds (bool, optional): if True, bounds are reprojected to EPGS:4326.
            verbose (bool, optional): if True, if True, verbose mode is used. Defaults to True.
            database (string, optional): database name. Defaults to "THRAWS".
        """

        self.__event_name = id_event

        if reproject_bounds:
            crs = "EPSG:4326:"
        else:
            crs = "EPSG:32633:"

        if bands_list is None:
            bands_list = list(BAND_SPATIAL_RESOLUTION_DICT.keys())

        (
            tiles_collection_dict,
            tiles_coordinates_dict,
            footprint_coordinates_dict,
            bands_file_name_dict_dict,
            event_class,
        ) = read_L1C_event_from_database(
            id_event,
            bands_list,
            cfg_file_dict,
            id_l0_l1_dict,
            database,
            reproject_bounds,
            verbose,
            device=self.__device,
        )
        tiles_names = list(tiles_collection_dict.keys())
        self.__bands_names = bands_list
        self.__event_class = event_class

        for tile_name in tiles_names:
            tile = tiles_collection_dict[tile_name]
            new_tile = L1C_tile(device=self.__device)
            new_tile.create_tile(
                bands_list,
                tile,
                tile_name,
                tiles_coordinates_dict[tile_name],
                footprint_coordinates_dict[tile_name],
                bands_file_name_dict_dict[tile_name],
                crs,
            )
            self.__tiles_collection.append(new_tile)

        self.n_tiles = len(self.__tiles_collection)

    def get_tile(self, tile_idx):
        """It returns the tile addressed by tile_idx.

        Args:
            tile_idx (int): tile index.
        """
        return self.__tiles_collection[tile_idx]

    def show_tiles_info(self):
        """Print tiles info."""
        tiles_info = self.get_tiles_info()
        tiles_names = list(tiles_info.keys())

        for n in range(len(tiles_names)):
            print(
                colored(
                    "------------------Tile "
                    + str(n)
                    + " ----------------------------",
                    "blue",
                )
            )
            print("Name: ", colored(tiles_info[tiles_names[n]][0], "red"))
            print("Sensing time: ", colored(tiles_info[tiles_names[n]][1], "red"))
            print("Creation time: ", colored(tiles_info[tiles_names[n]][2], "red"))
            coordinates = tiles_info[tiles_names[n]][3]
            footprint_coordinates = tiles_info[tiles_names[n]][4]
            print("Corners coordinates: \n")
            for n in range(len(coordinates)):
                print(
                    colored("\tP_" + str(n), "blue")
                    + " : "
                    + colored(str(coordinates[n]) + "\n", "red")
                )
            print("\n")
            print("Footprint's coordinates: \n")
            for n in range(len(footprint_coordinates)):
                print(
                    colored("\tP_" + str(n), "blue")
                    + " : "
                    + colored(str(footprint_coordinates[n]) + "\n", "red")
                )
            print("\n")

    def get_tiles_names(self, tiles_idx=None):
        """Return names of the tiles requested through tiles_idx from tiles names.

        Args:
            tiles_idx (list, optional): list of tiles for which getting the names.
                                      If None, all the names of the tiles in the collection are returned.
                                      Defaults to None.

        Raises:
            ValueError: Empty tiles lists

        Returns:
            list: tiles' names.
        """
        tiles_names = []
        if len(self.__tiles_collection) == 0:
            raise ValueError("Empty tiles lists.")

        if tiles_idx is None:
            tiles_idx = range(len(self.__tiles_collection))

        for tile_idx in tiles_idx:
            tiles_names.append(self.get_tile(tile_idx).tile_name)

        return tiles_names

    def get_tiles_info(self, tiles_idx=None):
        """Return info of the tiles requested through tiles_idx from tiles names.

        Args:
            tiles_idx (list, optional): list of tiles for which getting the names.
            If None, all the names of the tiles in the collection are returned. Defaults to None.

        Raises:
            ValueError: Empty tiles lists

        Returns:
            dictionary: tiles name : tiles info
        """
        tiles_names = []
        tiles_info = []
        if len(self.__tiles_collection) == 0:
            raise ValueError("Empty tiles lists.")

        if tiles_idx is None:
            tiles_idx = range(len(self.__tiles_collection))

        for tile_idx in tiles_idx:
            tile = self.get_tile(tile_idx)
            tile_info = tile.get_tile_info()
            tiles_info.append(tile_info)
            tiles_names.append(tile_info[0])

        return dict(zip(tiles_names, tiles_info))

    def set_useful_tiles(self, useful_tiles_list):
        """useful_tiles_list (list): Set useful tiles from list."""
        self.__l0_useful_tiles_idx = useful_tiles_list

    def is_void(self):
        """Returns true if the image is void.

        Returns:
            bool: True if the image is void.
        """

        if len(self.__tiles_collection) or (self.__tiles_collection is None):
            return False
        else:
            return True

    def show_tiles(self):
        """Show tiles."""
        tiles_collections = self.__tiles_collection

        n_tiles = len(tiles_collections)
        n_rows = int(np.ceil(n_tiles / 3))

        if n_rows > 1:
            fig, ax = plt.subplots(n_rows, 3, figsize=(40, 40))
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(n_rows, 3)

        for n in range(n_rows * 3):
            if n_rows > 1:
                row = int(np.floor(n / 3))
                column = int(n - 3 * row)
                ax_plot = ax[row, column]
            else:
                ax_plot = ax[n]

            if n < n_tiles:
                tile_n = self.get_tile(n)
                tile_n_tensor = tile_n.as_tensor(downsampling=True)

                if self.__device == torch.device("cuda"):
                    ax_plot.imshow(tile_n_tensor.detach().cpu().numpy())
                else:
                    ax_plot.imshow(tile_n_tensor)

                ax_plot.grid(False)
                ax_plot.axis("off")
                ax_plot.set_title("Tile " + str(n), fontsize=30)
            else:
                ax_plot.set_axis_off()

    def merge_tiles_band(self, band_name, merge_path):
        """Merge tiles by performing a mosaic. The merged file is saved at path ""merge_path""

        Args:
            band_name (src): band name
            merge_path (src): output path to save the merged file.
        """
        src_files_to_mosaic = []
        N_granules = len(self.__tiles_collection)
        raster_dataset = []
        for n in range(N_granules):
            tile_n = self.get_tile(n)
            timestamp = get_timestamp()
            tmp_tif_filename = (
                tile_n.get_tile_info()[0]
                + "_"
                + band_name
                + "_"
                + timestamp
                + "_tmp.tif"
            )
            src_files_to_mosaic.append(tmp_tif_filename)
            band_n_raster = rasterio.open(
                tile_n.get_bands_file_name_dict()[band_name], "r"
            )
            band_n_reprojected = reproject_raster(band_n_raster, tmp_tif_filename)
            band_n_raster.close()
            raster_dataset.append(band_n_reprojected)

        def custom_merge_works(
            old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None
        ):
            old_data[:] = np.maximum(old_data, new_data)

        if len(raster_dataset) > 1:
            mosaic, out_trans = merge(raster_dataset, method=custom_merge_works)
        else:
            mosaic, out_trans = merge(raster_dataset)

        out_meta = raster_dataset[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": raster_dataset[0].crs,
            }
        )

        with rasterio.open(merge_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        for n in range(N_granules):
            raster_dataset[n].close()
            os.remove(src_files_to_mosaic[n])
            os.remove(src_files_to_mosaic[n] + ".aux.xml")

    def crop_tile(
        self,
        l0_granule_coordinates,
        out_dir_path=None,
        out_name_ending=None,
        lat_lon_format=True,
        verbose=True,
        cfg_file_dict=None,
        id_l0_l1_dict=None,
        database="THRAWS",
        overwrite=False,
    ):
        """Create a mosaic of l1 tiles and crop the mosaic by using the l0_granule_coordinates.
        The output file is saved as TIF file into the directory specified through ""out_dir_path"".

        Args:
            l0_granule_coordinates (list): list of coordinates of a referene l0_granule.
            out_dir_path (src, optional): path to the output directory containing the cropped TIF file.
                                          If None, default path is used.  Defaults to None.
            out_name_ending (src, optional): optional ending for the output name. Defaults to None.
            lat_lon_format (bool, optional): if True, coordinates points are in (LAT, LON) format. Defaults to True.
            verbose (bool, optional): if True, verbose mode is used. Defaults to True.
            cfg_file_dict (dict, optional): dictionary containing paths to the different end2end directories.
                                            If None, internal CSV database will be parsed.
            id_l0_l1_dict (dict, optional): id-l0-l1 dictionary. If None, internal CSV database will be parsed.
            database (string, optional): database name. Defaults to "THRAWS".
            overwrite (bool, optional): if True, the file is overwritten if exists, otherwise tile generation is skipped.
                                        Defaults to False.

        Returns:
            src: output file name
        """

        if out_dir_path is None:
            out_dir_path = get_l1C_image_default_path(
                self.__event_name, cfg_file_dict, id_l0_l1_dict, database
            )
        else:
            out_dir_path = os.path.join(out_dir_path, self.__event_name)

        # Create output dir if not existing
        os.makedirs(out_dir_path, exist_ok=True)

        if out_name_ending is not None:
            out_name = os.path.join(out_dir_path + "_" + out_name_ending + ".tif")
        else:
            out_name = os.path.join(out_dir_path + ".tif")

        if not (overwrite) and os.path.exists(out_name):
            print(
                colored("Warning:", "red"),
                "the file "
                + colored(out_name, "blue")
                + " is already existing. The L1-C crop generation is skipped.",
            )
            return out_name

        if lat_lon_format:
            l0_granule_coordinates = [(y, x) for (x, y) in l0_granule_coordinates]
        l0_granule_polygon = Polygon(l0_granule_coordinates)

        timestamp = get_timestamp()
        merged_tile_path = self.__event_name + "_merged_tile_" + timestamp

        bands_to_merge = self.get_tile(0).get_bands_names()
        band_cropped_list = []
        band_meta_list = []

        if verbose:
            band_range = tqdm(bands_to_merge, "Processing bands...")
        else:
            band_range = bands_to_merge
        for band_name in band_range:
            if verbose:
                print(
                    "Creating a mosaic of different tiles for band: "
                    + colored(band_name, "red")
                )
            merged_tile_path_band = merged_tile_path + "_" + band_name + ".tif"

            self.merge_tiles_band(band_name, merged_tile_path_band)
            if verbose:
                print("Cropping mosaic file for band:" + colored(band_name, "red"))

            with rasterio.open(merged_tile_path_band) as src:
                out_image, out_transform = mask(src, [l0_granule_polygon], crop=True)
                out_meta = src.meta
            # Removing temporary mosaic crop
            os.remove(merged_tile_path_band)
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )
            band_cropped_list.append(out_image)
            band_meta_list.append(out_meta)

        # BANDS DOWNSAMPLING TO THE ONE WITH LOWEST RESOLUTION
        bands_resolution_list = [
            BAND_SPATIAL_RESOLUTION_DICT[b] for b in bands_to_merge
        ]
        lowest_resolution = max(bands_resolution_list)
        band_cropped_list_resampled = []
        band_resolution_index = 0
        for band_name, n in zip(bands_to_merge, range(len(bands_to_merge))):
            band = band_cropped_list[n]
            band_resolution = BAND_SPATIAL_RESOLUTION_DICT[band_name]
            if band_resolution != lowest_resolution:
                usf_v = int(lowest_resolution / band_resolution)
                usf_h = int(lowest_resolution / band_resolution)
                band = band[::usf_v, ::usf_h]
            else:
                band_resolution_index = n

            band_cropped_list_resampled.append(band)
        band_meta_list[band_resolution_index].update({"count": len(band_cropped_list)})
        with rasterio.open(
            out_name, "w", **band_meta_list[band_resolution_index]
        ) as dest:
            for band, n in zip(
                band_cropped_list_resampled, range(len(band_cropped_list_resampled))
            ):
                dest.write(band[0, :, :], n + 1)

        return out_name
