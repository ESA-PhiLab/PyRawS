import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from ..utils.constants import (
    BAND_SPATIAL_RESOLUTION_DICT,
    S2_DEFAULT_QUANTIFICATION_VALUE,
)
from ..utils.band_shape_utils import image_band_upsample
from ..utils.date_utils import parse_string_date
from ..utils.l1_utils import export_band_to_tif, swap_latlon


class L1C_tile:
    # ------------ PRIVATE VARIABLES ------------

    __bands_dict = None  # Bands dictionary

    __device = None  # Device

    __bands_names = None  # Bands's names

    __tile_name = None  # tile's name

    __crs = None  # CRS

    __tile_coordinates = None  # tile's corners coordinates.

    __tile_footprint_coordinates = None  # tile's footprints.

    __bands_file_name_dict = None  # bands filename dictionary

    def __init__(
        self,
        tile_bands=None,
        bands_names=None,
        tile_name=None,
        tile_coordinates=None,
        tile_footprint_coordinates=None,
        bands_file_name_dict=None,
        crs=None,
        device=torch.device("cpu"),
    ):
        """Initialize an L0 tile.

        Args:
            tile_bands (list, optional): list of torch tensors representing each band in the format [H,W].
                                         Defaults to None.
            bands_names (list, optional): list of band names. Defaults to None.
            tile_name (string, optionl): tile name. Defaults to None.
            tile_coordinates (list, optional): list containing [lan, lon] for every corner point.
                                               Defaults to None.
            tile_footprint_coordinates (list, optional): list containing [lan, lon] for every point of the tile.
                                                         Defaults to None.
            bands_file_name_dict (dict, optional): dictionary associating the orginal jp2 file to each band.
                                                   Defaults to None.
            csr (string, optional): crs. Defaults to None.
            device (torch.device, optional): torch.device. Defaults to torch.device("cpu").
        """

        if (tile_bands is not None) and (bands_names is not None):
            self.__bands_dict = dict(zip(bands_names, tile_bands))
        else:
            self.__bands_dict = None

        self.__device = device

        self.__bands_names = bands_names

        self.tile_name = tile_name

        if tile_coordinates is None:
            self.__tile_coordinates = []
        else:
            self.__tile_coordinates = tile_coordinates

        if tile_footprint_coordinates is None:
            self.__tile_footprint_coordinates = []
        else:
            self.__tile_footprint_coordinates = tile_footprint_coordinates

        self.__bands_file_name_dict = bands_file_name_dict

        self.__crs = crs

    # ------------PUBLIC METHODS ------------

    def get_tile_coordinates(self, latlon_format=True):
        """Returns the tile's coordinates.
        Args:
            latlon_format (bool, optional): if True, LAT, LON format is used. Defaults to True.

        Returns:
            list: polygon coordinates.
        """
        if latlon_format:
            return self.__tile_coordinates
        else:
            return swap_latlon(self.__tile_coordinates)

    def get_bands_names(self):
        """Returns band names

        Returns:
            list: list of band names.
        """
        return self.__bands_names

    def get_bands_file_name_dict(self):
        """Returns the tile's band_name jp2 dictionary.

        Returns:
            dict: tiles bands_file_name_dict.
        """
        return self.__bands_file_name_dict

    def get_tile_footprint_coordinates(self):
        """Returns the tile's footprint coordinates.

        Returns:
            list: polygon coordinates.
        """
        return self.__tile_footprint_coordinates

    def get_band(self, band_name):
        """Returns a specific band as a tensor.

        Args:
            band_name (string): band name

        Returns:
            torch.tensor: requested band.
        """
        return self.__bands_dict[band_name]

    def export_to_tif(self, save_path):
        """Export to TIF file.

        Args:
            save_path (str): save path.
            downsampling (bool, optional): If True, bands are downsampled. Defaults to True.
        """
        coords = self.get_tile_coordinates(latlon_format=False)

        for band_name in list(self.__bands_dict.keys()):
            band = self.__bands_dict[band_name] * S2_DEFAULT_QUANTIFICATION_VALUE
            export_band_to_tif(
                band, self.__crs, coords, os.path.join(save_path, band_name + ".tif")
            )

    def create_tile(
        self,
        bands_list,
        tile_bands,
        tile_name,
        tile_coordinates,
        footprint_coordinates,
        bands_file_name_dict,
        crs,
    ):
        """Creates a tile from the list of bands and from a torch tensor.

        Args:
            bands_list (list): list of band names.
            tile_bands (list): list of torch.tensors (bands)  in format [H,W]
            tile_name (str): tile name.
            tile_coordinates (list): list of tile's corners coordinates.
            footprint_coordinates (list): list of tile's footprint's coordinates.
            bands_file_name_dict (dict): bands_file_name_dict.
            crs (string): crs
        """
        self.__bands_dict = dict(zip(bands_list, bands_list))

        for band, n in zip(bands_list, range(len(bands_list))):
            self.__bands_dict[band] = tile_bands[n].to(self.__device)

        self.__bands_names = bands_list
        self.tile_name = tile_name
        self.__tile_coordinates = tile_coordinates
        self.__tile_footprint_coordinates = footprint_coordinates
        self.__bands_file_name_dict = bands_file_name_dict
        self.__crs = crs

    def get_tile_info(self):
        """Returns name, sensing time, acquisition time, and coners coordinates.

        Returns:
            string: name
            datetime: sensing time.
            datetime: acquisition time.
            list: tile's corners' coordinates.
            list: tile's footprint's coordinates.
        """
        tile_name = self.tile_name

        sensing_time = parse_string_date(tile_name[-15:])
        acquisition_time = parse_string_date(tile_name[11:26])

        return (
            tile_name,
            sensing_time,
            acquisition_time,
            self.get_tile_coordinates(),
            self.get_tile_footprint_coordinates(),
        )

    def as_tensor(self, requested_bands=None, downsampling=True):
        """Returns a tensor containing all the bands if all the requested bands have the same resolution.

        Args:
            requested_bands (list, optional): list of requested bands. If None, all the bands are used. Defaults to None.
            downsampling (boolean, optional): if True, bands are downsampled to the lowest resolution.
                                              Otherwise, they are upsampled to the highest one. Defaults to True.

        Raises:
            ValueError: The tile is empty

        Returns:
            torch.tensor: tensor containing the requested bands.
        """
        if self.__bands_dict is None:
            raise ValueError("The tile is empty.")

        if requested_bands is None:
            requested_bands = self.__bands_names

        if downsampling:
            bands_resolution_list = [
                BAND_SPATIAL_RESOLUTION_DICT[b] for b in requested_bands
            ]
            lowest_resolution = max(bands_resolution_list)

            band_shape = [self.get_band(b) for b in requested_bands][
                bands_resolution_list.index(lowest_resolution)
            ].shape
            bands = torch.zeros(
                [band_shape[0], band_shape[1], len(requested_bands)],
                device=self.__device,
            )

            for band_name, n in zip(requested_bands, range(len(requested_bands))):
                band = self.__bands_dict[band_name]
                band_resolution = BAND_SPATIAL_RESOLUTION_DICT[band_name]
                if band_resolution != lowest_resolution:
                    usf_v = int(lowest_resolution / band_resolution)
                    usf_h = int(lowest_resolution / band_resolution)

                    band = band[::usf_v, ::usf_h]
                bands[:, :, n] = band

        else:
            resolution_min = min(
                [BAND_SPATIAL_RESOLUTION_DICT[b] for b in requested_bands]
            )

            band_shape = [
                max(self.get_band(b).shape[0] for b in requested_bands),
                max(self.get_band(b).shape[1] for b in requested_bands),
            ]

            bands = torch.zeros(
                [band_shape[0], band_shape[1], len(requested_bands)],
                device=self.__device,
            )

            for band_name, n in zip(requested_bands, range(len(requested_bands))):
                band = self.__bands_dict[band_name]

                if resolution_min != BAND_SPATIAL_RESOLUTION_DICT[band_name]:
                    band_upsampled = image_band_upsample(
                        band, band_name, resolution_min, "bilinear"
                    )
                    if bands.shape[0] > band_upsampled.shape[0]:
                        bands[: band_upsampled.shape[0], :, n] = band_upsampled
                    elif bands.shape[0] < band_upsampled.shape[0]:
                        bands[:, :, n] = band_upsampled[: bands.shape[0], :]
                    else:
                        bands[:, :, n] = band_upsampled
                else:
                    bands[:, :, n] = band

        return bands

    def show_bands_superimposition(self, requested_bands=None, downsampling=True):
        """It shows the superimposition of bands in requested_bands

        Args:
            requested_bands (list, optional): requested bands list. If None, all the bands are used. Defaults to None.
            downsampling (boolean, optional): if True, bands are downsampled to have the same resolution. Default to False.
        Raises:
            ValueError: Impossible to superimpose more than 3 bands
        """
        if requested_bands is None:
            requested_bands = self.__bands_names

        if len(requested_bands) > 3:
            raise ValueError("Impossible to superimpose more than 3 bands.")

        bands_superimposed = self.as_tensor(requested_bands, downsampling=downsampling)

        if bands_superimposed.shape[2] == 2:
            bands_superimposed_zeros = torch.zeros(
                [bands_superimposed.shape[0], bands_superimposed.shape[1], 3]
            )
            bands_superimposed_zeros[:, :, :2] = bands_superimposed
            bands_superimposed_zeros[:, :, 2] = (
                bands_superimposed[:, :, 0] + bands_superimposed[:, :, 1]
            ) / 2
            bands_superimposed = bands_superimposed_zeros

        bands_superimposed = bands_superimposed / bands_superimposed.max()

        if self.__device == torch.device("cuda"):
            plt.imshow(bands_superimposed.detach().cpu().numpy())
        else:
            plt.imshow(bands_superimposed)

    def show_bands(self, requested_bands=None, downsampling=False, oversampling=False):
        """It shows the requested bands.

        Args:
            requested_bands (list, optional): list of requested bands to show.
                                             If None, all the bands are shown. Defaults to None.
            downsampling (bool, optional): if True, bands are downsampled to have the same resolution.
                                           Default to False.
            oversampling (bool, optional): if True, bands are oversampled to have the same resolution.
                                           Downsampling has priority over upampling. Default to False.
        """
        if requested_bands is None:
            requested_bands = self.__bands_names

        n_bands = len(requested_bands)
        n_rows = int(np.ceil(n_bands / 3))

        if n_rows > 1:
            fig, ax = plt.subplots(n_rows, 3, figsize=(40, 40))
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(n_rows, 3)

        if downsampling:
            n_printed = 0
            tile_tensor = self.as_tensor(downsampling=True)
            for n in range(len(self.__bands_names)):
                if self.__bands_names[n] in requested_bands:
                    if n_rows > 1:
                        row = int(np.floor(n / 3))
                        column = int(n_printed - 3 * row)
                        ax_plot = ax[row, column]
                    else:
                        ax_plot = ax[n_printed]

                    band_name = self.__bands_names[n]
                    band = tile_tensor[:, :, n]

                    if self.__device == torch.device("cuda"):
                        ax_plot.imshow(band.detach().cpu().numpy())
                    else:
                        ax_plot.imshow(band)

                    ax_plot.grid(False)
                    ax_plot.axis("off")
                    ax_plot.set_title(band_name, fontsize=30)
                    n_printed += 1

            if n_rows > 1:
                for n in range(n_printed, 3 * n_rows):
                    ax_plot = ax[n_rows - 1, n - n_printed + 1]
                    ax_plot.set_axis_off()
            else:
                for n in range(n_printed, 3 * n_rows):
                    ax_plot = ax[n]
                    ax_plot.set_axis_off()

        elif (oversampling) and not (downsampling):
            n_printed = 0
            tile_tensor = self.as_tensor(downsampling=False)
            for n in range(len(self.__bands_names)):
                if self.__bands_names[n] in requested_bands:
                    if n_rows > 1:
                        row = int(np.floor(n / 3))
                        column = int(n_printed - 3 * row)
                        ax_plot = ax[row, column]
                    else:
                        ax_plot = ax[n_printed]

                    band_name = self.__bands_names[n]

                    band = tile_tensor[:, :, n]

                    if self.__device == torch.device("cuda"):
                        ax_plot.imshow(band.detach().cpu().numpy())
                    else:
                        ax_plot.imshow(band)

                    ax_plot.grid(False)
                    ax_plot.axis("off")
                    ax_plot.set_title(band_name, fontsize=30)
                    n_printed += 1

            if n_rows > 1:
                for n in range(n_printed, 3 * n_rows):
                    ax_plot = ax[n_rows - 1, n - n_printed + 1]
                    ax_plot.set_axis_off()
            else:
                for n in range(n_printed, 3):
                    ax_plot = ax[n]
                    ax_plot.set_axis_off()

        else:
            for n in range(n_rows * 3):
                if n_rows > 1:
                    row = int(np.floor(n / 3))
                    column = int(n - 3 * row)
                    ax_plot = ax[row, column]
                else:
                    ax_plot = ax[n]

                if n < n_bands:
                    band_name = requested_bands[n]
                    band = self.get_band(band_name)

                    if self.__device == torch.device("cuda"):
                        ax_plot.imshow(band.detach().cpu().numpy())
                    else:
                        ax_plot.imshow(band)

                    ax_plot.grid(False)
                    ax_plot.axis("off")
                    ax_plot.title.set_text(band_name)
                    ax_plot.set_title(band_name, fontsize=30)
                else:
                    ax_plot.set_axis_off()
