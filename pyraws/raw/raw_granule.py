import torch
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import rotate
import geopy.distance
from copy import deepcopy
from shapely.geometry import Polygon
import os

from ..utils.constants import (
    BAND_SPATIAL_RESOLUTION_DICT,
    BAND_SPATIAL_RESOLUTION_DICT_ACROSS,
    BANDS_RAW_SHAPE_DICT,
    SWIR_BANDS,
)
from ..utils.raw_utils import (
    export_band_to_tif,
    get_bands_shift,
    get_granule_px_length,
    swap_latlon,
)
from ..utils.visualization_utils import equalize_tensor
from ..utils.band_shape_utils import image_band_upsample
from ..utils.date_utils import parse_string_date
import cv2


class Raw_granule:
    # ---------------- PRIVATE VARIABLES ----------------
    __bands_dict = None  # Bands dictionary
    __device = None  # Device
    __bands_names = None  # Bands's names
    __original = (
        True  # This parameter is used to keep track of the history of the granule.
    )
    # If it is True, it means the granule was created by reading an image, and no operation is applied.
    __granule_parents = (
        []
    )  # It is used to keep track of the parents granules originating this granule.

    __granule_polygon_coordinates = (
        []
    )  # Coordinates of the polygon interstecting granule area.

    __single_granule_along_track_size = None  # Along track size of a single granule.
    __cloud_percentage_list = []  # Cloud coverage percentage
    __cropped_pixels = [None, None]  # Cropped pixels along-across

    def __init__(
        self,
        granule_bands=None,
        bands_names=None,
        granule_name=None,
        granule_polygon_coordinates=None,
        single_granule_along_track_size=None,
        cloud_percentage=None,
        device=torch.device("cpu"),
    ):
        """Initialize an raw granule.

        Args:
            granule_bands (list, optional): list of torch tensors representing each band in the format [H,W]. Defaults to None.
            bands_names (list, optional): list of band names. Defaults to None.
            granule_name (string, optionl): granule name. Defaults to None.
            granule_polygon_coordinates (list, optional): list containing [lat, lon] for every polygon point. Defaults to None.
            single_granule_along_track_size (float, optional): size of a single granule along track.
                                                                Useful to keep this info when the granule has parents.
                                                                Defaults to None.
            cloud_percentage (list or float, optional): Cloud coverage percentage metadata. Defaults to None.
            device (torch.device, optional): torch.device. Defaults to torch.device("cpu").
        """

        if (granule_bands is not None) and (bands_names is not None):
            self.__bands_dict = dict(zip(bands_names, granule_bands))
        else:
            self.__bands_dict = None

        self.__device = device

        self.__bands_names = bands_names

        self.granule_name = granule_name

        self.__cropped_pixels = [None, None]

        self.__original = (
            True  # This parameter is used to keep track of the history of the granule.
        )
        # If it is True, it means the granule was created by reading an image and no operation is applied.

        self.__granule_parents = (
            []
        )  # It is used to keep track of the parents granules originating this granule.

        if granule_polygon_coordinates is None:
            self.__granule_polygon_coordinates = (
                []
            )  # Coordinates of the polygon interstecting granule area.
        else:
            self.__granule_polygon_coordinates = granule_polygon_coordinates

        self.__single_granule_along_track_size = single_granule_along_track_size

        if cloud_percentage is None:
            self.__cloud_percentage_list = []
        elif isinstance(cloud_percentage, list):
            self.__cloud_percentage_list = cloud_percentage
        else:
            self.__cloud_percentage_list.append(cloud_percentage)

    # ----------------------------------PRIVATE METHODS ----------------------------------

    def __set_cropped_pixels(self, cropped_pixels):
        """Set the new granule length.

        Args:
            cropped_pixels (list): the number of pixels cropped along-across track
        """
        self.__cropped_pixels = cropped_pixels

    def __set_single_granule_along_track_size(self, along_track_size):
        """Single track along track size.

        Args:
            along_track_size (float): single granule along track size
        """
        self.__single_granule_along_track_size = along_track_size

    def __set_originality(self, originality):
        """Set originality of granule.

        Args:
            originality (boolean): originality.
        """
        self.__original = originality

    def __set_parents(self, parents_list):
        """Set granule parents from parents list

        Args:
            parents_list (list): parents list.
        """
        for granule_parent in parents_list:
            self.__granule_parents.append(granule_parent)

    def __set_cloud_percentage(self, cloud_percentage_list):
        """Set the cloud percentage.

        Args:
            cloud_percentage_list (list): cloud percentage list.
        """
        self.__cloud_percentage_list = cloud_percentage_list

    # ---------------------- PUBLIC METHODS -----------------------

    def is_original(self):
        """It shows if the granule is original.
        Returns:
            boolean: originality.
        """
        return self.__original

    def get_parents(self):
        """Set granule parents from parents list

        Returns:
            list: parents list.
        """
        parents_list = []
        for granule_parent in self.__granule_parents:
            parents_list.append(granule_parent)

        return parents_list

    def get_band(self, band_name):
        """Returns a specific band as a tensor.

        Args:
            band_name (string): band name

        Returns:
            torch.tensor: requested band.
        """
        return self.__bands_dict[band_name]

    def show_raw_band(self, band_name):
        """Show a granule band specified by band_name.

        Args:
            band_name (string): band_name

        Raises:
            ValueError: Impossible to show the requested band: band_name
            ValueError: The requested granule is void.
        """

        if self.__bands_dict is not None:
            band = self.__bands_dict[band_name]
            try:
                if self.__device == torch.device("cuda"):
                    plt.imshow(band.detach().cpu().numpy())
                else:
                    plt.imshow(band)
            except:  # noqa: E722
                raise ValueError(
                    "Impossible to show the requested band "
                    + colored(band_name, "red")
                    + "."
                )
        else:
            raise ValueError("The requested granule is void.")

    def get_granule_coordinates(self, latlon_format=True):
        """Returns the granule's coordinates.

        Args:
            latlon_format (bool): format to use to get the granule's coordinates. Default to: Latitude, Longitude

        Returns:
            list: polygon coordinates.
        """

        if latlon_format:
            return self.__granule_polygon_coordinates
        else:
            return swap_latlon(self.__granule_polygon_coordinates)

    def get_single_granule_along_track_size(self):
        """Retuns single granule along track size.

        Returns:
            float: single ganule along track size
        """
        return self.__single_granule_along_track_size

    def get_along_track_size(self):
        """Returns along track size.

        Returns:
            float: along track size.
        """
        return geopy.distance.geodesic(
            self.__granule_polygon_coordinates[0], self.__granule_polygon_coordinates[1]
        ).km

    def get_across_track_size(self):
        """Returns across track size.

        Returns:
            float: across track size
        """
        return geopy.distance.geodesic(
            self.__granule_polygon_coordinates[1], self.__granule_polygon_coordinates[2]
        ).km

    def get_cloud_percentage(self):
        """Returns cloud percentage.

        Returns:
            list: cloud percentage.
        """
        return self.__cloud_percentage_list

    def create_granule(
        self,
        bands_list,
        granule_bands,
        granule_name,
        polygon_coordinates,
        single_granule_along_track_size,
        cloud_percentage,
    ):
        """Creates a granule from the list of bands and from a torch tensor.

        Args:
            bands_list (list): list of band names.
            granule_bands (list): list of torch.tensors (bands)  in format [H,W]
            granule_name (str): granule name.
            polygon_coordinates (list): list of [lan, lon] coordinates for every polygon point.
            single_granule_along_track_size (float): along track size for a single granule.
            cloud_percentage (float or list): cloud percentage.
        """
        self.__bands_dict = dict(zip(bands_list, bands_list))

        for band, n in zip(bands_list, range(len(bands_list))):
            self.__bands_dict[band] = granule_bands[n].to(self.__device)

        self.__bands_names = bands_list
        self.granule_name = granule_name
        self.__granule_polygon_coordinates = polygon_coordinates
        self.__single_granule_along_track_size = single_granule_along_track_size
        if isinstance(cloud_percentage, list):
            self.__cloud_percentage_list = cloud_percentage
        else:
            self.__cloud_percentage_list.append(cloud_percentage)

    def check_bands_size(self, requested_bands):
        """Checks if the requested bands have the same size

        Args:
            requested_bands (list): list of requested bands

        Returns:
            boolean: return True if all the requested bands have the same size. It returns False, otherwise.
        """

        band_shape = self.__bands_dict[requested_bands[0]].shape

        ok_flag = True
        for band_name in requested_bands:
            band = self.__bands_dict[band_name]
            if band.shape != band_shape:
                ok_flag = False

        if ok_flag:
            print(colored("Bands size is matching", "green"))
        else:
            print(colored("Bands size is not matching.", "red"))

        return ok_flag

    def as_tensor(self, requested_bands=None, downsampling=True):
        """Returns a tensor containing all the bands if all the requested bands have the same resolution.

        Args:
            requested_bands (list, optional): list of requested bands.
                                            If None, all the bands are used. Defaults to None.
            downsampling (boolean, optional): if True, bands are downsampled to the lowest resolution.
                                            Otherwise, they are upsampled to the highest one.
                                            Defaults to True.

        Raises:
            ValueError: The granule is empty

        Returns:
            torch.tensor: tensor containing the requested bands.
        """
        if self.__bands_dict is None:
            raise ValueError("The granule is empty.")

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

                    if lowest_resolution == 60:
                        usf_h = max(int(20 / (band_resolution)), 1)
                    else:
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
                    if BAND_SPATIAL_RESOLUTION_DICT[band_name] == 60:
                        band_upsampled = image_band_upsample(
                            band[:, ::3], band_name, resolution_min, "bicubic"
                        )
                    else:
                        band_upsampled = image_band_upsample(
                            band, band_name, resolution_min, "bicubic"
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

    def stack_to_granule(self, granule_other, position):
        """It stacks the granule to another input granule to a specificied position.
        For instance, if position == ""T"", the granule will be stacked at the top of the ""granule_other"".

        Args:
            granule_other (Raw_granule): other granule to which the input granule will be stacked to.
            position (string): Stacking poistion. Only [""T"", ""B"", ""R"", ""L""] are supported.

        Raises:
            ValueError: The granule is void.
            ValueError: Impossible to stack granules. Granules have different bands.
            ValueError: Only the following stacking positions are supported: [""T"", ""B"", ""R"", ""L""].

        Returns:
            Raw_granule: stacked granule
        """
        if self.__bands_dict is None:
            raise ValueError("The granule is void.")

        bands_names = self.__bands_names
        if bands_names != granule_other.__bands_names:
            raise ValueError(
                "Impossible to stack granules. Granules have different bands."
            )

        if position not in ["T", "B", "R", "L"]:
            raise ValueError(
                "Only the following stacking positions are supported: ["
                "T"
                ", "
                "B"
                ", "
                "R"
                ", "
                "L"
                "]."
            )

        granule_stacked_bands = []

        for band_name in bands_names:
            granule_1_band = self.__bands_dict[band_name]
            granule_2_band = granule_other.__bands_dict[band_name]

            if granule_1_band.device == torch.device("cuda"):
                granule_2_band = granule_2_band.cuda()
            else:
                granule_2_band = granule_2_band.cpu()

            if position == "T":
                granule_band = torch.zeros(
                    [
                        granule_1_band.shape[0] + granule_2_band.shape[0],
                        granule_1_band.shape[1],
                    ]
                )
                granule_band[: granule_1_band.shape[0]] = granule_1_band
                granule_band[granule_1_band.shape[0] :] = granule_2_band

            elif position == "B":
                granule_band = torch.zeros(
                    [
                        granule_2_band.shape[0] + granule_1_band.shape[0],
                        granule_1_band.shape[1],
                    ]
                )
                granule_band[: granule_2_band.shape[0]] = granule_2_band
                granule_band[granule_2_band.shape[0] :] = granule_1_band
            elif position == "L":
                granule_band = torch.zeros(
                    [
                        granule_1_band.shape[0],
                        granule_1_band.shape[0] + granule_2_band.shape[0],
                    ]
                )
                granule_band[:, : granule_1_band.shape[0]] = granule_1_band
                granule_band[:, granule_1_band.shape[0] :] = granule_2_band
            else:
                granule_band = torch.zeros(
                    [
                        granule_1_band.shape[0],
                        granule_2_band.shape[0] + granule_1_band.shape[0],
                    ]
                )
                granule_band[:, : granule_2_band.shape[0]] = granule_2_band
                granule_band[:, granule_2_band.shape[0] :] = granule_1_band

            granule_stacked_bands.append(granule_band)

        granule_name = (
            self.granule_name
            + "_STACKED_"
            + position
            + "_"
            + granule_other.granule_name
        )
        granule_other_coordinates = granule_other.get_granule_coordinates()
        granule_this_coordinates = self.get_granule_coordinates()
        granule_this_cloud_percentage = self.get_cloud_percentage()
        granule_other_cloud_percentage = granule_other.get_cloud_percentage()

        if position == "T":
            granule_coordinates_polygon = [
                granule_this_coordinates[0],
                granule_other_coordinates[1],
                granule_other_coordinates[2],
                granule_this_coordinates[3],
            ]
            stacked_granule_cloud_percentage = [
                granule_this_cloud_percentage,
                granule_other_cloud_percentage,
            ]

        elif position == "B":
            granule_coordinates_polygon = [
                granule_other_coordinates[0],
                granule_this_coordinates[1],
                granule_this_coordinates[2],
                granule_other_coordinates[3],
            ]
            stacked_granule_cloud_percentage = [
                granule_other_cloud_percentage,
                granule_this_cloud_percentage,
            ]
        elif position == "L":
            granule_coordinates_polygon = [
                granule_this_coordinates[0],
                granule_this_coordinates[1],
                granule_other_coordinates[2],
                granule_other_coordinates[3],
            ]
            stacked_granule_cloud_percentage = [
                granule_this_cloud_percentage,
                granule_other_cloud_percentage,
            ]
        else:
            granule_coordinates_polygon = [
                granule_other_coordinates[0],
                granule_other_coordinates[1],
                granule_this_coordinates[2],
                granule_this_coordinates[3],
            ]
            stacked_granule_cloud_percentage = [
                granule_other_cloud_percentage,
                granule_this_cloud_percentage,
            ]

        stacked_granule = Raw_granule(
            granule_bands=granule_stacked_bands,
            bands_names=bands_names,
            granule_name=granule_name,
            granule_polygon_coordinates=granule_coordinates_polygon,
            cloud_percentage=stacked_granule_cloud_percentage,
            device=self.__device,
        )

        stacked_granule.__set_originality(False)

        stacked_granule.__set_single_granule_along_track_size(
            self.__single_granule_along_track_size
        )

        if len(self.__granule_parents) == 0:
            stacked_granule.__set_parents(
                [self.granule_name, granule_other.granule_name]
            )
        else:
            stacked_granule.__set_parents(
                self.__granule_parents + [granule_other.granule_name]
            )

        return stacked_granule

    def rotate_band(self, requested_band):
        """Rotates one band of 180 degrees.

        Args:
            requested_band (string): band name.
        """
        band = self.get_band(requested_band)
        self.__bands_dict[requested_band] = rotate(band.unsqueeze(2), 180).squeeze(2)
        self.__set_originality(False)

    # def __update_coordinates_first_band(band_name):

    def get_detectors_number(self):
        """Returns the detector numbers list.

        Returns:
            list: List of detector numbers.
        """
        granule_detector_info = self.get_granule_info()
        return granule_detector_info[3]

    def coarse_coregistration(
        self,
        rotate_swir_bands=True,
        granule_filler_before=None,
        granule_filler_after=None,
        crop_empty_pixels=False,
        downsampling=True,
        bands_shifts=None,
        verbose=False,
    ):
        """It implements the coarse coregistration of the bands by compensating
        the along-track pixels shift with respect to the first band.

        Args:
            rotate_swir_bands (boolean, optional): if True, SWIR bands are rotated before applying coregistration.
                                                  Defaults to True.
            granule_filler_before (Raw_granule, optional): if not None, bands of this tile will be used to fill
                                                           the missing elements during the coregistration as before granule.
                                                           Defaults to None.
            granule_filler_after (Raw_granule, optional): if not None, bands of this tile will be used to fill the missing
                                                          elements during the coregistration as bottom granule.
                                                          Defaults to None.
            crop_empty_pixels (boolean, optional): if True and no fillers are available or granule fillers are None,
                                                   the image will be crop at the bottom and the top of a number
                                                   of pixels equal to the maximum zeros pixels stacked from each band.
            downsampling (boolean, optional): if True, higher resolution bands will be undersampled to match the bands
                                              with the lowest resolution.
                                              If False, lower resolution bands will be upsampled to match the bands
                                              with the highest resolution. Defaults to True.
            bands_shifts (list, optional): bands shift values compared to the first band.
                                           If None, they will be read by the LUT file. Defaults to None.
            verbose (boolean, optional): if True, verbose mode is used. Defaults to False.
        Returns:
            Raw_granule: granule with coarse-coregistered bands.
        """

        def shift_band(band, shifts):
            band_shifted = torch.zeros_like(band)
            shifts = [int(shifts[0]), int(shifts[1])]
            if (shifts[0] == 0) and (shifts[1] == 0):
                band_shifted = band
            elif (shifts[0] == 0) and (shifts[1] < 0):
                band_shifted[:, : int(shifts[1])] = band[:, -int(shifts[1]) :]
            elif (shifts[0] == 0) and (shifts[1] > 0):
                band_shifted[:, int(shifts[1]) :] = band[:, : -int(shifts[1])]
            elif (shifts[0] < 0) and (shifts[1] == 0):
                band_shifted[: int(shifts[0]), :] = band[-int(shifts[0]) :, :]
            elif (shifts[0] > 0) and (shifts[1] == 0):
                band_shifted[int(shifts[0]) :, :] = band[: -int(shifts[0]), :]
            elif (shifts[0] > 0) and (shifts[1] > 0):
                band_shifted[int(shifts[0]) :, int(shifts[1]) :] = band[
                    : -int(shifts[0]), : -int(shifts[1])
                ]
            elif (shifts[0] > 0) and (shifts[1] < 0):
                band_shifted[int(shifts[0]) :, : int(shifts[1])] = band[
                    : -int(shifts[0]), -int(shifts[1]) :
                ]
            elif (shifts[0] < 0) and (shifts[1] > 0):
                band_shifted[: int(shifts[0]), int(shifts[1]) :] = band[
                    -int(shifts[0]) :, : -int(shifts[1])
                ]
            else:
                band_shifted[: int(shifts[0]), : int(shifts[1])] = band[
                    -int(shifts[0]) :, -int(shifts[1]) :
                ]

            return band_shifted

        def shifts2TBLT(along_track: int, cross_track: int):
            """Transforms along and cross track offsets into top, bottom, left and right offsets
            Args:
                along_track (int): The offset in the along track direction
                cross_track (int): The offset in the cross track direction

            Returns:
                Shifts offset in top, bottom, left, right
            """
            top, left, right, bottom = 0, 0, 0, 0
            if along_track < 0:
                bottom = abs(along_track)
            elif along_track > 0:
                top = abs(along_track)
            if cross_track > 0:
                left = abs(cross_track)
            elif cross_track < 0:
                right = abs(cross_track)

            return top, bottom, left, right

        def maxOffsetUpdate(
            max_top: int,
            max_top_band: str,
            max_bottom: int,
            max_bottom_band: str,
            max_left: int,
            max_left_band: str,
            max_right: int,
            max_right_band: str,
            top: int,
            bottom: int,
            left: int,
            right: int,
            band: str,
        ):
            """Updates the maximum offsets in top, bottom, left, right directions
            Args:
                max_top (int): The maximum offset in the top direction
                max_top_band (str): band having the maximum top offset
                max_bottom (int): The maximum offset in the bottom direction
                max_bottom_band (str): band having the maximum top offset
                max_right (int): The maximum offset in the right direction
                max_left_band (str): band having the maximum top offset
                max_left (int): The maximum offset in the left direction
                max_right_band (str): band having the maximum top offset
                top (int): The current offset in the top direction
                bottom (int): The current offset in the bottom direction
                right (int): The current offset in the right direction
                left (int): The current offset in the left direction
                band (str): curent band name

            Returns:
                The updated maximum offsets in top, bottom, left, right directions,
                updated max_top, max_bottom, max_right, max_left
            """
            top_rescaled, bottom_rescaled = (
                BAND_SPATIAL_RESOLUTION_DICT[band]
                / BAND_SPATIAL_RESOLUTION_DICT[max_top_band]
                * top,
                BAND_SPATIAL_RESOLUTION_DICT[band]
                / BAND_SPATIAL_RESOLUTION_DICT[max_bottom_band]
                * bottom,
            )
            right_rescaled, left_rescaled = (
                min(BAND_SPATIAL_RESOLUTION_DICT[band], 20)
                / min(BAND_SPATIAL_RESOLUTION_DICT[max_right_band], 20)
                * right,
                min(BAND_SPATIAL_RESOLUTION_DICT[band], 20)
                / min(BAND_SPATIAL_RESOLUTION_DICT[max_left_band], 20)
                * left,
            )

            if top_rescaled > max_top:
                max_top = top
                max_top_band = band
            if bottom_rescaled > max_bottom:
                max_bottom = bottom
                max_bottom_band = band
            if right_rescaled > max_right:
                max_right = right
                max_right_band = band
            if left_rescaled > max_left:
                max_left = left
                max_left_band = band

            return (
                max_top,
                max_bottom,
                max_left,
                max_right,
                max_top_band,
                max_bottom_band,
                max_left_band,
                max_right_band,
            )

        granule_detector_number = self.get_detectors_number()[0]
        satellite = self.get_granule_info()[0][:3]
        if bands_shifts is None:
            bands_shifts = get_bands_shift(
                self.__bands_names,
                satellite,
                granule_detector_number,
                downsampling=downsampling,
                cfg_file_dict=None,
            )

            if verbose:
                print("Bands shifts: ", colored(bands_shifts, "green"))

        coregistered_bands = []
        max_top, max_bottom, max_left, max_right = (
            0,
            0,
            0,
            0,
        )  # Maximum of offsets init.

        max_top_band = self.__bands_names[0]
        max_bottom_band = self.__bands_names[0]
        max_left_band = self.__bands_names[0]
        max_right_band = self.__bands_names[0]

        # Assigning to None and changed later if needed.
        filler_name = {"top": None, "bottom": None}

        for n in range(len(self.__bands_names)):
            band = self.__bands_dict[self.__bands_names[n]]
            if (self.__bands_names[n] in SWIR_BANDS) and (rotate_swir_bands):
                band = rotate(band.unsqueeze(2), 180).squeeze(2)
            # No shift for the first band.
            if n == 0:
                coregistered_bands.append(band)  # MASTER BAND
            else:
                along_track, cross_track = bands_shifts[n - 1]
                top, bottom, left, right = shifts2TBLT(
                    along_track, cross_track
                )  # Shfits to top, bottom, right, left
                (
                    max_top,
                    max_bottom,
                    max_left,
                    max_right,
                    max_top_band,
                    max_bottom_band,
                    max_left_band,
                    max_right_band,
                ) = maxOffsetUpdate(
                    max_top,
                    max_top_band,
                    max_bottom,
                    max_bottom_band,
                    max_left,
                    max_left_band,
                    max_right,
                    max_right_band,
                    top,
                    bottom,
                    left,
                    right,
                    self.__bands_names[n],
                )
                band_name = self.__bands_names[n]
                band_shifted = shift_band(band, shifts=[along_track, cross_track])

                # -------------------------- BEFORE CASE:
                if granule_filler_before is not None:
                    band_filler_before = granule_filler_before.get_band(band_name)

                    if (self.__bands_names[n] in SWIR_BANDS) and (rotate_swir_bands):
                        band_filler_before = rotate(
                            band_filler_before.unsqueeze(2), 180
                        ).squeeze(2)
                    band_filler_before_across_shifted = shift_band(
                        band_filler_before, shifts=[0, cross_track]
                    )

                    # Band filler is on top:
                    if top != 0:
                        band_shifted[: int(top), :] = band_filler_before_across_shifted[
                            -int(top) :, :
                        ]
                        filler_name["top"] = granule_filler_before.get_granule_info()[0]

                # --------------------- AFTER CASE:
                if granule_filler_after is not None:
                    band_filler_after = granule_filler_after.get_band(band_name)

                    if (self.__bands_names[n] in SWIR_BANDS) and (rotate_swir_bands):
                        band_filler_after = rotate(
                            band_filler_after.unsqueeze(2), 180
                        ).squeeze(2)
                    band_filler_after_across_shifted = shift_band(
                        band_filler_after, shifts=[0, cross_track]
                    )

                    # Band filler is on bottom:
                    if bottom != 0:
                        band_shifted[
                            -int(bottom) :, :
                        ] = band_filler_after_across_shifted[: int(bottom), :]
                        filler_name["bottom"] = granule_filler_after.get_granule_info()[
                            0
                        ]

                coregistered_bands.append(band_shifted)

        # Managing crop exmpty pixel case
        if crop_empty_pixels:
            for band, n in zip(coregistered_bands, range(len(coregistered_bands))):
                band_name = self.__bands_names[n]

                # Resizing max pixel shifts with respect to the resolution of the current band
                max_top_resized = int(
                    max_top
                    * BAND_SPATIAL_RESOLUTION_DICT[max_top_band]
                    / BAND_SPATIAL_RESOLUTION_DICT[band_name]
                )
                max_bottom_resized = int(
                    max_bottom
                    * BAND_SPATIAL_RESOLUTION_DICT[max_bottom_band]
                    / BAND_SPATIAL_RESOLUTION_DICT[band_name]
                )

                max_right_resized = int(
                    max_right
                    * min(BAND_SPATIAL_RESOLUTION_DICT[max_right_band], 20)
                    / min(BAND_SPATIAL_RESOLUTION_DICT[band_name], 20)
                )
                max_left_resized = int(
                    max_left
                    * min(BAND_SPATIAL_RESOLUTION_DICT[max_left_band], 20)
                    / min(BAND_SPATIAL_RESOLUTION_DICT[band_name], 20)
                )

                # The band shall be cropped only if it has not been filled.
                if (filler_name["bottom"] is None) and (max_bottom_resized != 0):
                    band_cropped = band[:-max_bottom_resized]
                else:
                    band_cropped = band

                # The band shall be cropped only if it has not been filled.
                if (filler_name["top"] is None) and (max_top_resized != 0):
                    band_cropped = band_cropped[max_top_resized:]

                if max_right_resized != 0:
                    band_cropped = band_cropped[:, max_left_resized:-max_right_resized]
                else:
                    band_cropped = band_cropped[:, max_left_resized:]

                coregistered_bands[n] = band_cropped

            # Getting information to manage coordinates adjustments.
            polygon_coordinates = np.array(deepcopy(self.get_granule_coordinates()))
            n_stacked_granules = self.get_number_of_granules_along_track_line()
            granule_length = get_granule_px_length(
                n_stacked_granules, satellite, granule_detector_number
            )
            granule_length_across = int(BANDS_RAW_SHAPE_DICT["B02"][1])

            # Adjusting coordinates after cropping on top and bottom
            if max_top != 0 or max_bottom != 0:
                # Reprojecting max cropped pixels with respect to band 2 to have on ground resolution.
                max_top_resized = int(
                    max_top
                    * BAND_SPATIAL_RESOLUTION_DICT[max_top_band]
                    / BAND_SPATIAL_RESOLUTION_DICT["B02"]
                )
                max_bottom_resized = int(
                    max_bottom
                    * BAND_SPATIAL_RESOLUTION_DICT[max_bottom_band]
                    / BAND_SPATIAL_RESOLUTION_DICT["B02"]
                )

                band_shift_01_top = (
                    np.abs(max_top_resized)
                    / granule_length
                    * (polygon_coordinates[1] - polygon_coordinates[0])
                )
                band_shift_32_top = (
                    np.abs(max_top_resized)
                    / granule_length
                    * (polygon_coordinates[2] - polygon_coordinates[3])
                )
                band_shift_01_bottom = (
                    np.abs(max_bottom_resized)
                    / granule_length
                    * (polygon_coordinates[1] - polygon_coordinates[0])
                )
                band_shift_32_bottom = (
                    np.abs(max_bottom_resized)
                    / granule_length
                    * (polygon_coordinates[2] - polygon_coordinates[3])
                )

                # Adjusting coordinates only in absence of top filling.
                if filler_name["top"] is None:
                    # Reprojecting top shift with respect to band 9
                    # (used inside the granule to manage the lenght of the
                    # pixel when bands coordinates are used.)
                    max_top_resized_09 = int(
                        max_top
                        * BAND_SPATIAL_RESOLUTION_DICT[max_top_band]
                        / BAND_SPATIAL_RESOLUTION_DICT["B09"]
                    )
                    point_0_shifted = polygon_coordinates[0] + band_shift_01_top
                    point_3_shifted = polygon_coordinates[3] + band_shift_32_top
                else:
                    # No pixels cropped.
                    max_top_resized_09 = 0
                    point_0_shifted = polygon_coordinates[0]
                    point_3_shifted = polygon_coordinates[3]

                # Adjusting coordinates only in absence of bottom filling.
                if filler_name["bottom"] is None:
                    # Reprojecting bottom shift with respect to band 9
                    # (used inside the granule to manage the lenght of the pixel
                    # when bands coordinates are used.)
                    max_bottom_resized_09 = int(
                        max_bottom
                        * BAND_SPATIAL_RESOLUTION_DICT[max_bottom_band]
                        / BAND_SPATIAL_RESOLUTION_DICT["B09"]
                    )
                    point_1_shifted = polygon_coordinates[1] - band_shift_01_bottom
                    point_2_shifted = polygon_coordinates[2] - band_shift_32_bottom
                else:
                    # No pixels cropped.
                    max_bottom_resized_09 = 0
                    point_1_shifted = polygon_coordinates[1]
                    point_2_shifted = polygon_coordinates[2]

                coregistered_polygon_coordinates = [
                    point_0_shifted,
                    point_1_shifted,
                    point_2_shifted,
                    point_3_shifted,
                ]
                polygon_coordinates = coregistered_polygon_coordinates
                # Used later in left and right adjustment to take into account of cropped coordinates on top and right.
            else:  # No coordinates adjustment on top and bottom.
                max_top_resized_09 = 0
                max_bottom_resized_09 = 0
                polygon_coordinates = np.array(deepcopy(self.get_granule_coordinates()))
                # Same coordinates of original granule. No top or bottom coordinates adjustment.

            # Adjusting coordinates after cropping on right and left
            if max_left != 0 or max_right != 0:
                max_left_resized = int(
                    max_left
                    * BAND_SPATIAL_RESOLUTION_DICT_ACROSS[max_left_band]
                    / BAND_SPATIAL_RESOLUTION_DICT_ACROSS["B02"]
                )
                band_shift_30_left = (
                    np.abs(max_left_resized)
                    / granule_length_across
                    * (polygon_coordinates[3] - polygon_coordinates[0])
                )
                band_shift_21_left = (
                    np.abs(max_left_resized)
                    / granule_length_across
                    * (polygon_coordinates[2] - polygon_coordinates[1])
                )

                max_right_resized = int(
                    max_right
                    * BAND_SPATIAL_RESOLUTION_DICT_ACROSS[max_right_band]
                    / BAND_SPATIAL_RESOLUTION_DICT_ACROSS["B02"]
                )
                band_shift_30_right = (
                    np.abs(max_right_resized)
                    / granule_length_across
                    * (polygon_coordinates[3] - polygon_coordinates[0])
                )
                band_shift_21_right = (
                    np.abs(max_right_resized)
                    / granule_length_across
                    * (polygon_coordinates[2] - polygon_coordinates[1])
                )

                point_0_shifted = polygon_coordinates[0] + band_shift_30_left
                point_3_shifted = polygon_coordinates[3] + band_shift_30_right
                point_1_shifted = polygon_coordinates[1] + band_shift_21_left
                point_2_shifted = polygon_coordinates[2] + band_shift_21_right
                coregistered_polygon_coordinates = [
                    point_0_shifted,
                    point_1_shifted,
                    point_2_shifted,
                    point_3_shifted,
                ]
                # Reprojecting right shift with respect to band 9
                # (used inside the granule to manage the lenght of the pixel
                # across-track when bands coordinates are used.)
                max_left_resized_09 = int(
                    max_left
                    * BAND_SPATIAL_RESOLUTION_DICT_ACROSS[max_left_band]
                    / BAND_SPATIAL_RESOLUTION_DICT_ACROSS["B09"]
                )
                max_right_resized_09 = int(
                    max_right
                    * BAND_SPATIAL_RESOLUTION_DICT_ACROSS[max_right_band]
                    / BAND_SPATIAL_RESOLUTION_DICT_ACROSS["B09"]
                )

            else:
                # Set to polygon coordinates, which takes into account vertically cropped pixels if needed.
                coregistered_polygon_coordinates = polygon_coordinates
                # No pixels cropped.
                max_left_resized_09 = 0
                max_right_resized_09 = 0

        else:  # No cropped pixels
            max_top_resized_09 = 0
            max_bottom_resized_09 = 0
            max_left_resized_09 = 0
            max_right_resized_09 = 0
            coregistered_polygon_coordinates = self.get_granule_coordinates()

        # Adjusting granule name.
        granule_name = self.granule_name

        # Adding COMPLEMENTED_WITH_top_granule_number_bottom_granule_number
        # depending on the case
        if filler_name["top"] is not None or filler_name["bottom"] is not None:
            granule_name = granule_name + "_COMPLEMENTED_WITH"
        if filler_name["top"] is not None:
            granule_name = granule_name + f"_top_{filler_name['top']}"
        if filler_name["bottom"] is not None:
            granule_name = granule_name + f"_bottom_{filler_name['bottom']}"

        # In any case, adding _COREGISTERED mark
        granule_name = granule_name + "_COREGISTERED"

        # Creating a new granule
        coregistered_granule = Raw_granule(
            granule_bands=coregistered_bands,
            bands_names=self.__bands_names,
            granule_name=granule_name,
            granule_polygon_coordinates=coregistered_polygon_coordinates,
            cloud_percentage=self.get_cloud_percentage(),
            device=self.__device,
        )
        # Adding cropped pixels information to manage a correct extraction of bands coordinateds for the new granule.
        coregistered_granule.__set_cropped_pixels(
            cropped_pixels=[
                (max_bottom_resized_09 + max_top_resized_09),
                (max_left_resized_09 + max_right_resized_09),
            ]
        )
        coregistered_granule.__set_originality(False)
        # Adding parents.
        coregistered_granule.__set_parents(self.__granule_parents)
        # Adding information on being a single granule along track
        coregistered_granule.__set_single_granule_along_track_size(
            self.__single_granule_along_track_size
        )

        return coregistered_granule

    def show_bands_superimposition(
        self, requested_bands=None, downsampling=True, equalize=False, n_std=2, ax=None
    ):
        """It shows the superimposition of bands in requested_bands

        Args:
            requested_bands (list, optional): requested bands list.
                                              If None, all the bands are used.
                                              Defaults to None.
            downsampling (boolean, optional): if True, bands are downsampled to have the same resolution.
                                              Default to False.
            equalize (boolean, optional): if True, bands are equalized for a better plotting by saturating
                                          the outliers of each band with an upper and lower valuer respectively
                                          equal to pixel value mean *- n_std * histogram standard deviation.
                                          Default to False.
            n_std (integer, optional):  number of times the value of the pixel values standard deviation used
                                        for histogram cropping. Defaults to 2.
            ax (integer, optional): matplotlib axis to be used to plot if None. Defaults to None.
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

        if equalize:
            bands_superimposed_equalized = equalize_tensor(bands_superimposed, n_std)
            bands_superimposed_equalized = (
                bands_superimposed_equalized
                / (2 ** 12 - 1)
                * bands_superimposed.max()
                / bands_superimposed_equalized.max()
            )
        else:
            bands_superimposed_equalized = bands_superimposed
            bands_superimposed_equalized = (
                bands_superimposed_equalized / bands_superimposed_equalized.max()
            )
        if ax is not None:
            ax.imshow(bands_superimposed_equalized.detach().cpu().numpy(),)
        else:
            plt.imshow(bands_superimposed_equalized.detach().cpu().numpy(),)

    def show_bands(
        self,
        requested_bands=None,
        downsampling=False,
        oversampling=False,
        rotate_swir_bands=False,
    ):
        """It shows the requested bands.

        Args:
            requested_bands (list, optional): list of requested bands to show.
                                              If None, all the bands are shown. Defaults to None.
            downsampling (bool, optional): if True, bands are downsampled to have the same resolution.
                                           Default to False.
            oversampling (bool, optional): if True, bands are oversampled to have the same resolution.
                                           Downsampling has priority over upampling. Default to False.
            rotate_swir_bands (bool, optional): if True, SWIR bands are rotated. Default to False.

        """
        if requested_bands is None:
            requested_bands = self.__bands_names

        n_bands = len(requested_bands)
        n_rows = int(np.ceil(n_bands / 3))

        if n_rows > 1:
            fig, ax = plt.subplots(n_rows, 3)
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(n_rows, 3)

        if downsampling:
            n_printed = 0
            granule_tensor = self.as_tensor(downsampling=True)
            for n in range(len(self.__bands_names)):
                if self.__bands_names[n] in requested_bands:
                    if n_rows > 1:
                        row = int(np.floor(n / 3))
                        column = int(n_printed - 3 * row)
                        ax_plot = ax[row, column]
                    else:
                        ax_plot = ax[n_printed]

                    band_name = self.__bands_names[n]

                    band = granule_tensor[:, :, n]
                    if rotate_swir_bands and (band_name in SWIR_BANDS):
                        band = rotate(band.unsqueeze(2), 180).squeeze(2)

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
            granule_tensor = self.as_tensor(downsampling=False)
            for n in range(len(self.__bands_names)):
                if self.__bands_names[n] in requested_bands:
                    if n_rows > 1:
                        row = int(np.floor(n / 3))
                        column = int(n_printed - 3 * row)
                        ax_plot = ax[row, column]
                    else:
                        ax_plot = ax[n_printed]

                    band_name = self.__bands_names[n]

                    band = granule_tensor[:, :, n]
                    if rotate_swir_bands and (band_name in SWIR_BANDS):
                        band = rotate(band.unsqueeze(2), 180).squeeze(2)

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
                    if rotate_swir_bands and (band_name in SWIR_BANDS):
                        band = rotate(band.unsqueeze(2), 180).squeeze(2)

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

    def get_granule_info(self):
        """Returns name, sensing time, acquisition time, detector number information,
        originality, parents list, polygon coordinates list, cloud percentages.
        If the granule have parents, sensing time, acquisition time, detector number of parents,
        and cloud percentages are returned.

        Returns:
            string: name
            list of datetime: list of sensing time.
            list of datetime: acquisition time.
            list of int: detector numbers.
            boolean: originality
            list: parents_list.
            list: polygon coordinates.
            list: granules percentages.
        """
        granule_parents = self.get_parents()
        if len(granule_parents) == 0:
            granule_name = self.granule_name

            # Removing _COREGISTERED tag
            if granule_name.find("_COREGISTERED") > 0:
                granule_name = granule_name[: granule_name.find("_COREGISTERED")]
            sensing_time = parse_string_date(granule_name[-26:-11])
            acquisition_time = parse_string_date(granule_name[-43:-29])
            detector_numer = int(granule_name[-9:-7])
            return (
                granule_name,
                [sensing_time],
                [acquisition_time],
                [detector_numer],
                self.is_original(),
                granule_parents,
                self.__granule_polygon_coordinates,
                self.__cloud_percentage_list,
            )
        else:
            sensing_time_list = []
            acquisition_time_list = []
            detector_numer_list = []

            for granule_parent in granule_parents:
                granule_name = granule_parent
                sensing_time_list.append(parse_string_date(granule_name[-26:-11]))
                acquisition_time_list.append(parse_string_date(granule_name[-43:-29]))
                detector_numer_list.append(int(granule_name[-9:-7]))
            return (
                self.granule_name,
                sensing_time_list,
                acquisition_time_list,
                detector_numer_list,
                self.is_original(),
                granule_parents,
                self.__granule_polygon_coordinates,
                self.__cloud_percentage_list,
            )

    def show_granule_info(self):
        """Print granule info."""
        granule_info = self.get_granule_info()
        print(colored("------------------Granule ----------------------------", "blue"))
        print("Name: ", colored(granule_info[0], "red"))
        print("Sensing time: ", colored(granule_info[1], "red"))
        print("Creation time: ", colored(granule_info[2], "red"))
        print("Detector number: ", colored(granule_info[3], "red"))
        print("Originality: ", colored(granule_info[4], "red"))
        print("Parents: ", colored(granule_info[5], "red"))
        coordinates = granule_info[6]
        print("Polygon coordinates: \n")
        for m in range(len(coordinates)):
            print(
                colored("\tP_" + str(m), "blue")
                + " : "
                + colored(str(coordinates[m]) + "\n", "red")
            )
        print("Cloud coverage: ", colored(granule_info[7], "red"))
        print("\n")

    def get_number_of_granules_along_track_line(self):
        """Returns the maximum number of granules stacked along track among parents.

        Returns:
            int: Number of granules correctly stacked (same detector number and correct sensing time difference) along track.
        """
        if not (len(self.__granule_parents)):
            return 1
        else:
            parents = self.__granule_parents
            granule_info = self.get_granule_info()
            detector_number_list = granule_info[3]
            sensing_time_list = granule_info[1]

        max_along_track = 1
        tested_parent_list = [0 for n in range(len(parents))]
        for n in range(len(parents)):
            stacked_to_n_list = [n]
            if not (tested_parent_list[n]):
                tested_parent_list[n] = 1
                for m in range(n + 1, len(parents)):
                    if detector_number_list[n] == detector_number_list[m]:
                        if not (
                            (sensing_time_list[m] - sensing_time_list[n]).seconds % 3
                        ) or not (
                            (sensing_time_list[n] - sensing_time_list[m]).seconds % 4
                        ):
                            tested_parent_list[m] = 1
                            stacked_to_n_list.append(m)

            max_along_track = max(max_along_track, len(stacked_to_n_list))
        return max_along_track

    def get_bands_coordinates(self, downsampling=True, latlon_format=True):
        """Returns the coordinates of the bands.
        Args:
            downsampling (bool, optional): if True, bands are downsampled to have the same resolution.
                                           Default to False.
            latlon_format (bool): format to use to get the granule's coordinates.
                                  Default to: Latitude, Longitude.
        Returns:
            dict: band-names/band coordinates dict.
        """

        detector_number = self.get_detectors_number()[0]
        polygon_coordinates = np.array(deepcopy(self.get_granule_coordinates()))

        band_names = self.__bands_names

        band_shifted_list = []
        n_stacked_granules = self.get_number_of_granules_along_track_line()

        satellite = self.get_granule_info()[0][:3]

        granule_length = get_granule_px_length(
            n_stacked_granules, satellite, detector_number, self.__cropped_pixels[0]
        )

        for band_name, n in zip(band_names[:], range(len(band_names[:]))):
            res_scale_factor = (
                BAND_SPATIAL_RESOLUTION_DICT[band_name]
                / BAND_SPATIAL_RESOLUTION_DICT["B02"]
            )

            band = self.get_band(band_name)

            p0_1_angle = (
                res_scale_factor
                * len(band)
                / granule_length
                * (polygon_coordinates[1] - polygon_coordinates[0])
            )
            p3_2_angle = (
                res_scale_factor
                * len(band)
                / granule_length
                * (polygon_coordinates[2] - polygon_coordinates[3])
            )

            if band_name == "B02":
                point_1_shifted = polygon_coordinates[0] + p0_1_angle
                point_2_shifted = polygon_coordinates[3] + p3_2_angle
                shifted_points = [
                    polygon_coordinates[0],
                    point_1_shifted,
                    point_2_shifted,
                    polygon_coordinates[3],
                ]

            elif band_name == "B09":
                point_0_shifted = polygon_coordinates[1] - p0_1_angle
                point_3_shifted = polygon_coordinates[2] - p3_2_angle
                shifted_points = [
                    point_0_shifted,
                    polygon_coordinates[1],
                    polygon_coordinates[2],
                    point_3_shifted,
                ]

            else:
                band_shift = abs(
                    get_bands_shift(
                        ["B02", band_name],
                        satellite,
                        detector_number,
                        downsampling=downsampling,
                    )[0][0]
                )
                band_shift_01 = (
                    np.abs(band_shift * res_scale_factor)
                    / granule_length
                    * (polygon_coordinates[1] - polygon_coordinates[0])
                )
                band_shift_32 = (
                    np.abs(band_shift * res_scale_factor)
                    / granule_length
                    * (polygon_coordinates[2] - polygon_coordinates[3])
                )
                if not (detector_number % 2):
                    point_0_shifted = polygon_coordinates[0] + band_shift_01
                    point_1_shifted = (
                        polygon_coordinates[0] + band_shift_01 + p0_1_angle
                    )
                    point_2_shifted = (
                        polygon_coordinates[3] + band_shift_32 + p3_2_angle
                    )
                    point_3_shifted = polygon_coordinates[3] + band_shift_32
                else:
                    point_0_shifted = polygon_coordinates[1] - band_shift_01
                    point_1_shifted = (
                        polygon_coordinates[1] - band_shift_01 - p0_1_angle
                    )
                    point_2_shifted = (
                        polygon_coordinates[2] - band_shift_32 - p3_2_angle
                    )
                    point_3_shifted = polygon_coordinates[2] - band_shift_32
                shifted_points = [
                    point_0_shifted,
                    point_1_shifted,
                    point_2_shifted,
                    point_3_shifted,
                ]

            shifted_points_list = [x.tolist() for x in shifted_points]
            polygon = Polygon([list(x) for x in shifted_points_list]).convex_hull
            if latlon_format:
                band_shifted_list.append(
                    [
                        (x, y)
                        for (x, y) in zip(
                            polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1]
                        )
                    ]
                )
            else:
                band_shifted_list.append(
                    [
                        (y, x)
                        for (x, y) in zip(
                            polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1]
                        )
                    ]
                )

        return dict(zip(band_names, band_shifted_list))

    def export_to_tif(self, save_path):
        """Export to TIF file.

        Args:
            save_path (str): save path.
            downsampling (bool, optional): If True, bands are downsampled. Defaults to True.
        """
        coords = self.get_bands_coordinates(downsampling=True, latlon_format=False)

        for band_name in list(self.__bands_dict.keys()):
            band = self.__bands_dict[band_name]
            export_band_to_tif(
                band, coords[band_name], os.path.join(save_path, band_name + ".tif")
            )

    def get_raw_bbox(self, l1c_tif, bbox, mode="standard"):
        """Changing bbox coordinates from green system (L1C) to raw.

        Args:
            l1c_tif: tensor of the l1c image
            bbox: (list): [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
        Ret:
            list: corrected bbox list
        """

        def get_points_from_bbox(bbox):
            p1 = bbox[:2]
            p2 = bbox[2:]
            bottleft = (p2[0], p1[1])
            topright = (p1[0], p2[1])
            return (p1, topright, p2, bottleft)

        def get_srcPoints(l1c_tif):
            for k in range(l1c_tif.shape[1]):
                y = l1c_tif[:, k, 0]
                dy = torch.where(y != 0)[0]
                if len(dy) != 0:
                    dy = dy[0]
                    break

            for m in range(l1c_tif.shape[0]):
                x = l1c_tif[m, :, 0]
                dx = torch.where(x != 0)[0]
                if len(dx) != 0:
                    dx = dx[0]
                    break

            for k in range(l1c_tif.shape[1] - 1, 0, -1):
                y_end = l1c_tif[:, k, 0]
                dy_end = torch.where(y_end != 0)[0]
                if len(dy_end) != 0:
                    dy_end = dy_end[0]
                    break

            for q in range(l1c_tif.shape[0] - 1, 0, -1):
                x_end = l1c_tif[q, :, 0]
                dx_end = torch.where(x_end != 0)[0]
                if len(dx_end) != 0:
                    dx_end = dx_end[0]
                    break
            if mode == "standard":
                return [np.array([dy, k]), np.array([m, dx]), np.array([q, dx_end])]
            else:
                return [np.array([m, dx]), np.array([dy_end, k]), np.array([dy, k])]

        def prep_point(point):
            arr = np.array([point[0], point[1], 1])
            return arr

        raw_shape = self.as_tensor(downsampling=True).shape
        r_S1, c_S1, _ = raw_shape
        dstTri = np.array([(r_S1, 0), (0, 0), (r_S1, c_S1)]).astype(np.float32)
        srcTri = np.array(get_srcPoints(l1c_tif)).astype(np.float32)

        bbox_points = get_points_from_bbox(bbox)

        M = cv2.getAffineTransform(srcTri, dstTri)
        transformed_points = []
        for x in bbox_points:
            transformed_points.append(M @ prep_point(np.array(x)))

        return transformed_points
