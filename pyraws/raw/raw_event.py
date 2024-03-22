import torch
from termcolor import colored
import geopy.distance


from .raw_granule import Raw_granule
from ..utils.constants import BANDS_RAW_SHAPE_DICT
from ..utils.raw_utils import (
    read_Raw_event_from_database,
    find_granules_names,
    read_Raw_event_from_path,
)


class Raw_event:
    __device = None
    __bands_names = []
    __granules_collection = []
    __event_class = None
    __raw_useful_granules_idx = []
    __raw_complementary_granules_idx = []
    __useful_granule_bounding_box_dict = {}

    def __init__(
        self,
        granules_collection=None,
        bands_names=None,
        event_class=None,
        raw_useful_granules_idx=None,
        raw_complementary_granules_idx=None,
        useful_granule_bounding_box_dict=None,
        device=torch.device("cpu"),
    ):
        """Creates an raw event from a granules collection and band_names.
        It is possible to associate an image class to the event.

        Args:
            granules_collection (list, optional): list of Raw_granule. Defaults to None.
            bands_names (list, optional): list of band names. Defaults to None.
            event_class (string, optional): class name. Defaults to None.
            raw_useful_granules_idx (list, optional): list of useful granules indices. Defaults to None.
            raw_complementary_granules_idx (list, optional): list of complementary granules indices. Defaults to None.
            useful_granule_bounding_box_dict (dict, optional): {useful granule : bounding boxes}. Defaults to None.
            device (torch.device, optional): device for each raw granule in the image. Defaults to torch.device("cpu").
        """
        if bands_names is None:
            self.__bands_names = []
        else:
            self.__bands_names = bands_names

        self.__event_class = event_class
        if granules_collection is None:
            self.__granules_collection = []
        else:
            self.__granules_collection = granules_collection
        self.n_granules = 0
        if raw_useful_granules_idx is None:
            self.__raw_useful_granules_idx = []
        else:
            self.__raw_useful_granules_idx = raw_useful_granules_idx

        if raw_complementary_granules_idx is None:
            self.__raw_complementary_granules_idx = []
        else:
            self.__raw_complementary_granules_idx = raw_complementary_granules_idx

        if granules_collection is not None:
            for n in range(len(granules_collection)):
                self.__granules_collection.append(
                    torch.device(granules_collection[n], device=device)
                )
            self.n_granules = len(self.__granules_collection)

        if useful_granule_bounding_box_dict is not None:
            self.__useful_granule_bounding_box_dict = useful_granule_bounding_box_dict
        else:
            self.__useful_granule_bounding_box_dict = {}

        self.__device = device

    def from_path(self, raw_dir_path, bands_list, verbose=True):
        """Read specific bands of the Sentinel-2 raw event located at ""raw_dir_path"".

        Args:
            raw_dir_path (str): path to the raw event dir.
            bands list. If None, all bands are used and sorted according to the datasheet order. Defaults to None.
            verbose (bool, optional): if True, if True, verbose mode is used. Defaults to True.
        """

        if not (self.is_void()):
            print(
                "Impossible to create a new event from path: "
                + colored(raw_dir_path, "blue")
                + ". "
                + colored("Event already instantiated.", "red")
            )
        else:
            if bands_list is None:
                bands_list = list(BANDS_RAW_SHAPE_DICT.keys())
            try:
                (
                    granules_collection,
                    granules_paths,
                    polygon_coordinates_list,
                    cloud_percentages_list,
                ) = read_Raw_event_from_path(
                    raw_dir_path, bands_list, verbose, self.__device
                )
            except:  # noqa: E722
                raise ValueError(
                    "Impossible to open the raw file at: "
                    + colored(raw_dir_path, "red")
                    + "."
                )

            granule_names = find_granules_names(granules_paths)
            self.__bands_names = bands_list
            # These pieces of information are not available since the event is not read from database.
            self.__raw_useful_granules_idx = []
            self.__raw_complementary_granules_idx = []
            self.__event_class = None
            self.__useful_granule_bounding_box_dict = {}

            for granule, n in zip(granules_collection, range(len(granules_collection))):
                new_granule = Raw_granule(device=self.__device)
                along_track_size = geopy.distance.geodesic(
                    polygon_coordinates_list[n][0], polygon_coordinates_list[n][1]
                ).km
                new_granule.create_granule(
                    bands_list,
                    granule,
                    granule_names[n],
                    polygon_coordinates_list[n],
                    along_track_size,
                    cloud_percentages_list[n],
                )
                self.__granules_collection.append(new_granule)

            self.n_granules = len(self.__granules_collection)

    def from_database(
        self,
        id_event,
        bands_list=None,
        cfg_file_dict=None,
        id_raw_l1_dict=None,
        verbose=True,
        database="THRAWS",
    ):
        """Read specific bands of the Sentinel-2 raw event ""id_event"", specified in "bands_list", from database.

        Args:
            id_event (str): event ID.
            bands_list (list, optional): bands list. If None, all bands are used and sorted according to the datasheet order.
                                         Defaults to None.
            cfg_file_dict (dict, optional): dictionary containing paths to the different end2end directories.
                                          If None, internal CSV database will be parsed.
            id_raw_l1_dict (dict, optional): id-raw-l1 dictionary. If None, internal CSV database will be parsed.
            verbose (bool, optional): if True, if True, verbose mode is used. Defaults to True.
            database (string, optional): database name. Defaults to "THRAWS".
        """
        if not (self.is_void()):
            print(
                "Impossible to create a new event from: "
                + colored(id_event, "blue")
                + ". "
                + colored("Event already instantiated.", "red")
            )
        else:
            if bands_list is None:
                bands_list = list(BANDS_RAW_SHAPE_DICT.keys())

            (
                granules_collection,
                event_class,
                granule_names,
                raw_useful_granules,
                raw_complementary_granules,
                polygon_coordinates_list,
                cloud_percentages_list,
                useful_granule_bounding_box_dict,
            ) = read_Raw_event_from_database(
                id_event,
                bands_list,
                cfg_file_dict,
                id_raw_l1_dict,
                database,
                verbose,
                device=self.__device,
            )
            self.__bands_names = bands_list
            self.__raw_useful_granules_idx = raw_useful_granules
            self.__raw_complementary_granules_idx = raw_complementary_granules
            self.__event_class = event_class
            self.__useful_granule_bounding_box_dict = useful_granule_bounding_box_dict

            for granule, n in zip(granules_collection, range(len(granules_collection))):
                new_granule = Raw_granule(device=self.__device)
                along_track_size = geopy.distance.geodesic(
                    polygon_coordinates_list[n][0], polygon_coordinates_list[n][1]
                ).km
                new_granule.create_granule(
                    bands_list,
                    granule,
                    granule_names[n],
                    polygon_coordinates_list[n],
                    along_track_size,
                    cloud_percentages_list[n],
                )
                self.__granules_collection.append(new_granule)

            self.n_granules = len(self.__granules_collection)

    def get_bands_list(self):
        """Returns the list of bands of every Raw_granule object in the collection.

        Returns:
            list: band names.
        """
        return self.__bands_names

    def get_bounding_box_dict(self):
        """Bounding box dictionaries getter.

        Returns:
            dict: Returns {useful granules : bounding box dictionary}
        """
        return self.__useful_granule_bounding_box_dict

    def get_event_class(self):
        """Event class getter.

        Returns:
            dict: Returns {useful granules : bounding box dictionary}
        """
        return self.__event_class

    def get_granule(self, granule_idx):
        """Returns the granule addressed by granule_idx.

        Args:
            granule_idx (int): granule index.
        Returns:
            raw_granule: raw granule matching the corresponding index.
        """
        return self.__granules_collection[granule_idx]

    def get_device(self):
        """Returns the used device.

        Returns:
            torch.device: used torch device.
        """
        return self.__device

    def get_stackable_granules(self):
        """Returns list of stackable granules couples indices and stacking positions.

        Returns:
            list: list of stackable granules couples.
            list: stacking poisition for each couple of stackable granule.
        """
        granules_info = self.get_granules_info()
        granules_names = list(granules_info.keys())
        n_granules = len(granules_info)

        granules_couples_list = []
        granules_postions_list = []

        for granule_name, n in zip(granules_names, range(n_granules)):
            if granules_info[granule_name][4]:
                sensing_time = granules_info[granule_name][1][0]
                detector_number = granules_info[granule_name][3]

                for granule_name_m, m in zip(
                    granules_names[n + 1 :], range(n + 1, n_granules)
                ):
                    sensing_time_m = granules_info[granule_name_m][1][0]
                    detector_number_m = granules_info[granule_name_m][3]

                    if (granules_info[granule_name_m][4]) and (
                        detector_number == detector_number_m
                    ):
                        if ((sensing_time_m - sensing_time).seconds == 3) or (
                            (sensing_time_m - sensing_time).seconds == 4
                        ):
                            granules_couples_list.append([n, m])
                            granules_postions_list.append(["T"])
                            break
                        elif ((sensing_time - sensing_time_m).seconds == 3) or (
                            (sensing_time - sensing_time_m).seconds == 4
                        ):
                            granules_couples_list.append([n, m])
                            granules_postions_list.append(["B"])
                            break
        return granules_couples_list, granules_postions_list

    def show_granules_info(self):
        """Print granules info."""
        granules_info = self.get_granules_info()
        granules_names = list(granules_info.keys())
        for n in range(len(granules_names)):
            print(
                colored(
                    "------------------Granule "
                    + str(n)
                    + " ----------------------------",
                    "blue",
                )
            )
            print("Name: ", colored(granules_info[granules_names[n]], "red"))
            print("Sensing time: ", colored(granules_info[granules_names[n]][1], "red"))
            print(
                "Creation time: ", colored(granules_info[granules_names[n]][2], "red")
            )
            print(
                "Detector number: ", colored(granules_info[granules_names[n]][3], "red")
            )
            print("Originality: ", colored(granules_info[granules_names[n]][4], "red"))
            print("Parents: ", colored(granules_info[granules_names[n]][5], "red"))
            coordinates = granules_info[granules_names[n]][6]
            print("Polygon coordinates: \n")
            for m in range(len(coordinates)):
                print(
                    colored("\tP_" + str(m), "blue")
                    + " : "
                    + colored(str(coordinates[m]) + "\n", "red")
                )

            print(
                "Cloud coverage: ",
                colored(granules_info[granules_names[n]][7][0], "red"),
            )
            print("\n")

    def stack_granules(self, granules_idx, positions):
        """Stack different granules recursively specified by granules_idx.
        Positions will specify the stacking positions.
        If granules_idx=[0,1,2] and positions=["T", "B"], granule_0 will be stacked at the top of granule_1
        and the result will be stacked at the bottom of granule_2.

        Args:
            granules_idx (list): list of granule indices.
            positions (list): list of stacking positions.

        Returns:
            Raw_granule: recursively stacked granule.
        """
        granule_stacked = self.get_granule(granules_idx[0])

        for granule_idx, n in zip(granules_idx[1:], range(len(granules_idx[1:]))):
            granule_stacked = granule_stacked.stack_to_granule(
                self.get_granule(granule_idx), positions[n]
            )

        return granule_stacked

    def stack_granules_couples(self):
        """Stacks automatically couples of granules by detector number and sensing time.

        Returns:
            list: stacked couples of granules
            list: indidces of couples of granules
        """
        stackable_couples, positions = self.get_stackable_granules()

        if len(stackable_couples) == 0:
            print("No granules to stack.")
            return []

        stacked_granule_list = []
        for stackable_couple, position in zip(stackable_couples, positions):
            stacked_granule_list.append(self.stack_granules(stackable_couple, position))

        stackable_couple_sorted = []  # Sorted to be the first on top

        for couple, position in zip(stackable_couples, positions):
            if position[0] == "T":
                stackable_couple_sorted.append(couple)
            else:
                stackable_couple_sorted.append([couple[1], couple[0]])

        return stacked_granule_list, stackable_couple_sorted

    def get_granules_names(self, granules_idx=None):
        """Return names of the granules requested through granules_idx from granules names.

        Args:
            granules_idx (list, optional): list of granules for which getting the names.
                                         If None, all the names of the granules in the collection are returned.
                                         Defaults to None.

        Raises:
            ValueError: Empty granules lists

        Returns:
            list: granules' names.
        """
        granules_names = []
        if len(self.__granules_collection) == 0:
            raise ValueError("Empty granules lists.")

        if granules_idx is None:
            granules_idx = range(len(self.__granules_collection))

        for granule_idx in granules_idx:
            granules_names.append(self.get_granule(granule_idx).granule_name)

        return granules_names

    def get_granules_info(self, granules_idx=None):
        """Return info of the granules requested through granules_idx from granules names.

        Args:
            granules_idx (list, optional): list of granules for which getting the names.
                                         If None, all the names of the granules in the collection are returned.
                                         Defaults to None.

        Raises:
            ValueError: Empty granules lists

        Returns:
            dictionary: granules name : granules info
        """
        granules_names = []
        granules_info = []
        if len(self.__granules_collection) == 0:
            raise ValueError("Empty granules lists.")

        if granules_idx is None:
            granules_idx = range(len(self.__granules_collection))

        for granule_idx in granules_idx:
            granule = self.get_granule(granule_idx)
            granule_info = granule.get_granule_info()
            granules_info.append(granule_info)
            granules_names.append(granule_info[0])

        return dict(zip(granules_names, granules_info))

    def set_useful_granules(self, useful_granules_list):
        """useful_granules_list (list): Set useful granules from list."""
        self.__raw_useful_granules_idx = useful_granules_list

    def get_useful_granules_idx(self):
        """Returns useful granules indices.

        Returns:
            list: useful granules indices.
        """
        return self.__raw_useful_granules_idx

    def get_complementary_granules_idx(self):
        """Returns complementary granules indices.

        Returns:
            list: complementary granules indices.
        """
        return self.__raw_complementary_granules_idx

    def set_complementary_granules(self, complementary_granules):
        """useful_granules_list (list): Set useful granules from list."""
        self.__raw_complementary_granules_idx = complementary_granules

    def is_void(self):
        """Returns true if the image is void.

        Returns:
            bool: True if the image is void.
        """

        if len(self.__granules_collection) or (self.__granules_collection is None):
            return False
        else:
            return True

    def coarse_coregistration(
        self,
        granules_idx=None,
        use_complementary_granules=False,
        crop_empty_pixels=False,
        downsampling=True,
        bands_shifts=None,
        verbose=False,
    ):
        """It implements the coarse coregistration of the bands by compensating the along-track pixels shift
        with respect to the first band.

        Args:
            granules_idx (list, optional): Indices of granules to stack and coregister.
                                           If None, internal __raw_useful_granules_idx are used.
                                           Defaults to None.
            use_complementary_granules (boolean, optional): if True, coregistration is performed with filler elements.
                                                          Defaults to False.
            crop_empty_pixels (boolean, optional):  if True and use_complementary_granules is False or no filler is available,
                                                  empty pixels are cropped. Defaults to False.
            bands_shifts (list, optional): bands shift values compared to the first band.
                                           If None, they will be read by the LUT file. Defaults to None.
            downsampling (boolean, optional): if True, higher resolution bands will be undersampled
                                              to match the bands with the lowest resolution.
                                              If False, lower resolution bands will be upsampled to match the bands
                                            with the highest resolution. Defaults to True.
            verbose (boolean, optional): if True, verbose mode is used. Defaults to False.
        Returns:
            Raw_granule: granule with coarse-coregistered bands.
        """

        if granules_idx is None:
            print("No granule indexes was specified. Using default.")
            granules_idx = self.__raw_useful_granules_idx

        if len(granules_idx) > 1:
            granule_stacked = self.stack_granules(
                granules_idx, ["T" for n in range(len(granules_idx))]
            )
        else:
            granule_stacked = self.get_granule(granules_idx[0])

        granule_filler_before = None
        granule_filler_after = None

        if use_complementary_granules:
            if len(granules_idx) == 1:
                stackable_couples, positions = self.get_stackable_granules()
                stackable_couple_sorted = []

                # Every couple will be ordered as [acquired before, acquired after] in that couple.
                for couple, position in zip(stackable_couples, positions):
                    if position[0] == "T":
                        stackable_couple_sorted.append(couple)
                    else:
                        stackable_couple_sorted.append([couple[1], couple[0]])

                if verbose:
                    print("This granule: " + colored(str(granules_idx[0]), "red"))
                    print(
                        "Stackable granules: "
                        + colored(str(stackable_couple_sorted), "red")
                    )

                for couple in stackable_couple_sorted:
                    if granules_idx[0] == couple[0]:
                        granule_filler_after = self.get_granule(couple[1])
                        if verbose:
                            print("Granule after: " + colored(str(couple[1]), "red"))

                    if granules_idx[0] == couple[1]:
                        granule_filler_before = self.get_granule(couple[0])
                        if verbose:
                            print("Granule before: " + colored(str(couple[0]), "red"))

            else:
                raise ValueError(
                    "Use of complementary granules is not supported for more than one granules at the time."
                )

        return granule_stacked.coarse_coregistration(
            rotate_swir_bands=True,
            crop_empty_pixels=crop_empty_pixels,
            granule_filler_before=granule_filler_before,
            granule_filler_after=granule_filler_after,
            downsampling=downsampling,
            bands_shifts=bands_shifts,
            verbose=verbose,
        )
