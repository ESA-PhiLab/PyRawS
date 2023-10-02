try:
    from pyraws.sys_cfg import PYRAWS_HOME_PATH, DATA_PATH
except:  # noqa: E722
    raise ValueError(
        "sys_cfg.py not found. Please, refer to README.md for instructions on how to generate it."
    )
import os
import pandas as pd
from termcolor import colored
import csv
from .constants import DATABASE_FILE_DICTIONARY, BAND_NAMES_REAL_ORDER
from pathlib import Path
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from ast import literal_eval


class DatabaseHandler:
    """Creates an Raw-L1C csv database for the APIs."""

    def __init__(self, db_name, datapath=DATA_PATH):
        """
        Initializes the Raw-L1C database.
        Args:
            db_name (str): The name of the dataset as named in the data folder.
            datapath (str, optional): Paht to the folder where the database is located.
        """
        self.datapath = Path(datapath) / db_name
        self.fetcher()

        columns = [
            "ID_event",
            "Start",
            "End",
            "Sat",
            "class",
            "Polygon",
            "Raw_useful_granules",
            "Raw_complementary_granules",
            "raw_files",
            "l1c_files",
            "bbox_list",
        ]
        self.db = pd.DataFrame({x: [] for x in columns})

    def fetcher(self):
        """Fetches the products in the the database."""
        self.Raw = self.datapath / "raw"
        try:
            self.L1 = self.datapath / "l1c"
            self.L1_products = [x for x in self.L1.iterdir() if x.is_dir()]
            print("Found L1c products: ", len(self.L1_products))
        except:  # noqa: E722
            print(
                "L1 folder not found. If this behaviour is not desidered, please, refer to README.md for instructions."
            )

        print("Fetching the database...")
        self.Raw_products = [x for x in self.Raw.iterdir() if x.is_dir()]
        assert len(self.Raw_products) > 0, "No Raw products found. Aborting..."
        print("Completed.")

    def single_parser(self, raw_Event_folderpath):
        """
        Parses a single Raw product.
        Args:
            raw_Event_folderpath (str): datapath of the Raw event.
        """
        id_event = raw_Event_folderpath.name

        def gain_xml(prod_path: Path):
            """
            Helper function to get an xml file.
            Args:
                prod_path (str): datapath of the Raw event.
            Returns:
                Invetory Metadata xml path of one of the Raw granules.
            """
            xmls = [
                x
                for x in prod_path.glob("**/*")
                if x.name.endswith("Inventory_Metadata.xml")
            ]
            return xmls[0]

        def meta_extract(xml_path):
            """
            Helper function to get an xml file.
            Args:
                xml_path (str): datapath of the Raw event.
            Returns:
                meta (dict): metadata dictionary containing useful information for the event.
            """
            tree = ET.parse(xml_path)
            root = tree.getroot()
            keys = [x for x in root.attrib.keys()]
            # Find the xmlns attribute
            xsi_xmlns = "{" + root.attrib[keys[0]] + "}"
            info = {}
            for child in root.iter():
                for query in ["Satellite_Code"]:
                    if child.tag.removeprefix(xsi_xmlns) == query:
                        info[query] = child.text
            try:
                geo_localization = root.find(f".//{xsi_xmlns}Geographic_Localization")
                points = geo_localization.findall(f".//{xsi_xmlns}Geo_Pnt")

                # extract latitude and longitude from each point
                coordinates = []
                for point in points:
                    latitude = point.find(f"{xsi_xmlns}LATITUDE").text
                    longitude = point.find(f"{xsi_xmlns}LONGITUDE").text
                    coordinates.append([float(longitude), float(latitude)])

                polygon = Polygon(coordinates)
                info["polygon"] = polygon.wkt
            except AttributeError:
                info["polygon"] = None
            return info

        xml_path = gain_xml(raw_Event_folderpath)
        meta = meta_extract(xml_path)
        raw_files = [x.name for x in raw_Event_folderpath.iterdir()]

        l1c_files = self.L1_products
        l1c_event_folder = [x for x in l1c_files if x.name == id_event]
        if len(l1c_event_folder) > 0:
            l1c_subfiles = [x for x in l1c_event_folder[0].iterdir() if x.is_dir()]
        else:
            l1c_subfiles = []

        return {
            "event": f"{raw_Event_folderpath.name}",
            "l1c_files": l1c_subfiles,
            "raw_files": raw_files,
            "Satellite": meta["Satellite_Code"],
            "poly": meta["polygon"],
        }

    def parser(self):
        """Parses all the Raw products and creates the database."""
        for idx, raw_path in enumerate(self.Raw_products):
            meta = self.single_parser(raw_path)
            data = {
                "ID_event": meta["event"],
                "Start": None,
                "End": None,
                "Sat": meta["Satellite"],
                "class": None,
                "Polygon": meta["poly"],
                "Raw_useful_granules": None,
                "Raw_complementary_granules": None,
                "Raw_files": meta["raw_files"],
                "l1c_files": meta["l1c_files"],
                "bbox_list": None,
            }
            new_row = pd.DataFrame.from_dict(data, orient="index")
            new_row = new_row.transpose()
            self.db = pd.concat([self.db, new_row])
        self.db = self.db.reset_index(drop=True)


def get_cfg_file_dict():
    """Returns a dictionary containing paths to the different pyraws directories.
    Returns:
        dict: cfg file dict.
    """
    PYRAWS_PACKAGE_PATH = os.path.join(PYRAWS_HOME_PATH, "pyraws")
    SCRIPTS_PATH = os.path.join(PYRAWS_PACKAGE_PATH, "scripts")
    NOTEBOOKS_PATH = os.path.join(PYRAWS_PACKAGE_PATH, "notebooks")

    SCRIPTS_PATH = os.path.join(PYRAWS_PACKAGE_PATH, "scripts")
    DATABASE_PATH = os.path.join(PYRAWS_PACKAGE_PATH, "database")

    GEOIDS_PATH = os.path.join(PYRAWS_PACKAGE_PATH, "geoids")

    paths_list = [
        SCRIPTS_PATH,
        PYRAWS_PACKAGE_PATH,
        PYRAWS_HOME_PATH,
        DATA_PATH,
        NOTEBOOKS_PATH,
        SCRIPTS_PATH,
        DATABASE_PATH,
        GEOIDS_PATH,
    ]
    cfg_dir_list = [
        "scripts",
        "PYRAWS_pkg",
        "pyraws",
        "data",
        "notebooks",
        "scripts",
        "database",
        "geoids",
    ]
    return dict(zip(cfg_dir_list, paths_list))


def get_raw_shift_lut(
    satellite, detector_number, downsampling=True, cfg_file_dict=None
):
    """Get Raw shift LUT.

    Args:
        satellite (str, optional): "S2A" or "S2B" respectively for "Sentinel-2A" data and "Sentinel-2B" data.
        detector_number (int): Detectorn number.
        downsampling (boolean, optional): if True, shift values for downsampled bands of the chosen satellite are used.
                                        Otherwise, values for upsampled bands are used. Defaults to True.
        cfg_file_dict (dict, optional): cfg_file_dict (dict, optional): dictionary containing paths to
                                        the different pyraws directories. If None, internal CSV database will be parsed.
                                        Defaults to None.
    Returns:
        dict: returns the Raw shift LUT.
    """
    if cfg_file_dict is None:
        cfg_file_dict = get_cfg_file_dict()

    csv_path = os.path.join(cfg_file_dict["database"], "shift_lut.csv")

    lut_df = pd.read_csv(csv_path)
    lut_df = lut_df[lut_df["satellite"] == satellite]
    if downsampling:
        lut_df = lut_df[lut_df["registration_mode"] == "downsampling"]
    else:
        lut_df = lut_df[lut_df["registration_mode"] == "upsampling"]

    lut_df = lut_df[lut_df["detector"] == detector_number]

    BAND_SHIFTS_NAMES = [
        "S08_02",
        "S03_08",
        "S10_03",
        "S04_10",
        "S05_04",
        "S11_05",
        "S06_11",
        "S07_06",
        "S8A_07",
        "S12_8A",
        "S01_12",
        "S09_01",
    ]

    return dict(
        zip(
            BAND_NAMES_REAL_ORDER[1:] + [BAND_NAMES_REAL_ORDER[0]],
            [
                [
                    -float(
                        lut_df[band_name]
                        .iloc[0]
                        .replace("[", "")
                        .replace("]", "")
                        .split(",")[0]
                    ),
                    -float(
                        lut_df[band_name]
                        .iloc[0]
                        .replace("[", "")
                        .replace("]", "")
                        .split(",")[1]
                    ),
                ]
                for band_name in BAND_SHIFTS_NAMES
            ]
            + [0],
        )
    )


def get_id_raw_l1_dict(database="THRAWS", cfg_file_dict=None):
    """Returns a dictionary containing raw (directory name), l1c (product ID and correspondent granule) and class,
    raw useful and complementary tiles.
    Args:
        database (string, optional): database name. Defaults to ""THRAWS"".
        cfg_file_dict (dict, optional): dictionary containing paths to the different pyraws directories.
                                       If None, internal CSV database will be parsed.
    Returns:
        dict: label_raw-l1_dict
    """
    if cfg_file_dict is None:
        cfg_file_dict = get_cfg_file_dict()
    image_csv_path = os.path.join(
        cfg_file_dict["database"], DATABASE_FILE_DICTIONARY[database]
    )
    try:
        l1c_name_list = []
        class_list = []
        id_list = []
        raw_name_list = []
        raw_useful_granules_list = []
        raw_complementary_granules_list = []
        event_coordinates_list = []
        requested_polygon_list = []
        bbox_event_list = []
        with open(image_csv_path, mode="r", encoding="utf-8-sig") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                id_list.append(row["ID_event"])
                l1c_name_list.append(row["ID_event"])
                class_list.append(row["class"])
                raw_name_list.append(row["ID_event"])
                raw_useful_granules_list.append(row["Raw_useful_granules"])
                raw_complementary_granules_list.append(
                    row["Raw_complementary_granules"]
                )
                event_coordinates_list.append(row["Coords (Lat, Lon)"])
                polygon = row["Polygon"]
                bbox_event_list.append(row["bbox_list"])
                try:
                    polygon = polygon[10:-2].split(",")
                    requested_polygon_list.append(
                        [
                            [
                                float(polygon[0].split(" ")[0]),
                                float(polygon[0].split(" ")[1]),
                            ]
                        ]
                        + [
                            [float(x[1:].split(" ")[0]), float(x[1:].split(" ")[1])]
                            for x in polygon[1:]
                        ]
                    )
                except:  # noqa: E722
                    requested_polygon_list.append([[None, None]])
    except:  # noqa: E722
        print(
            colored("ERROR", "red")
            + ". Impossible to parse the file: "
            + colored(image_csv_path, "blue")
            + "."
        )
        return
    #
    id_raw_l1_dict_list = []

    for n in range(len(id_list)):
        id_raw_l1_dict_list.append(
            {
                "raw": raw_name_list[n],
                "l1c": l1c_name_list[n],
                "class": class_list[n],
                "raw_useful_granules": raw_useful_granules_list[n],
                "raw_complementary_granules": raw_complementary_granules_list[n],
                "events_coords": event_coordinates_list[n],
                "requested_polygon": requested_polygon_list[n],
                "bbox_list": bbox_event_list[n],
            }
        )

    return dict(zip(id_list, id_raw_l1_dict_list))


def get_event_granule_bb_dict(event_id, database="THRAWS", cfg_file_dict=None):
    """Function to extract the dictionary {useful_granule : bounding_box_list} from the bbox_str got from the database.

    Args:
        event_id (string): event ID.
        database (string, optional): database name. Defaults to ""THRAWS"".
        cfg_file_dict (dict, optional): dictionary containing paths to the different pyraws directories.
                                    If None, internal CSV database will be parsed.

    Returns:
        dict: {useful_granule : bounding_box_list} for the requested event.
    """
    event_dict = get_id_raw_l1_dict(database=database, cfg_file_dict=cfg_file_dict)[
        event_id
    ]
    raw_useful_granules_bb_dict = literal_eval(event_dict["bbox_list"])
    return raw_useful_granules_bb_dict


def get_events_list(database="THRAWS", cfg_file_dict=None):
    """Returns the list of events in the database.

    Args:
        database (string, optional): database name. Defaults to ""THRAWS"".
        cfg_file_dict (dict, optional): dictionary containing paths to the different pyraws directories.
                                      If None, internal CSV database will be parsed.
    Returns:
        str: list of events.
    """
    my_dict = get_id_raw_l1_dict(database=database, cfg_file_dict=cfg_file_dict)
    return list(my_dict.keys())


def get_event_info(
    event_id, cfg_file_dict=None, id_raw_l1_dict=None, database="THRAWS"
):
    """From the event_ID, it returns raw directory path, the path to the l1 image, to the L1C auxiliary file,
    and the class of the image. If no `id_raw_l1_dict`is provided, it is done by parsing the internal CSV database file.

    Args:
        event_id (string): image name.
        cfg_file_dict (dict, optional): dictionary containing paths to the different pyraws directories.
                                      If None, internal CSV database will be parsed.
        id_raw_l1_dict (dict, optional): id-raw-l1 dictionary. If None, internal CSV database will be parsed.
        database (string, optional): database name. Defaults to "THRAWS".

    Raises:
        ValueError: If the image is not in the database.

    Returns:
        string: path to the Raw image.
        string: path to the L1 image.
        string: path to the post-processed L1C tiff file.
        string: expected class.
        list: list of raw useful tiles.
        list: list of complementary raw tiles (to coregister without 0).
        dict: event coordinates dict {"lat" : lat, "lon" : lon}.
        list: list of the requested polygon coordinates.
    """

    def parse_string(str, skip_marks=True, return_int=False):
        str_tiles = []

        str = str.replace(" ", "")
        if not (len(str)):
            return [""]
        last_char_pos = 0
        while str[last_char_pos:].find(",") != -1:
            comma_pos = str[last_char_pos:].find(",") + last_char_pos
            if skip_marks:
                str_tile = str[last_char_pos + 1 : comma_pos - 1]
            else:
                str_tile = str[last_char_pos:comma_pos]

            if return_int:
                if str_tile == "None":
                    str_tiles.append(None)
                else:
                    str_tiles.append(int(str_tile))
            else:
                str_tiles.append(str_tile)

            last_char_pos = comma_pos + 1
        if skip_marks:
            str_tile = str[last_char_pos + 1 : -1]
        else:
            str_tile = str[last_char_pos:]

        if return_int:
            if str_tile == "None":
                str_tiles.append(None)
            else:
                str_tiles.append(int(str_tile))
        else:
            str_tiles.append(str_tile)

        return str_tiles

    if cfg_file_dict is None:
        cfg_file_dict = get_cfg_file_dict()

    if id_raw_l1_dict is None:
        id_raw_l1_dict = get_id_raw_l1_dict(database, cfg_file_dict)

    try:
        event_class = id_raw_l1_dict[event_id]["class"]
        l1c_dir_name = id_raw_l1_dict[event_id]["l1c"]
        raw_dir_name = id_raw_l1_dict[event_id]["raw"]
        raw_useful_granules_str = id_raw_l1_dict[event_id]["raw_useful_granules"]
        raw_complementary_granules_str = id_raw_l1_dict[event_id][
            "raw_complementary_granules"
        ]
    except:  # noqa: E722
        raise ValueError(
            colored("ERROR", "red")
            + ". The image: "
            + colored(event_id, "blue")
            + " is not in the database "
            + colored(
                os.path.join(cfg_file_dict["database"], database + ".csv"), "blue"
            )
            + "."
        )
    if (raw_useful_granules_str is not None) and (len(raw_useful_granules_str) != 0):
        raw_useful_granules_str = (
            raw_useful_granules_str[1:-1].replace(" ", "").split("],")
        )
        if len(raw_useful_granules_str) == 1:
            # Single granules list
            raw_useful_granules = [
                int(x) if ((x is not None) and (x != "None")) else None
                for x in raw_useful_granules_str[0].split(",")
            ]
        else:
            print(raw_useful_granules_str)
            raw_useful_granules = [
                [int(x), int(y)] if (x != "None") and (y != "None") else [None, None]
                for [x, y] in [
                    x.replace("[", "").replace("]", "").split(",")
                    for x in raw_useful_granules_str
                ]
            ]
    else:
        raw_useful_granules = []

    if raw_complementary_granules_str is not None:
        raw_complementary_granules_str = raw_complementary_granules_str[1:-1]

    raw_img_path = os.path.join(cfg_file_dict["data"], database, "raw", raw_dir_name)
    l1_img_path = os.path.join(cfg_file_dict["data"], database, "l1c", l1c_dir_name)
    l1c_post_processed_path = os.path.join(
        cfg_file_dict["data"], database, "l1c", "l1c_cropped_tif", event_id
    )

    raw_complementary_granules = parse_string(
        raw_complementary_granules_str, skip_marks=False, return_int=True
    )
    coords = id_raw_l1_dict[event_id]["events_coords"]
    requested_polygon = id_raw_l1_dict[event_id]["requested_polygon"]

    return (
        raw_img_path,
        l1_img_path,
        l1c_post_processed_path,
        event_class,
        raw_useful_granules,
        raw_complementary_granules,
        {
            "lat": float(coords[1:].split(",")[0]),
            "lon": float(coords[:-1].split(",")[1]),
        },
        requested_polygon,
    )
