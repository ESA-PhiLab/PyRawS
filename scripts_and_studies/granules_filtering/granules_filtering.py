import os
import sys

sys.path.insert(1, os.path.join("..", ".."))
sys.path.insert(1, os.path.join("..", "hta_detection_algorithms"))
from pyraws.raw.raw_event import Raw_event
from pyraws.l1.l1_event import L1C_event
from pyraws.utils.l1_utils import (
    read_L1C_image_from_tif,
    get_event_bounding_box,
)
from pyraws.utils.database_utils import get_events_list, get_cfg_file_dict
from pyraws.utils.constants import DATABASE_FILE_DICTIONARY
from s2pix_detector import s2pix_detector, filter_bbox_list
from glob import glob
import torch
from termcolor import colored
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import csv
import matplotlib.patches as patches
import pandas as pd
import numpy as np


# Function to handle the pixel outside the Raw frame in case of bbox rotating close to the border
def RotationHandler(num: int):
    if num < 0:
        return 0
    else:
        return num


# Function to update csv
def update_csv_file(
    event_name,
    useful_granules,
    bbox_list,
    complementary_granules=[None],
    csv_path="thraws_db.csv",
):
    row_list = []

    with open(csv_path, mode="r", encoding="utf-8-sig") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            row_list.append(row)

    fieldnames = list(row_list[0].keys())
    if not ("bbox_list" in fieldnames):
        fieldnames = fieldnames + ["bbox_list"]
    # print("Updating useful granules: ", useful_granules)

    with open(csv_path, mode="w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            if row["ID_event"] == event_name:
                row["Eaw_useful_granules"] = useful_granules
                row["bbox_list"] = bbox_list
                row["Raw_complementary_granules"] = complementary_granules

            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_tif_dir",
        type=str,
        help="Path to the output directory containing cropped l1c tif files.",
        default=r"C:\Users\Gabriele Meoni\Documents\projects\end2end\pyraws\data\THRAWS\l1c\l1c_cropped_tif",
    )
    parser.add_argument(
        "--output_quicklooks_dir",
        type=str,
        help="Path to the output directory containing quicklooks for raw and l1 images.",
        default=r"C:\Users\Gabriele Meoni\Documents\projects\end2end\pyraws\data\THRAWS\l1c\l1c_cropped_tif\qls",
    )
    parser.add_argument(
        "--skip_l1c_tif_creation",
        action="store_true",
        help="If used, L1C granules mosaicing and cropping on Raw images is skipped.",
        default=False,
    )
    parser.add_argument(
        "--use_single_granules",
        action="store_true",
        help="If used, not-stacked-granules are used.",
        default=False,
    )
    parser.add_argument(
        "--use_all_granules",
        action="store_true",
        help="If used, all the granules will be used. Otherwise, only useful granules will be used..",
        default=False,
    )
    parser.add_argument(
        "--event_start", type=str, help="Event name to start.", default=None
    )
    parser.add_argument(
        "--single_shot",
        action="store_true",
        help="If used, only one event will be checked.",
        default=False,
    )
    parser.add_argument(
        "--event_type",
        type=str,
        help="If specified, only the events of that type will be used.",
        default=None,
    )
    parser.add_argument(
        "--num_pixels_threshold",
        type=int,
        help="Minimum number of active pixels in a bounding box.",
        default=9,
    )

    pargs = parser.parse_args()
    requested_bands = ["B8A", "B11", "B12"]
    output_tif_dir = pargs.output_tif_dir
    output_quicklooks_dir = pargs.output_quicklooks_dir
    output_raw_l1_qls_dir = os.path.join(output_quicklooks_dir, "raw_l1_qls")
    l1_processed_qls_dir = os.path.join(output_quicklooks_dir, "l1_processed_qls")

    os.makedirs(output_tif_dir, exist_ok=True)
    os.makedirs(output_raw_l1_qls_dir, exist_ok=True)
    os.makedirs(l1_processed_qls_dir, exist_ok=True)
    if pargs.skip_l1c_tif_creation:
        Skip_L1C_generation = True
    else:
        Skip_L1C_generation = False

    if pargs.use_all_granules:
        use_all_granules = True
    else:
        use_all_granules = False

    # database path
    database_path = os.path.join(
        get_cfg_file_dict()["database"], DATABASE_FILE_DICTIONARY["THRAWS"]
    )
    warm_events_path = os.path.join(
        get_cfg_file_dict()["database"], DATABASE_FILE_DICTIONARY["THRAWS"]
    )
    warm_events_db = pd.read_csv(warm_events_path).drop_duplicates()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    plt.rcParams["figure.figsize"] = [10, 10]

    if pargs.event_type is not None:
        warm_events_path = os.path.join(
            get_cfg_file_dict()["database"], DATABASE_FILE_DICTIONARY["THRAWS"]
        )
        warm_events_db = pd.read_csv(warm_events_path).drop_duplicates()
        warm_events_db = warm_events_db[warm_events_db["class"] == pargs.event_type]
        warm_events_db.reset_index(inplace=True, drop=True)
        events_list = warm_events_db["ID_event"]
        print(
            "Processing only events marked as: "
            + colored(pargs.event_type, "green")
            + "..."
        )
    else:
        events_list = get_events_list()

    event_useful_granules_dict = dict(
        zip(events_list, [[None, None] for x in events_list[1:]])
    )
    event_bbox_dict = dict(zip(events_list, [[None] for x in events_list[1:]]))
    event_complementary_granules_dict = dict(
        zip(events_list, [[None] for x in events_list[1:]])
    )

    if pargs.event_start is not None:
        start_idx = warm_events_db[
            warm_events_db["ID_event"] == pargs.event_start
        ].index.item()
        # start_idx=events_list.index(pargs.event_start).item()
    else:
        start_idx = 1

    for event in tqdm(events_list[start_idx:], desc="Processing events... "):
        print("Processing event: " + colored(event, "blue"))
        # ------------------Create quicklook---------------------
        plt.close()
        try:
            raw_event = Raw_event(device=device)
            raw_event.from_database(event, requested_bands, verbose=False)
            if not (Skip_L1C_generation):
                l1c_event = L1C_event(device=device)
                l1c_event.from_database(event, requested_bands, verbose=False)
        except:  # noqa: E722
            print(
                "Impossible to open either Raw either L1c data. Skipping event: ",
                colored(event, "red"),
            )
            if pargs.single_shot:
                return
            continue

        if raw_event.is_void():
            print("Raw event is void. Skipping event: ", colored(event, "red"))
            if pargs.single_shot:
                return
            continue

        if not (Skip_L1C_generation):
            if l1c_event.is_void():
                print("L1C event is void. Skipping event: ", colored(event, "red"))
                if pargs.single_shot:
                    return
                continue

        event_useful_granules_event_list = []
        event_complementary_granules_list = []
        event_bbox_list = []

        if pargs.use_single_granules:
            print("Using single granules...")
            raw_granules_names = raw_event.get_granules_names()
            granules = [n for n in range(len(raw_granules_names))]
        else:
            if use_all_granules:
                # Extracting stackable granules
                _, granules = raw_event.stack_granules_couples()

            else:
                # Extracting useful_granules
                granules = raw_event.get_useful_granules_idx()

            if (
                not (len(granules))
                or (granules is None)
                or any(
                    [
                        True if (x[0] is None) or (x[1] is None) else False
                        for x in granules
                    ]
                )
            ):
                print(
                    "Either the granule list is empty or one of the granules' list components is None. Skipping event: ",
                    colored(event, "red"),
                )
                event_useful_granules_event_list = [None, None]
                event_complementary_granules_list = [None, None]
                event_bbox_list = [None]
                if pargs.single_shot:
                    return
                continue
        granules_range = tqdm(granules, desc="Parsing useful granules...")
        for granule in granules_range:
            # Event detected flag
            event_detected = False
            if granule is [None, None]:
                print("Skipping couple (None,None): ", colored(granule, "red"))
                if pargs.single_shot:
                    return
                continue
            if not (pargs.use_single_granules):
                granule_list = sorted(granule)
            else:
                granule_list = [granule]

            raw_granule = raw_event.coarse_coregistration(
                granule_list,
                crop_empty_pixels=True,
                use_complementary_granules=pargs.use_single_granules,
            )
            band_shifted_dict = raw_granule.get_bands_coordinates()
            raw_granule_coordinates = band_shifted_dict[requested_bands[0]]
            if pargs.use_single_granules:
                # In this case granule is a single number, not a list.
                registered_granule_name = raw_granule.get_granule_info()[0]
                if "_COMPLEMENTED_WITH" in registered_granule_name:
                    granules_names = registered_granule_name.split("_COMPLEMENTED_WITH")
                    granule_name = raw_event.get_granule(granule).get_granule_info()[0]
                    if granules_names[0] == granule_name:
                        complementary_granules_name = granules_names[1]
                    else:
                        complementary_granules_name = granules_names[0]

                    complementary_granule_names_list = []

                    complementary_granules_name_bottom = complementary_granules_name.split(
                        "_bottom_"
                    )

                    for name in complementary_granules_name_bottom:
                        complementary_granule_names_list += name.split("_top_")

                    try:
                        complementary_granule_names_list.remove(
                            ""
                        )  # Removing empty names
                    except:  # noqa: E722
                        pass

                    # Removing ending character
                    for name in complementary_granule_names_list:
                        if name[-1] == "_":
                            name = name[:-1]

                    ending = str(granule)
                    for complementary_granule_name in complementary_granule_names_list:
                        complementary_granule_idx = raw_granules_names.index(
                            complementary_granule_name
                        )

                        ending = ending + "_" + str(complementary_granule_idx)
                else:
                    complementary_granule_idx = None
                    ending = str(granule)
            else:
                complementary_granule_idx = None
                ending = str(granule[0]) + "_" + str(granule[1])

            if not (Skip_L1C_generation):
                if os.path.join(output_tif_dir, event + "_" + ending + ".tif") in glob(
                    os.path.join(output_tif_dir, "*")
                ):
                    print(
                        "Skipping creation of L1C crop for the granule: "
                        + colored(granule, "green")
                    )
                else:
                    print(
                        "Creating L1C crop for the granule: " + colored(granule, "cyan")
                    )
                    _ = l1c_event.crop_tile(
                        raw_granule_coordinates,
                        None,
                        verbose=False,
                        out_name_ending=ending,
                        lat_lon_format=True,
                    )
                    l1c_tif, _, _ = read_L1C_image_from_tif(
                        event, out_name_ending=ending, device=device
                    )
                    plt.close()
                    bands_superimposed = raw_granule.as_tensor()
                    bands_superimposed = bands_superimposed / bands_superimposed.max()
                    fig, ax = plt.subplots(1, 2)
                    if device == torch.device("cuda"):
                        ax[0].imshow(bands_superimposed[:, :, :].detach().cpu().numpy())
                    else:
                        ax[0].imshow(bands_superimposed[:, :, :])

                    ax[1].imshow(l1c_tif[:, :, :] / l1c_tif.max())
                    plt.savefig(
                        os.path.join(
                            output_raw_l1_qls_dir, event + "_" + ending + "_qls.png"
                        )
                    )

            try:
                l1c_tif, coords_dict, _ = read_L1C_image_from_tif(
                    event, out_name_ending=ending, device=device, database="THRAWS"
                )
                print("Processing granule: " + colored(granule, "cyan"))
                eruption_prediction, _, l1c_alert_matrix = s2pix_detector(
                    l1c_tif[:, :, :]
                )
                event_detected = eruption_prediction
                if eruption_prediction:
                    bbox_list, _ = get_event_bounding_box(l1c_alert_matrix, coords_dict)
                    bbox_list_filtered, _ = filter_bbox_list(
                        l1c_alert_matrix,
                        bbox_list,
                        event_bbox_coordinates_list=None,
                        num_pixels_threshold=pargs.num_pixels_threshold,
                    )
                    if len(bbox_list_filtered) > 0:
                        raw_tensor = raw_granule.as_tensor(downsampling=True)
                        raw_tensor = raw_tensor / raw_tensor.max()

                        fig, ax = plt.subplots(1, 2, figsize=(20, 20))
                        ax[0].imshow(raw_tensor[:, :, :])
                        ax[1].imshow(l1c_tif[:, :, :] / l1c_tif.max())
                        granule_bbox_xy_list = []

                        for prop in bbox_list_filtered:
                            bbox = prop.bbox
                            rect = patches.Rectangle(
                                (bbox[1], bbox[0]),
                                abs(bbox[1] - bbox[3]),
                                abs(bbox[0] - bbox[2]),
                                linewidth=1,
                                edgecolor="y",
                                facecolor="none",
                            )
                            ax[1].add_patch(rect)

                            poly_list = raw_granule.get_raw_bbox(l1c_tif, bbox)

                            poly_list = [
                                [
                                    RotationHandler(int(np.round(y))),
                                    RotationHandler(int(np.round(x))),
                                ]
                                for [x, y] in poly_list
                            ]
                            granule_bbox_xy_list.append(poly_list)
                            poly = patches.Polygon(
                                poly_list, linewidth=1, edgecolor="y", facecolor="none"
                            )
                            ax[0].add_patch(poly)
                        plt.savefig(
                            os.path.join(
                                l1_processed_qls_dir,
                                event + "_" + ending + "_hotmap_bb.png",
                            )
                        )
                        plt.close()
                    else:
                        print(
                            colored(
                                "Skipping event: all bounding boxes filtered.", "red"
                            )
                        )
                        event_detected = False
                else:
                    print(colored("Skipping event: Predicted as NOT EVENT.", "red"))
            except:  # noqa: E722
                print(
                    "Errors in processing the L1C granule. Skipping event: ",
                    colored(event, "red"),
                )
                if pargs.single_shot:
                    return
                continue

            if event_detected:
                event_useful_granules_event_list.append(granule)
                event_bbox_list.append(granule_bbox_xy_list)
                event_complementary_granules_list.append(complementary_granule_idx)
                print(colored("Eruption found! Updated event: " + event, "green"))

        if len(event_useful_granules_event_list) == 0:  # In case it is still empty.
            event_useful_granules_event_list = [None, None]
            event_complementary_granules_list = [None, None]

        if len(event_bbox_list) == 0:  # In case it is still empty.
            event_bbox_list = [None]

        event_useful_granules_dict[event] = event_useful_granules_event_list
        event_bbox_dict[event] = dict(
            zip(event_useful_granules_event_list, event_bbox_list)
        )
        event_complementary_granules_dict[event] = event_complementary_granules_list
        update_csv_file(
            event,
            event_useful_granules_dict[event],
            event_bbox_dict[event],
            event_complementary_granules_dict[event],
            csv_path=database_path,
        )


if __name__ == "__main__":
    main()
