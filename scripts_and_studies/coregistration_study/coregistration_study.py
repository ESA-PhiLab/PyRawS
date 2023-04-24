import sys
import os

sys.path.insert(1, os.path.join("..", ".."))
from coregistration_study_utils import (
    get_shift_SuperGlue,
    update_coregistration_analysis,
)
import argparse
from copy import deepcopy
from pyraws.raw.raw_event import Raw_event
from pyraws.utils.database_utils import get_events_list
from pyraws.utils.date_utils import get_timestamp
import matplotlib.pyplot as plt
import numpy as np
import os
from termcolor import colored
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bands",
        type=str,
        help='bands to coregister list in format ""[Bxx,Byy,...,Bzz]"" ',
        default="[B8A, B11]",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        help="Path to the output directory containing the simulation results.",
        default=None,
    )
    parser.add_argument(
        "--output_quicklooks_dir",
        type=str,
        help="Path to the output directory containing quicklooks DIC-based vs SuperGlue-based correlation for every granule.",
        default="compare_coregistration_qls",
    )
    parser.add_argument(
        "--equalize",
        action="store_true",
        help="If used, bands are equalized by cropping outliers in pixels values histogram.",
        default=False,
    )
    parser.add_argument(
        "--n_std",
        type=int,
        help="Outliers are saturated for equalization at histogram_mean*- n_std * histogram_std.",
        default=2,
    )
    parser.add_argument(
        "--n_granule_per_img_max",
        type=int,
        help="Number of stacked granules computed per image. If -1, all are used.",
        default=-1,
    )
    parser.add_argument(
        "--event_class",
        type=str,
        help="If specified, only granules of those classes will be used.",
        default=None,
    )
    parser.add_argument(
        "--detector_n",
        type=int,
        help="If specified, only granules with that detector number will be used.",
        default=None,
    )

    pargs = parser.parse_args()
    requested_bands_str = pargs.bands
    requested_bands_str = requested_bands_str.replace(" ", "")[1:-1]
    requested_bands = [x for x in requested_bands_str.split(",")]
    events_list = get_events_list()
    output_dir_path = pargs.output_dir_path

    if output_dir_path is None:
        output_dir_path = os.getcwd()

    qls_out_dir = pargs.output_quicklooks_dir
    n_granule_per_image_max = pargs.n_granule_per_img_max
    event_class_req = pargs.event_class
    detector_number_req = pargs.detector_n

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    plt.rcParams["figure.figsize"] = [40, 40]

    qls_out_dir = qls_out_dir + "_" + get_timestamp()

    os.makedirs(qls_out_dir, exist_ok=True)
    events_list = get_events_list()
    coregistration_database = []
    csv_name = os.path.join(
        output_dir_path,
        "coregistration_study_"
        + requested_bands[0]
        + "_"
        + requested_bands[1]
        + "_"
        + get_timestamp()
        + ".csv",
    )
    for event in tqdm(events_list[1:], desc="Processing events... "):
        n_granule_per_image = 0
        print("Processing event: " + colored(event, "green"))
        try:
            raw_event = Raw_event(device=device)
            raw_event.from_database(event, requested_bands, verbose=False)
        except:
            print("Skipping event: ", colored(event, "red"))
            continue

        if raw_event.is_void():
            print("Skipping event: ", colored(event, "red"))
            continue

        # Checking class
        if event_class_req is not None:
            if raw_event.get_event_class() != event_class_req:
                print(
                    "The event class does not match the requested one. Skipping event: ",
                    colored(event, "red"),
                )
                continue

        # Extracting stackable granules
        _, stackable_couples = raw_event.stack_granules_couples()

        if (
            not (len(stackable_couples))
            or (stackable_couples is None)
            or any(
                [
                    True if (x[0] is None) or (x[1] is None) else False
                    for x in stackable_couples
                ]
            )
        ):
            print("Skipping event: ", colored(event, "red"))
            continue

        for stackable_couple in tqdm(
            stackable_couples, desc="Parsing stackable granules..."
        ):
            if (n_granule_per_image < n_granule_per_image_max) or (
                n_granule_per_image_max <= 0
            ):
                granule_info_dict = {
                    "ID_event": event,
                    "granule_couple": [None, None],
                    "coordinates": [None, None],
                    "sensing_time": 0,
                    "detector_number": 0,
                    "N_v": 0,
                    "N_h": 0,
                    "N_eff": 0,
                }
                if stackable_couple is [None, None]:
                    print("Skipping couple: ", colored(stackable_couple, "red"))
                    continue

                try:
                    print("Processing granule: " + colored(stackable_couple, "cyan"))
                    event_name = (
                        event
                        + "_"
                        + str(stackable_couple[0])
                        + "_"
                        + str(stackable_couple[1])
                    )
                    raw_granule = raw_event.stack_granules(stackable_couple, ["T"])

                    if detector_number_req is not None:
                        if raw_granule.get_granule_info()[3][0] != detector_number_req:
                            print(
                                "The detector number does not match the requested one. Skipping: ",
                                colored(event_name, "red"),
                            )
                            continue

                    shift, _, _, granule_info, _ = get_shift_SuperGlue(
                        event_name,
                        raw_granule,
                        requested_bands,
                        qls_out_dir,
                        pargs.equalize,
                        pargs.n_std,
                        device,
                    )
                    if (shift[0] is None) or (shift[1] is None):
                        print("Skipping event: ", colored(event, "red"))
                        continue

                    sensing_time = granule_info[1][0]
                    detector_number = granule_info[3][0]

                    coordinates = np.array(granule_info[6])
                    center_coordinates = (
                        coordinates[0] + (coordinates[2] - coordinates[0]) / 2
                    )
                    granule_info_dict["granule_couple"] = stackable_couple
                    granule_info_dict["sensing_time"] = sensing_time
                    granule_info_dict["detector_number"] = detector_number
                    granule_info_dict["coordinates"] = center_coordinates
                    granule_info_dict["N_v"] = float(shift[1])
                    granule_info_dict["N_h"] = float(shift[0])
                    granule_info_dict["N_eff"] = np.sqrt(
                        float(shift[0]) ** 2 + float(shift[1]) ** 2
                    )
                    coregistration_database.append(deepcopy(granule_info_dict))
                    update_coregistration_analysis(
                        csv_name,
                        list(granule_info_dict.keys()),
                        coregistration_database,
                    )
                    n_granule_per_image += 1
                except:
                    print("Skipping couple: ", colored(stackable_couple, "red"))
                    continue
            else:
                print(
                    "Maximum number of stacked granules per image reached.",
                    colored(stackable_couple, "red"),
                )
                break


if __name__ == "__main__":
    main()
