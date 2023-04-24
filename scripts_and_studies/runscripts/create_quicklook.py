import os
import sys

sys.path.insert(1, os.path.join("..", ".."))
sys.path.insert(1, os.path.join("..", "hta_detection_algorithms"))
from pyraws.raw.raw_event import Raw_event
from pyraws.l1.l1_event import L1C_event
from pyraws.utils.l1_utils import (
    read_L1C_image_from_tif,
    read_L1C_image_from_tif,
    get_event_bounding_box,
    get_l1C_image_default_path,
)
from s2pix_detector import s2pix_detector
from glob import glob
import torch
from termcolor import colored
import matplotlib.pyplot as plt
import argparse
import matplotlib.patches as patches


def swap_tensor(tensor):
    print("init:", tensor.shape)
    A = tensor[:, :, 0]
    B = tensor[:, :, 1]
    C = tensor[:, :, 2]
    end = torch.stack([C, B, A]).permute(1, 2, 0)
    print("end:", end.shape)
    return end


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events_list", type=str, help="event list.", default="[Etna_00]"
    )
    parser.add_argument(
        "--granules_number_list", type=str, help="granules number list.", default="[0]"
    )
    parser.add_argument(
        "--bands",
        type=str,
        help='bands to coregister list in format ""[Bxx,Byy,...,Bzz]""',
        default="[B8A,B11,B12]",
    )
    parser.add_argument("--database", type=str, help="database name", default="THRAWS")
    parser.add_argument("--font_size", type=int, help="font size in images", default=20)

    pargs = parser.parse_args()
    requested_bands_str = pargs.bands
    requested_bands_str = requested_bands_str.replace(" ", "")[1:-1]
    bands_list = [x for x in requested_bands_str.split(",")]
    events_list = pargs.events_list
    events_list_str = events_list.replace(" ", "")[1:-1]
    events_list = [x for x in events_list_str.split(",")]
    events_list = pargs.events_list
    events_list_str = events_list.replace(" ", "")[1:-1]
    events_list = [x for x in events_list_str.split(",")]
    database = pargs.database
    granules_number_list = pargs.granules_number_list
    granules_number_str = granules_number_list.replace(" ", "")[1:-1]
    granules_numbers_list = [int(x) for x in granules_number_str.split(",")]
    events_list = pargs.events_list
    events_list_str = events_list.replace(" ", "")[1:-1]
    events_list = [x for x in events_list_str.split(",")]
    plt.rcParams.update({"font.size": pargs.font_size})

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    plt.rcParams["figure.figsize"] = [10, 10]

    fig, ax = plt.subplots(len(events_list), 4, figsize=(20, 15))
    n = 0
    for event, granule in zip(events_list, granules_numbers_list):
        try:
            print("Parsing L0 event: " + colored(event, "green"))
            raw_event = Raw_event(device=device)
            raw_event.from_database(event, bands_list, verbose=False, database=database)
        except:
            raise ValueError(
                "Impossible to open L0 data for: "
                + colored(event, "red")
                + ". Check it is included in the database: "
                + colored(event, "red")
                + "."
            )

        if raw_event.is_void():
            raise ValueError("L0 data for " + colored(event, "red") + " is void.")

        try:
            raw_granule = raw_event.get_granule(granule)
        except:
            raise ValueError(
                "Impossible to get the granule:"
                + colored(str(granule), "red")
                + " for the event: "
                + colored(event, "red")
                + "."
            )

        raw_granule_registered = raw_event.coarse_coregistration(
            [granule], crop_empty_pixels=True, use_complementary_granules=True
        )
        band_shifted_dict = raw_granule.get_bands_coordinates()
        raw_granule_coordinates = band_shifted_dict[bands_list[0]]

    if event + "_" + ending + ".tif" in glob(os.path.join(".", "*")):
        print(
            "Skipping creation of L1C crop for the granule: "
            + colored(granule, "green")
        )
    else:
        print("Creating L1C crop for the granule: " + colored(granule, "cyan"))
        _ = l1c_event.crop_tile(
            raw_granule_coordinates,
            None,
            verbose=False,
            out_name_ending=ending,
            lat_lon_format=True,
        )

        registered_granule_name = raw_granule_registered.get_granule_info()[0]
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
                complementary_granule_names_list.remove("")  # Removing empty names
            except:
                pass

            # Removing ending character
            for name in complementary_granule_names_list:
                if name[-1] == "_":
                    name = name[:-1]

            ending = str(granule)
            raw_granules_names = raw_event.get_granules_names()

            for complementary_granule_name in complementary_granule_names_list:
                complementary_granule_idx = raw_granules_names.index(
                    complementary_granule_name
                )

                ending = ending + "_" + str(complementary_granule_idx)
        else:
            complementary_granule_idx = None
            ending = str(granule)

        l1c_tif_path = get_l1C_image_default_path(event, database=database)

        if l1c_tif_path + "_" + ending + ".tif" in glob(
            os.path.join(
                l1c_tif_path[: -len(l1c_tif_path.split(os.path.sep)[-1]) - 1], "*"
            )
        ):
            print(
                "Skipping creation of L1C crop for the granule: "
                + colored(granule, "green")
            )
        else:
            print("Creating L1C crop for the granule: " + colored(granule, "cyan"))
            try:
                print("Parsing L1C event: " + colored(event, "green"))
                l1c_event = L1C_event(device=device)
                l1c_event.from_database(
                    event, bands_list, verbose=False, database=database
                )
            except:
                raise ValueError(
                    "Impossible to open L1 data for: "
                    + colored(event, "red")
                    + ". Check it is included in the database: "
                    + colored(event, "red")
                    + "."
                )

            if l1c_event.is_void():
                raise ValueError("L1C data for " + colored(event, "red") + " is void.")

            _ = l1c_event.crop_tile(
                raw_granule_coordinates,
                None,
                verbose=False,
                out_name_ending=ending,
                lat_lon_format=True,
            )

        try:
            l1c_tif, coords_dict, _ = read_L1C_image_from_tif(
                event, out_name_ending=ending, device=device, database="THRAWS"
            )
            print("Processing granule: " + colored(granule, "cyan"))
            eruption_prediction, l1c_filtered_alert_matrix, _ = s2pix_detector(
                l1c_tif[:, :, :]
            )
            if eruption_prediction:
                bbox_list, _ = get_event_bounding_box(
                    l1c_filtered_alert_matrix, coords_dict
                )
                raw_tensor = raw_granule.as_tensor(downsampling=True)
                raw_tensor_registered = raw_granule_registered.as_tensor(
                    downsampling=True
                )
                raw_tensor = raw_tensor / raw_tensor.max()
                raw_tensor_registered = (
                    raw_tensor_registered / raw_tensor_registered.max()
                )
                l1c_tif = l1c_tif / l1c_tif.max()

                ax[n, 0].imshow(swap_tensor(raw_tensor))
                ax[n, 1].imshow(swap_tensor(raw_tensor_registered))
                ax[n, 2].imshow(swap_tensor(l1c_tif))
                ax[n, 3].imshow(swap_tensor(raw_tensor_registered))

                for prop in bbox_list:
                    bbox = prop.bbox
                    rect = patches.Rectangle(
                        (bbox[1], bbox[0]),
                        abs(bbox[1] - bbox[3]),
                        abs(bbox[0] - bbox[2]),
                        linewidth=1,
                        edgecolor="y",
                        facecolor="none",
                    )
                    ax[n, 2].add_patch(rect)

                bbox_list = raw_event.get_bounding_box_dict()[granule]
                raw_granule = raw_event.coarse_coregistration(
                    [granule], crop_empty_pixels=True, use_complementary_granules=True
                )
                for bbox in bbox_list:
                    bbox_int = [[int(x), int(y)] for [x, y] in bbox]
                    poly = patches.Polygon(
                        bbox_int, linewidth=1, edgecolor="y", facecolor="none"
                    )
                    ax[n, 3].add_patch(poly)

                ax[n, 0].set_title("Raw Original")
                ax[n, 1].set_title("Raw Coregistered")
                ax[n, 2].set_title("L1C cropped + bound. boxes")
                ax[n, 3].set_title("Raw + warped bound. boxes")
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

            else:
                print(colored("Warning:", "red") + " no bounding box found.")
        except:
            print(
                "Errors in processing the L1C granule. Skipping event: ",
                colored(event, "red"),
            )
        n += 1
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.savefig("quicklook_hotmap_bb.png")
    plt.close()


if __name__ == "__main__":
    # python .\create_quicklook.py --events_list "[Etna_00,Piton_de_la_Fournaise_31, Australia_1]" --bands "[B8A,B11,B12]" --granules_number_list "[2,0,1]"
    main()
