import os
from pyraws.l0.l0_event import L0_event
from pyraws.utils.database_utils import get_events_list

import argparse
import torch
from termcolor import colored
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bands",
        type=str,
        help='bands to coregister list in format ""[Bxx,Byy,...,Bzz]""',
        default="[B02,B08,B03,B10,B04,B05,B11,B06,B07,B8A,B12,B01,B09]",
    )
    parser.add_argument(
        "--output_tif_dir",
        type=str,
        help="output TIF dir.",
        default="my_tif_dir_complementary",
    )

    pargs = parser.parse_args()
    requested_bands_str = pargs.bands
    requested_bands_str = requested_bands_str.replace(" ", "")[1:-1]
    bands = [x for x in requested_bands_str.split(",")]
    output_tif_dir = pargs.output_tif_dir

    os.makedirs(output_tif_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    events_list = get_events_list("THRAWS")

    for event in tqdm(events_list, "Accessing event..."):
        print("Processing event: ", colored(event, "blue") + ".")
        try:
            l0_event = L0_event(device=device)
            l0_event.from_database(event, bands, verbose=False)
        except:  # noqa: E722
            print("Skipping event: ", colored(event, "red") + ".")

        if l0_event.is_void():
            print("Skipping event: ", colored(event, "red") + ".")
            continue

        l0_event_swir = L0_event()
        l0_event_swir.from_database(event, ["B8A", "B11", "B12"], verbose=False)
        l0_event_rgb = L0_event()
        l0_event_rgb.from_database(event, ["B02", "B03", "B04"], verbose=False)
        granules_list = list(range(len(l0_event.get_granules_info().keys())))

        for granule in granules_list:
            l0_granule_n = l0_event.get_granule(granule)
            granule_info = l0_granule_n.get_granule_info()
            save_path_n = os.path.join(
                pargs.output_tif_dir,
                event
                + "_"
                + str(granule)
                + "_"
                + granule_info[0][:3]
                + "_"
                + granule_info[4],
            )
            os.makedirs(save_path_n, exist_ok=True)
            print(
                "Exporting to tif file: " + colored(event + "_" + str(granule), "green")
            )
            l0_granule_n.export_to_tif(save_path_n)
            l0_granule_rgb_n = l0_event_rgb.coarse_coregistration([granule])
            l0_granule_swir_n = l0_event_swir.coarse_coregistration([granule])
            l0_granule_rgb_n.show_bands_superimposition()
            plt.savefig(
                os.path.join(save_path_n, event + "_" + str(granule) + "_rgb.png")
            )
            plt.close()
            l0_granule_swir_n.show_bands_superimposition()
            plt.savefig(
                os.path.join(save_path_n, event + "_" + str(granule) + "_swir.png")
            )

    print("processing " + colored("finished", "green") + ".")


if __name__ == "__main__":
    main()
