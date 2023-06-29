import os
import sys

sys.path.insert(1, "..")
from pyraws.raw.raw_event import Raw_event
from pyraws.l1.l1_event import L1C_event
from pyraws.utils.l1_utils import read_L1C_image_from_tif
from pyraws.utils.database_utils import get_events_list
import torch
from termcolor import colored
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


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
        help="Path to the output directory containing cropped l1c tif files.",
        default=r"C:\Users\Gabriele Meoni\Documents\end2end\end2end\data\l1c\l1c_cropped_tif",
    )
    parser.add_argument(
        "--output_quicklooks_dir",
        type=str,
        help="Path to the output directory containing quicklooks for raw and l1 images.",
        default=r"C:\Users\Gabriele Meoni\Documents\end2end\end2end\data\l1c\l1c_cropped_tif\qls",
    )
    parser.add_argument("--database", type=str, help="Database name.", default="THRAWS")

    pargs = parser.parse_args()
    requested_bands_str = pargs.bands
    requested_bands_str = requested_bands_str.replace(" ", "")[1:-1]
    requested_bands = [x for x in requested_bands_str.split(",")]
    events_list = get_events_list(database=pargs.database)
    output_tif_dir = pargs.output_tif_dir
    output_quicklooks_di = pargs.output_quicklooks_dir

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    plt.rcParams["figure.figsize"] = [10, 10]
    for event in tqdm(
        events_list[1:], desc="Processing events... "
    ):  # Skipping first event, being ":"
        print("Processing event: " + colored(event, "blue"))
        # -------------Create quicklook-------------
        plt.close()
        try:
            raw_event = Raw_event(device=device)
            raw_event.from_database(
                event, requested_bands, verbose=False, database=pargs.database
            )
            l1c_event = L1C_event(device=device)
            l1c_event.from_database(event, requested_bands, database=pargs.database)
        except:  # noqa: E722
            print("Skipping event: ", colored(event, "red"))
            continue

        if raw_event.is_void() or l1c_event.is_void():
            print("Skipping event: ", colored(event, "red"))
            continue

        figure, ax = plt.subplots(1, 2)

        # Extracting useful_granules
        useful_granules = raw_event.get_useful_granules_idx()

        if (
            not (len(useful_granules))
            or (useful_granules is None)
            or any(
                [
                    True if (x[0] is None) or (x[1] is None) else False
                    for x in useful_granules
                ]
            )
        ):
            print("Skipping event: ", colored(event, "red"))
            continue

        for useful_granule in tqdm(useful_granules, desc="Parsing useful granules..."):
            if useful_granule is [None, None]:
                print("Skipping event: ", colored(event, "red"))
                break
            print("Processing granule: " + colored(useful_granule, "cyan"))
            raw_granule = raw_event.coarse_coregistration(
                sorted(useful_granule), crop_empty_pixels=False
            )
            band_shifted_dict = raw_granule.get_bands_coordinates()
            raw_granule_coordinates = band_shifted_dict[requested_bands[0]]
            _ = l1c_event.crop_tile(
                raw_granule_coordinates,
                output_tif_dir,
                out_name_ending=str(useful_granule[0]) + "_" + str(useful_granule[1]),
                lat_lon_format=True,
            )
            l1c_tif, _, _ = read_L1C_image_from_tif(
                event,
                out_name_ending=str(useful_granule[0]) + "_" + str(useful_granule[1]),
                device=device,
            )
            plt.close()
            bands_superimposed = raw_granule.as_tensor()
            bands_superimposed = bands_superimposed / bands_superimposed.max()
            fig, ax = plt.subplots(1, 2)
            if device == torch.device("cuda"):
                ax[0].imshow(bands_superimposed.detach().cpu().numpy())
            else:
                ax[0].imshow(bands_superimposed)

            ax[1].imshow(l1c_tif)
            plt.savefig(
                os.path.join(
                    output_quicklooks_di,
                    event
                    + "_"
                    + str(useful_granule[0])
                    + "_"
                    + str(useful_granule[1])
                    + "_qls.png",
                )
            )


if __name__ == "__main__":
    main()
