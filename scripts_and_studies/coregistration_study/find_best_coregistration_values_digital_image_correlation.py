import csv
import torch
import argparse
from termcolor import colored
from tqdm import tqdm
from pyraws.utils.parse_csv_utils import parse_csv
from torchvision.transforms.functional import rotate

from pyraws.raw.raw_event import Raw_event
from pyraws.utils.constants import (
    SWIR_BANDS,
    BAND_SPATIAL_RESOLUTION_DICT,
    BANDS_RAW_SHAPE_DICT,
)


def update_csv_shift_lut(csv_name, bands_shifts_dict):
    band_names = list(bands_shifts_dict.keys())

    with open(csv_name, mode="w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=band_names)
        writer.writeheader()
        for band_x in band_names:
            row = bands_shifts_dict[band_x]
            writer.writerow(row)


def get_correlation(bx_scaled, by_shifted):
    return torch.sum(by_shifted * bx_scaled) / torch.sqrt(
        torch.sum(bx_scaled * bx_scaled) * torch.sum(by_shifted * by_shifted)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--event_name", type=str, help="Event ID", default="Piton_de_la_Fournaise_01"
    )
    parser.add_argument(
        "--bands",
        type=str,
        help='bands to coregister list in format ""[Bxx,Byy,...,Bzz]""',
        default="[B02,B08,B03,B10,B04,B05,B11,B06,B07,B8A,B12,B01,B09]",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="LUT csv_path.",
        default=r"C:\Users\Gabriele Meoni\Documents\end2end\end2end\end2end\database\shift_lut_algo.csv",
    )
    parser.add_argument(
        "--simulation_step", type=int, help="Grid simulation step.", default=10
    )
    parser.add_argument(
        "--tiles_to_stack",
        type=str,
        help='Tiles to stack in format ""[N,M]"".',
        default="[0,1]",
    )
    parser.add_argument(
        "--n_early",
        type=int,
        help="Number of iterations without optimum after which the search is stopped for a specific (Bx,By).",
        default=10,
    )
    parser.add_argument(
        "--csv_start", type=str, help="CSV path with starting values.", default=None
    )
    parser.add_argument(
        "--downsampling",
        action="store_true",
        help="If used, coregistration is performed by downsampling images to the lowest resolution.",
    )

    pargs = parser.parse_args()
    event_name = pargs.event_name
    requested_bands_str = pargs.bands
    requested_bands_str = requested_bands_str.replace(" ", "")[1:-1]
    band_names = [x for x in requested_bands_str.split(",")]
    csv_path = pargs.csv_path
    csv_path = csv_path.split(".")
    if len(csv_path) > 1:
        csv_path = csv_path[0] + "_" + event_name + "." + csv_path[1]
    else:
        csv_path = csv_path[0]

    if pargs.downsampling:
        downsampling = True
    else:
        downsampling = False

    simulation_step = pargs.simulation_step
    tiles_to_stack_str = pargs.tiles_to_stack
    tiles_to_stack = [int(x) for x in tiles_to_stack_str[1:-1].split(",")]
    n_early = pargs.n_early

    print("Processing event: " + colored(event_name, "blue"))
    print("Processing bands: " + colored(band_names, "red"))
    print("Processing tiles: " + colored(tiles_to_stack, "cyan"))
    if downsampling:
        print(colored("Downsampling mode", "yellow"))
    else:
        print(colored("Upsampling mode: ", "yellow"))

    print("Saving output data to: " + colored(csv_path, "green"))

    band_all_names = [
        "B02",
        "B08",
        "B03",
        "B10",
        "B04",
        "B05",
        "B11",
        "B06",
        "B07",
        "B8A",
        "B12",
        "B01",
        "B09",
    ]
    band_shifts_lut_row = dict(
        zip(band_all_names, [0.0 for n in range(len(band_all_names))])
    )
    band_shifts_lut = dict(
        zip(
            list(band_shifts_lut_row.keys()),
            [band_shifts_lut_row for n in range(len(band_all_names))],
        )
    )

    if pargs.csv_start is not None:
        start_values_lut = parse_csv(pargs.csv_start)
        start_values_lut = dict(zip(band_all_names, start_values_lut))
    else:
        start_values_lut = None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    band_all_names = [
        "B02",
        "B08",
        "B03",
        "B10",
        "B04",
        "B05",
        "B11",
        "B06",
        "B07",
        "B8A",
        "B12",
        "B01",
        "B09",
    ]

    for band_name_x in band_names:
        band_shifts_lut_row = dict(
            zip(band_all_names, [0.0 for n in range(len(band_all_names))])
        )

        for band_name_y in band_all_names:
            print(
                "-------------------"
                + band_name_x
                + "/"
                + band_name_y
                + "-------------------\n"
            )

            if band_name_y != band_name_x:
                if (
                    BAND_SPATIAL_RESOLUTION_DICT[band_name_x]
                    == BAND_SPATIAL_RESOLUTION_DICT[band_name_y]
                ) and (band_shifts_lut[band_name_y][band_name_x] != 0):
                    best_shift = -band_shifts_lut[band_name_y][band_name_x]
                    print("Exploiting simmetry and previous results.")
                else:
                    n_without_optimum = 0
                    raw_event = Raw_event(device=device)
                    raw_event.from_database(
                        event_name, [band_name_x, band_name_y], verbose=False
                    )
                    stacked_granules = Raw_event.stack_granules(
                        tiles_to_stack, ["T" for n in range(len(tiles_to_stack) - 1)]
                    )
                    stacked_granules_tensor = stacked_granules.as_tensor(
                        downsampling=downsampling
                    )
                    if band_all_names.index(band_name_y) > band_all_names.index(
                        band_name_x
                    ):
                        sign = 1 - 2 * stacked_granules.get_detectors_number()[0] % 2
                    else:
                        sign = -1 * (
                            1 - stacked_granules.get_detectors_number()[0] % 2
                        ) + (stacked_granules.get_detectors_number()[0] % 2)

                    bx = stacked_granules_tensor[:, :, 0]
                    by = stacked_granules_tensor[:, :, 1]
                    if band_name_y in SWIR_BANDS:
                        by = rotate(by.unsqueeze(2), 180).squeeze(2)

                    if band_name_x in SWIR_BANDS:
                        bx = rotate(bx.unsqueeze(2), 180).squeeze(2)

                    bx_scaled = bx - bx.mean()

                    if start_values_lut is not None:
                        start_value = int(start_values_lut[band_name_x][band_name_y])
                        print(
                            colored("Starting from: " + str(start_value) + ".", "cyan")
                        )
                        if not (stacked_granules.get_detectors_number()[0] % 2):
                            start_value = -start_value

                        best_shift = start_value - simulation_step
                        print("sart_value")

                        shift_range = [
                            range(
                                start_value - simulation_step,
                                start_value + simulation_step,
                            )
                        ]
                    else:
                        shift_max = max(
                            BANDS_RAW_SHAPE_DICT[band_name_x][0],
                            BANDS_RAW_SHAPE_DICT[band_name_y][0],
                        )
                        shift_range = [
                            range(1, shift_max - 1, simulation_step)
                            if sign > 0
                            else range(-1, -(shift_max) + 1, -simulation_step)
                        ]
                        start_value = 0
                        best_shift = 0

                    best_correlation = -1

                    optimum_reached = False
                    negative_correlation_found = False
                    for shift in tqdm(shift_range[0]):
                        try:
                            coarse_coregistered_granule = (
                                stacked_granules.coarse_coregistration(
                                    crop_empty_pixels=False,
                                    bands_shifts=[shift],
                                    downsampling=downsampling,
                                )
                            )
                            coarse_coregistered_granule_tensor = (
                                coarse_coregistered_granule.as_tensor(
                                    downsampling=downsampling
                                )
                            )
                            # coarse_coregistered_granule_tensor=coarse_coregistered_granule_tensor/coarse_coregistered_granule_tensor.max()
                            by_shifted = coarse_coregistered_granule_tensor[:, :, 1]
                            by_shifted = by_shifted - by_shifted.mean()

                            correlation = float(get_correlation(bx_scaled, by_shifted))

                            if (
                                (correlation < 0)
                                and (shift == start_value)
                                and (start_values_lut is not None)
                            ):
                                # Impossible to coregister, bands are too dissimilar.
                                best_shift = "NaN"
                                band_shifts_lut_row[band_name_y] = best_shift
                                band_shifts_lut[band_name_x] = band_shifts_lut_row
                                print(
                                    colored(
                                        "Found negative correalations at start value. Skipping bands couple.",
                                        "red",
                                    )
                                )
                                print(
                                    colored(
                                        "------------------Saving simulation results-------------------------------",
                                        "red",
                                    )
                                )
                                update_csv_shift_lut(csv_path, band_shifts_lut)
                                negative_correlation_found = True
                                break

                            if best_correlation < correlation:
                                print(
                                    "Found new best correlation value: "
                                    + colored(
                                        "old (" + str(best_correlation) + ") ", "blue"
                                    )
                                    + colored("new (" + str(correlation) + ") ", "red")
                                )
                                print(
                                    "Found new shift value: "
                                    + colored("old (" + str(best_shift) + ") ", "blue")
                                    + colored("new (" + str(shift) + ") ", "red")
                                )
                                best_correlation = correlation
                                best_shift = shift
                                n_without_optimum = 0
                                optimum_reached = True
                            else:
                                if optimum_reached:
                                    n_without_optimum += 1

                            if n_without_optimum == n_early:
                                break
                        except:  # noqa: E722
                            continue

                    if not (negative_correlation_found):
                        if not (stacked_granules.get_detectors_number()[0] % 2):
                            best_shift = -best_shift

                band_shifts_lut_row[band_name_y] = best_shift
                band_shifts_lut[band_name_x] = band_shifts_lut_row
                print(
                    colored(
                        "------------------Saving simulation results-------------------------------",
                        "green",
                    )
                )
                update_csv_shift_lut(csv_path, band_shifts_lut)


if __name__ == "__main__":
    main()
