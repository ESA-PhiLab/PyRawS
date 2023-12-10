import argparse
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.transforms.functional import rotate
import sys
import os

from coregistration_study_utils import get_shift_SuperGlue_profiling, get_shift_SIFT_profiling, get_shift_lightglue_profiling
from pyraws.raw.raw_event import Raw_event
from pyraws.utils.raw_utils import get_bands_shift
import matplotlib.pyplot as plt
from termcolor import colored

import torch


def get_shift_values_dict(bands):
    """Returns shift values dictionary for satellite and detector number.

    Args:
        bands (list): bands list.

    Returns:
        dict: dictionary containing shift values for satellite and detector number.
    """

    shift_values_dict = {
        "S2A": dict(zip(list(range(1, 13)), [[0, 0] for n in range(13)])),
        "S2B": dict(zip(list(range(13)), [[0, 0] for n in range(1, 13)])),
    }

    for detector_number in range(1, 13):
        shift_values_dict["S2A"][detector_number] = get_bands_shift(
            bands, "S2A", detector_number, True, None
        )
        shift_values_dict["S2B"][detector_number] = get_bands_shift(
            bands, "S2B", detector_number, True, None
        )

    return shift_values_dict

def coarse_coregistration(raw_granule, shift_values_dict):
    """Coarse coregistration value from preloaded shift values.

    Args:
        raw_granule (raw_granule): raw granule to coregister.
        shift_values_dict (dict): dictionary containing shift values for satellite and detector number.

    Returns:
        raw_granule: registered granule.
    """
    raw_granule_info = raw_granule.get_granule_info()
    sat = raw_granule_info[0][:3]
    detector_number = raw_granule_info[3][0]
    bands_shifts = shift_values_dict[sat][detector_number]
    return raw_granule.coarse_coregistration(
        crop_empty_pixels=False, downsampling=True, bands_shifts=bands_shifts
    )

def superGlue_coregistration(
    raw_granule,
    requested_bands,
    n_max_keypoints,
    sinkhorn_iterations,
    equalize,
    n_std,
    device,
):
    """_summary_

    Args:
        raw_granule (raw_granule): Raw granule to coregister.
        requested_bands (list): bands lists.
        n_max_keypoints (int): maximum number of keypoints.
        sinkhorn_iterations (int): number of keypoints iterations.
        equalize (bool): if true, equalization is applied.
        n_std (int): number of std values used for equalization.
        device (torch.device): torch device.

    Returns:
        raw_granule: registered granule.
    """
    bands = raw_granule.as_tensor(downsampling=True)
    for n, band in enumerate(requested_bands):
        if band in ["B10", "B11", "B12"]:
            bands[:, :, n] = rotate(bands[:, :, n].unsqueeze(2), 180).squeeze(2)

    bands_shifts = []

    for n in range(bands.shape[2] - 1):
        bands_shifts.append(
            get_shift_SuperGlue_profiling(
                bands[:, :, 0],
                bands[:, :, n + 1],
                n_max_keypoints=n_max_keypoints,
                sinkhorn_iterations=sinkhorn_iterations,
                equalize=equalize,
                n_std=n_std,
                device=device,
            )
        )

    return raw_granule.coarse_coregistration(
        crop_empty_pixels=False, downsampling=True, bands_shifts=bands_shifts
    )

def sift_coregistration(
    raw_granule,
    requested_bands,
    equalize,
    n_std,
    device,
):
    """_summary_

    Args:
        raw_granule (raw_granule): Raw granule to coregister.
        requested_bands (list): bands lists.
        n_max_keypoints (int): maximum number of keypoints.
        sinkhorn_iterations (int): number of keypoints iterations.
        equalize (bool): if true, equalization is applied.
        n_std (int): number of std values used for equalization.
        device (torch.device): torch device.

    Returns:
        raw_granule: registered granule.
    """
    bands = raw_granule.as_tensor(downsampling=True)
    for n, band in enumerate(requested_bands):
        if band in ["B10", "B11", "B12"]:
            bands[:, :, n] = rotate(bands[:, :, n].unsqueeze(2), 180).squeeze(2)

    bands_shifts = []

    for n in range(bands.shape[2] - 1):
        bands_shifts.append(
            get_shift_SIFT_profiling(
                bands[:, :, 0],
                bands[:, :, n + 1],
                equalize=equalize,
                n_std=n_std,
                device=device,
            )
        )

    return raw_granule.coarse_coregistration(
        crop_empty_pixels=False, downsampling=True, bands_shifts=bands_shifts
    )
    
def lightglue_coregistration(
    raw_granule,
    requested_bands,
    equalize,
    n_std,
    device,
    feature_extractor = 'alik', # ALIK, DISK, or superpoint
    width_coefficient = 0.9, # pruning threshold, disable with -1
    depth_coefficient = 0.9, # Controls the early stopping, disable with -1
    filter_threshold = 0.9 # Match confidence. Increase this value to obtain less, but stronger matches.
):
    """_summary_

    Args:
        raw_granule (raw_granule): Raw granule to coregister.
        requested_bands (list): bands lists.
        n_max_keypoints (int): maximum number of keypoints.
        sinkhorn_iterations (int): number of keypoints iterations.
        equalize (bool): if true, equalization is applied.
        n_std (int): number of std values used for equalization.
        device (torch.device): torch device.

    Returns:
        raw_granule: registered granule.
    """
    bands = raw_granule.as_tensor(downsampling=True)
    for n, band in enumerate(requested_bands):
        if band in ["B10", "B11", "B12"]:
            bands[:, :, n] = rotate(bands[:, :, n].unsqueeze(2), 180).squeeze(2)

    bands_shifts = []

    for n in range(bands.shape[2] - 1):
        bands_shifts.append(
            get_shift_lightglue_profiling(
                bands[:, :, 0],
                bands[:, :, n + 1],
                equalize=equalize,
                n_std=n_std,
                device=device,
                feature_extractor = feature_extractor, # alik, disk, sift or superpoint
                width_coefficient = width_coefficient, # pruning threshold, disable with -1
                depth_coefficient = depth_coefficient, # Controls the early stopping, disable with -1
                filter_threshold = filter_threshold # Match confidence. Increase this value to obtain less, but stronger matches.
            )
        )

    return raw_granule.coarse_coregistration(
        crop_empty_pixels=False, downsampling=True, bands_shifts=bands_shifts
    )
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bands",
        type=str,
        help='bands to coregister list in format ""[Bxx,Byy,...,Bzz]"" ',
        default="[B8A, B11, B12]",
    )
    parser.add_argument(
        "--n_event", type=int, help='number of events to coregister" ', default=10
    )
    parser.add_argument(
        "--coreg_type",
        type=str,
        help='Coregistration type between ""coarse"", ""super_glue"", ""SIFT"", ""lightglue""',
        default="coarse",
    )
    parser.add_argument(
        "--device", type=str, help='Device between ""cpu"" and ""gpu""', default="cpu"
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
        "--n_max_keypoints",
        type=int,
        help="Number of maximum keypoints to extract.",
        default=1024,
    )
    parser.add_argument(
        "--extractor",
        type=str,
        help="Feature extractor: 'alik', 'disk', 'sift' or 'superpoint'. Default: 'superpoint' ",
        default='superpoint',
    )
    parser.add_argument(
        "--sinkhorn_iterations",
        type=int,
        help="Number of sinkhorn iterations.",
        default=30,
    )
    parser.add_argument(
        "--test_iteration", type=str, help="Test ieration.", default="1"
    )

    pargs = parser.parse_args()
    requested_bands_str = pargs.bands
    requested_bands_str = requested_bands_str.replace(" ", "")[1:-1]
    requested_bands = [x for x in requested_bands_str.split(",")]
    events = ["Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00", "Barren_Island_00"]
    extractor = str(pargs.extractor )

    # Creating output dirs if needed.
    os.makedirs("tests", exist_ok=True)
    os.makedirs(os.path.join("tests", "images"), exist_ok=True)

    if pargs.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if pargs.n_event > 17:
        raise ValueError("Maximum 17 events to coregister.")

    # Filling events to test
    raw_granules_list = []

    n_inserted = 0

    print(colored("Loading " + str(pargs.n_event) + " granules...", "blue"))

    for event in events:
        raw_event = Raw_event(device=device)

        raw_event.from_database(event, requested_bands, verbose=False)

        n_granules = len(raw_event.get_granules_info())

        for n in range(n_granules):
            if n_inserted < pargs.n_event:
                raw_granules_list.append(raw_event.get_granule(n))
                n_inserted += 1
            else:
                break

        if n_inserted == pargs.n_event:
            break

    shift_values_dict = get_shift_values_dict(requested_bands)

    print(colored("Start profiling...", "green"))

    if pargs.coreg_type == "coarse":
        if pargs.device == "cpu":
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("coarse_coregistration"):
                    for n in range(pargs.n_event):
                        raw_granule_registered = coarse_coregistration(
                            raw_granules_list[n], shift_values_dict
                        )
        else:
            with profile(
                activities=[ProfilerActivity.CUDA], record_shapes=True
            ) as prof:
                with record_function("coarse_coregistration"):
                    for n in range(pargs.n_event):
                        raw_granule_registered = coarse_coregistration(
                            raw_granules_list[n], shift_values_dict
                        )
###############################################
    # else: 
    #     # pargs.coreg_type == "lightglue":
    #     if pargs.device == "cpu":
    #         with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #             with record_function("lightglue_coregistration_"+extractor):
    #                 for n in range(pargs.n_event):
    #                     raw_granule_registered = lightglue_coregistration(
    #                         raw_granules_list[n],
    #                         requested_bands,
    #                         pargs.equalize,
    #                         pargs.n_std,
    #                         device,
    #                         feature_extractor = extractor, # alik, disk, sift or superpoint
    #                         width_coefficient = 0.9, # pruning threshold, disable with -1
    #                         depth_coefficient = 0.9, # Controls the early stopping, disable with -1
    #                         filter_threshold = 0.9, # Match confidence. Increase this value to obtain less, but stronger matches.
    #                     )
    #     else:
    #         with profile(
    #             activities=[ProfilerActivity.CUDA], record_shapes=True
    #         ) as prof:
    #             with record_function("lightglue_coregistration_"+extractor):
    #                 for n in range(pargs.n_event):
    #                     raw_granule_registered = lightglue_coregistration(
    #                         raw_granules_list[n],
    #                         requested_bands,
    #                         pargs.equalize,
    #                         pargs.n_std,
    #                         device,
    #                         feature_extractor = extractor, # alik, disk, sift or superpoint
    #                         width_coefficient = 0.9, # pruning threshold, disable with -1
    #                         depth_coefficient = 0.9, # Controls the early stopping, disable with -1
    #                         filter_threshold = 0.9, # Match confidence. Increase this value to obtain less, but stronger matches.
    #                     )
                        
###############################################
    # if pargs.coreg_type == "SIFT":
    else:
        # print('Using SIFT \n\n\n')
        
        # if pargs.device == "cpu":
        #     with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        #         with record_function("sift_coregistration"):
        #             for n in range(pargs.n_event):
        #                 raw_granule_registered = sift_coregistration(
        #                     raw_granules_list[n],
        #                     requested_bands,
        #                     pargs.equalize,
        #                     pargs.n_std,
        #                     device,
        #                 )
        # else:
        #     with profile(
        #         activities=[ProfilerActivity.CUDA], record_shapes=True
        #     ) as prof:
        #         with record_function("sift_coregistration"):
        #             for n in range(pargs.n_event):
        #                 raw_granule_registered = sift_coregistration(
        #                     raw_granules_list[n],
        #                     requested_bands,
        #                     pargs.equalize,
        #                     pargs.n_std,
        #                     device,
        #                 )
###############################################
                        
    # if pargs.coreg_type == "superglue":
        print('Using Superglue \n\n\n')
        
        if pargs.device == "cpu":
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("superglue_coregistration"):
                    for n in range(pargs.n_event):
                        raw_granule_registered = superGlue_coregistration(
                            raw_granules_list[n],
                            requested_bands,
                            pargs.n_max_keypoints,
                            pargs.sinkhorn_iterations,
                            pargs.equalize,
                            pargs.n_std,
                            device,
                        )
        else:
            with profile(
                activities=[ProfilerActivity.CUDA], record_shapes=True
            ) as prof:
                with record_function("superglue_coregistration"):
                    for n in range(pargs.n_event):
                        raw_granule_registered = superGlue_coregistration(
                            raw_granules_list[n],
                            requested_bands,
                            pargs.n_max_keypoints,
                            pargs.sinkhorn_iterations,
                            pargs.equalize,
                            pargs.n_std,
                            device,
                        )
    # else:
    #     raise KeyError('Invalid feature extractor. Please specify one of: SIFT, lightglue, superglue, or coarse')

    raw_granule_registered.show_bands_superimposition()
    plt.show()
    granule_info = raw_granule_registered.get_granule_info()
    plt.savefig(
        os.path.join(
            "tests",
            "images",
            granule_info[0]
            + "_n_event_"
            + str(pargs.n_event)
            + "_"
            + pargs.coreg_type
            + ".png",
        )
    )
    print(colored("End profiling. Saving results.", "red"))

    f = open(
        os.path.join(
            "tests",
            "profiling_n_images_"
            + str(pargs.n_event)
            + "_coreg_"
            + str(pargs.coreg_type)
            + "_extractor_"
            + extractor
            + "_device_"
            + str(pargs.device)
            + "_equalize_"
            + str(pargs.equalize)
            + "_bands_"
            + pargs.bands[1:-1].replace(",", "_").replace(" ", "")
            + "_test_iteration_"
            + str(pargs.test_iteration)
            + ".txt",
        ),
        "w",
    )
    if pargs.device == "gpu":
        f.write(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total"
            )
        )
    else:
        f.write(
            prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total")
        )
    f.close()


if __name__ == "__main__":
    main()
