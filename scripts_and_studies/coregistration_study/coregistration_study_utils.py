try:
    from superglue_models.matching import Matching
    from superglue_models.utils import make_matching_plot
except:  # noqa: E722
    raise ValueError(
        "SuperGlue model not found. Please, follow the instructions at: "
        + "https://github.com/ESA-PhiLab/PyRawS#set-up-for-coregistration-study."
    )

import csv
import sys
import os

sys.path.insert(1, os.path.join("..", ".."))
from pyraws.utils.visualization_utils import equalize_tensor
import matplotlib.cm as cm
import torch
import kornia
from kornia.feature import *



def get_shift_SuperGlue_profiling(
    b0,
    b1,
    n_max_keypoints=1024,
    sinkhorn_iterations=30,
    equalize=True,
    n_std=2,
    device=torch.device("cpu"),
):
    """Get a shift between two bands of a specific event by using SuperGlue.

    Args:
        b0 (torch.tensor): tensor containing band 0 to coregister.
        b1 (torch.tensor): tensor containing band 1 to coregister.
        n_max_keypoints (int, optional): number of max keypoints to match. Defaults to 1024.
        sinkhorn_iterations (int, optional): number of sinkorn iterations. Defaults to 30.
        requested_bands (list): list containing two bands for which perform the study.
        equalize (bool, optional): if True, equalization is performed. Defaults to True.
        n_std (int, optional): Outliers are saturated for equalization at histogram_mean*- n_std * histogram_std.
                               Defaults to 2.
        device (torch.device, optional): torch.device. Defaults to torch.device("cpu").

    Returns:
        float: mean value of the shift.
        torch.tensor: band 0.
        torch.tensor: band 1.
        dict: granule info.
        float: number of matched kyepoints.
    """

    config = {
        "superpoint": {
            "nms_radius": 1,
            "keypoint_threshold": 0.05,
            "max_keypoints": n_max_keypoints,
        },
        "superglue": {
            "weights": "outdoor",
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": 0.9,
        },
    }
    matching = Matching(config).eval().to(device)
    bands = torch.zeros([b0.shape[0], b0.shape[1], 2], device=device)
    bands[:, :, 0] = b0
    bands[:, :, 1] = b1
    if equalize:
        l0_granule_tensor_equalized = equalize_tensor(bands[:, :, :2], n_std)
        b0 = (
            l0_granule_tensor_equalized[:, :, 0]
            / l0_granule_tensor_equalized[:, :, 0].max()
        )
        b1 = (
            l0_granule_tensor_equalized[:, :, 1]
            / l0_granule_tensor_equalized[:, :, 1].max()
        )
    else:
        b0 = bands[:, :, 0] / bands[:, :, 0].max()
        b1 = bands[:, :, 1] / bands[:, :, 1].max()

    pred = matching(
        {"image0": b0.unsqueeze(0).unsqueeze(0), "image1": b1.unsqueeze(0).unsqueeze(0)}
    )
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
    matches, _ = pred["matches0"], pred["matching_scores0"]
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    if len(mkpts1) and len(mkpts0):
        shift = torch.tensor([x - y for (x, y) in zip(mkpts1, mkpts0)], device=device)
        shift_v, shift_h = shift[:, 0], shift[:, 1]
        shift_v_mean, shift_v_std = torch.mean(shift_v), torch.std(shift_v)
        shift_h_mean, shift_h_std = torch.mean(shift_h), torch.std(shift_h)
        shift_v = shift_v[
            torch.logical_and(
                shift_v > shift_v_mean - shift_v_std,
                shift_v < shift_v_mean + shift_v_std,
            )
        ]
        shift_h = shift_h[
            torch.logical_and(
                shift_h > shift_h_mean - shift_h_std,
                shift_h < shift_h_mean + shift_h_std,
            )
        ]
        shift_mean = torch.round(
            torch.tensor([-shift_h.mean(), -shift_v.mean()], device=device)
        )
    else:
        return [None, None]
    return shift_mean


def get_shift_SIFT_profiling(
    b0,
    b1,
    equalize=True,
    n_std=2,
    device=torch.device("cpu"),
):
    """Get a shift between two bands of a specific event by using SuperGlue.

    Args:
        b0 (torch.tensor): tensor containing band 0 to coregister.
        b1 (torch.tensor): tensor containing band 1 to coregister.
        equalize (bool, optional): if True, equalization is performed. Defaults to True.
        n_std (int, optional): Outliers are saturated for equalization at histogram_mean*- n_std * histogram_std.
                               Defaults to 2.
        device (torch.device, optional): torch.device. Defaults to torch.device("cpu").

    Returns:
        float: mean value of the shift.
        torch.tensor: band 0.
        torch.tensor: band 1.
        dict: granule info.
        float: number of matched kyepoints.
    """
    # Aux:
    def compute_offsets(image1, image2, device, verbose = False):
        assert len(image1.shape) == 2, 'Error with shapes of image1'
        assert len(image2.shape) == 2, 'Error with shapes of image2'

        PS = 16
        # Initialize SIFT descriptor
        sift = kornia.feature.SIFTDescriptor(PS, rootsift=True).to(device)
        descriptor = sift

        # Set up components for feature detection
        resp = kornia.feature.BlobDoG()
        scale_pyr = kornia.geometry.ScalePyramid(3, 1.6, PS, double_image=True)
        nms = kornia.geometry.ConvQuadInterp3d(10)
        n_features = 4000
        detector = kornia.feature.ScaleSpaceDetector(
            n_features,
            resp_module=resp,
            scale_space_response=True,  # Required for DoG
            nms_module=nms,
            scale_pyr_module=scale_pyr,
            ori_module=kornia.feature.LAFOrienter(19),
            mr_size=6.0,
            minima_are_also_good=True
        ).to(device)

        # Process each image
        def process_image(img):
            with torch.no_grad():
                lafs, _ = detector(img)
                patches = kornia.feature.extract_patches_from_pyramid(img, lafs, PS)
                B, N, CH, H, W = patches.size()
                descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
            return lafs, descs

        lafs1, descs1 = process_image(image1.unsqueeze(0).unsqueeze(0))
        lafs2, descs2 = process_image(image2.unsqueeze(0).unsqueeze(0))
        # Match features between the two images
        scores, matches = kornia.feature.match_snn(descs1[0], descs2[0], 0.95)
        
        # Compute Homography and inliers
        src_pts = lafs1[0, matches[:, 0], :, 2].data.cpu().numpy()
        dst_pts = lafs2[0, matches[:, 1], :, 2].data.cpu().numpy()
        return src_pts, dst_pts

    bands = torch.zeros([b0.shape[0], b0.shape[1], 2], device=device)
    bands[:, :, 0] = b0
    bands[:, :, 1] = b1
    if equalize:
        l0_granule_tensor_equalized = equalize_tensor(bands[:, :, :2], n_std)
        b0 = (
            l0_granule_tensor_equalized[:, :, 0]
            / l0_granule_tensor_equalized[:, :, 0].max()
        )
        b1 = (
            l0_granule_tensor_equalized[:, :, 1]
            / l0_granule_tensor_equalized[:, :, 1].max()
        )
    else:
        b0 = bands[:, :, 0] / bands[:, :, 0].max()
        b1 = bands[:, :, 1] / bands[:, :, 1].max()

    mkpts0, mkpts1 = compute_offsets(b0, b1, device=device)

    if len(mkpts1) and len(mkpts0):
        shift = torch.tensor([x - y for (x, y) in zip(mkpts1, mkpts0)], device=device)
        shift_v, shift_h = shift[:, 0], shift[:, 1]
        shift_v_mean, shift_v_std = torch.mean(shift_v), torch.std(shift_v)
        shift_h_mean, shift_h_std = torch.mean(shift_h), torch.std(shift_h)
        shift_v = shift_v[
            torch.logical_and(
                shift_v > shift_v_mean - shift_v_std,
                shift_v < shift_v_mean + shift_v_std,
            )
        ]
        shift_h = shift_h[
            torch.logical_and(
                shift_h > shift_h_mean - shift_h_std,
                shift_h < shift_h_mean + shift_h_std,
            )
        ]
        shift_mean = torch.round(
            torch.tensor([-shift_h.mean(), -shift_v.mean()], device=device)
        )
    else:
        return [None, None]
    return shift_mean



def get_shift_SuperGlue(
    event_name,
    raw_granule,
    requested_bands,
    save_path=None,
    equalize=True,
    n_std=2,
    device=torch.device("cpu"),
):
    """Get a shift between two bands of a specific event by using SuperGlue.

    Args:
        event_name (str): event_Granule_IDX0_IDX1.
        raw_granule (Raw_granule): raw granule.
        requested_bands (list): list containing two bands for which perform the study.
        save_path (str, optional): path to save quicklooks. No quicklook is generated if None. Defaults to None.
        equalize (bool, optional): if True, equalization is performed. Defaults to True.
        n_std (int, optional): Outliers are saturated for equalization at histogram_mean*- n_std * histogram_std.
                               Defaults to 2.
        device (torch.device, optional): torch.device. Defaults to torch.device("cpu").

    Returns:
        float: mean value of the shift.
        torch.tensor: band 0.
        torch.tensor: band 1.
        dict: granule info.
        float: number of matched kyepoints.
    """
    for band in requested_bands:
        if band in ["B10", "B11", "B12"]:
            raw_granule.rotate_band(band)

    config = {
        "superpoint": {
            "nms_radius": 1,
            "keypoint_threshold": 0.05,
            "max_keypoints": 1024,
        },
        "superglue": {
            "weights": "outdoor",
            "sinkhorn_iterations": 30,
            "match_threshold": 0.9,
        },
    }
    matching = Matching(config).eval().to(device)
    raw_granule_tensor = raw_granule.as_tensor(downsampling=True)
    if equalize:
        raw_granule_tensor_equalized = equalize_tensor(
            raw_granule_tensor[:, :, :2], n_std
        )
        b0 = (
            raw_granule_tensor_equalized[:, :, 0]
            / raw_granule_tensor_equalized[:, :, 0].max()
        )
        b1 = (
            raw_granule_tensor_equalized[:, :, 1]
            / raw_granule_tensor_equalized[:, :, 1].max()
        )
    else:
        b0 = raw_granule_tensor[:, :, 0] / raw_granule_tensor[:, :, 0].max()
        b1 = raw_granule_tensor[:, :, 1] / raw_granule_tensor[:, :, 1].max()

    pred = matching(
        {"image0": b0.unsqueeze(0).unsqueeze(0), "image1": b1.unsqueeze(0).unsqueeze(0)}
    )
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
    matches, conf = pred["matches0"], pred["matching_scores0"]
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    if len(mkpts1) and len(mkpts0):
        shift = torch.tensor([x - y for (x, y) in zip(mkpts1, mkpts0)], device=device)
        N_matched_keypoints = shift.size(dim=0)
        print("Number of matched keypoints:", N_matched_keypoints)
        # Removing outliers
        shift_v, shift_h = shift[:, 0], shift[:, 1]
        shift_v_mean, shift_v_std = torch.mean(shift_v), torch.std(shift_v)
        shift_h_mean, shift_h_std = torch.mean(shift_h), torch.std(shift_h)
        shift_v = shift_v[
            torch.logical_and(
                shift_v > shift_v_mean - shift_v_std,
                shift_v < shift_v_mean + shift_v_std,
            )
        ]
        shift_h = shift_h[
            torch.logical_and(
                shift_h > shift_h_mean - shift_h_std,
                shift_h < shift_h_mean + shift_h_std,
            )
        ]
        shift_mean = torch.round(
            torch.tensor([shift_v.mean(), shift_h.mean()], device=device)
        )
        if save_path is not None:
            color = cm.jet(mconf)

            text = [
                "SuperGlue",
                "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
                "Matches: {}".format(len(mkpts0)),
            ]
            k_thresh = matching.superpoint.config["keypoint_threshold"]
            m_thresh = matching.superglue.config["match_threshold"]
            small_text = [
                "Keypoint Threshold: {:.4f}".format(k_thresh),
                "Match Threshold: {:.2f}".format(m_thresh),
                "Image Pair: {}:{}".format(requested_bands[0], requested_bands[1]),
            ]
            out_filename = os.path.join(
                save_path,
                event_name
                + "_H_"
                + str(int(shift_mean[0]))
                + "_V_"
                + str(int(shift_mean[1])),
            )
            make_matching_plot(
                b0 * 255,
                b1 * 255,
                kpts0,
                kpts1,
                mkpts0,
                mkpts1,
                color,
                text,
                out_filename,
                True,
                False,
                True,
                "Matches",
                small_text,
            )
    else:
        print("WARNING: no mktpoints found!")
        return [None, None], b0, b1, None
    return shift_mean, b0, b1, raw_granule.get_granule_info(), N_matched_keypoints


def bands_coregistration(b0, b1, shift_mean, device):
    """Coregister two bands by applying a shift.

    Args:
        b0 (torch.tensor): band 0.
        b1 (torch.tensor): band 1.
        shift_mean (list): [shift along, shift across].
        device (torch.device): torch.device to use.

    Returns:
        torch.tensor: tensor containing registered bands (b1_shifted repeated on the last channel).
    """
    b1_shifted = torch.zeros_like(b1)
    if (shift_mean[0] == 0) and (shift_mean[1] == 0):
        b1_shifted = b1
    elif (shift_mean[0] == 0) and (shift_mean[1] < 0):
        b1_shifted[:, : -int(shift_mean[1])] = b1[:, int(shift_mean[1]) :]
    elif (shift_mean[0] == 0) and (shift_mean[1] > 0):
        b1_shifted[:, int(shift_mean[1]) :] = b1[:, : -int(shift_mean[1])]
    elif (shift_mean[0] < 0) and (shift_mean[1] == 0):
        b1_shifted[: int(shift_mean[0]), :] = b1[-int(shift_mean[0]) :, :]
    elif (shift_mean[0] > 0) and (shift_mean[1] == 0):
        b1_shifted[-int(shift_mean[0]) :, :] = b1[0 : int(shift_mean[0]), :]
    elif (shift_mean[0] > 0) and (shift_mean[1] > 0):
        b1_shifted[int(shift_mean[0]) :, int(shift_mean[1]) :] = b1[
            : -int(shift_mean[0]), : -int(shift_mean[1])
        ]
    elif (shift_mean[0] > 0) and (shift_mean[1] < 0):
        b1_shifted[-int(shift_mean[0]) :, : -int(shift_mean[1])] = b1[
            0 : int(shift_mean[0]), int(shift_mean[1]) :
        ]
    elif (shift_mean[0] < 0) and (shift_mean[1] > 0):
        b1_shifted[: int(shift_mean[0]), int(shift_mean[1]) :] = b1[
            -int(shift_mean[0]) :, : -int(shift_mean[1])
        ]
    else:
        b1_shifted[: int(shift_mean[0]), : int(shift_mean[1])] = b1[
            -int(shift_mean[0]) :, -int(shift_mean[1]) :
        ]

    b_registered = torch.zeros(
        [b1_shifted.shape[0], b1_shifted.shape[1], 3], device=device
    )
    b_registered[:, :, 0] = b0
    b_registered[:, :, 1] = b1_shifted
    b_registered[:, :, 2] = b1_shifted
    return b_registered


def update_coregistration_analysis(csv_name, fieldnames, database):
    """Create or update the CSV containing the results of the coregistration analysis.

    Args:
        csv_name (str): csv name.
        fieldnames (list): list of CSV fields.
        database (dict): coregistration study results. Dict (field name : value)
    """
    with open(csv_name, mode="w+") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in database:
            writer.writerow(row)
