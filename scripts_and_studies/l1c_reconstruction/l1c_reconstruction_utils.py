import torch
import cv2
import numpy as np
from glob import glob
import os


def get_intersection_points(l1c_tif):
    """Gets interesection of l1c tif image with image border.

    Args:
        l1c_tif (torch.tensor): input l1c tensor.

    Returns:
        list: list of interesection points
    """

    for k in range(l1c_tif.shape[1]):
        y = l1c_tif[:, k, 0]
        dy = torch.where(y != 0)[0]
        if len(dy) != 0:
            dy = dy[0]
            break

    for m in range(l1c_tif.shape[0]):
        x = l1c_tif[m, :, 0]
        dx = torch.where(x != 0)[0]
        if len(dx) != 0:
            dx = dx[0]
            break

    for s in range(l1c_tif.shape[1] - 1, 0, -1):
        y_end = l1c_tif[:, s, 0]
        dy_end = torch.where(y_end != 0)[-1]

        if len(dy_end) != 0:
            dy_end = dy_end[-1]
            break

    for q in range(l1c_tif.shape[0] - 1, 0, -1):
        x_end = l1c_tif[q, :, 0]
        dx_end = torch.where(x_end != 0)[0]
        if len(dx_end) != 0:
            dx_end = dx_end[0]
            break

    return [
        torch.tensor([dy, k], device=l1c_tif.device),
        torch.tensor([m, dx], device=l1c_tif.device),
        torch.tensor([dy_end, s], device=l1c_tif.device),
        torch.tensor([q, dx_end], device=l1c_tif.device),
    ]


def rotate_l1c_tif(l1c_tif):
    """Rotate l1c_tif

    Args:
        l1c_tif (torch.tensor): rotate l1c_tif

    Returns:
        torch.tensor: rotated matrix
    """

    l1c_numpy = l1c_tif.detach().numpy()
    l1c_numpy = np.round(l1c_numpy / l1c_numpy.max() * 255).astype(np.uint8)
    a, b, _, d = get_intersection_points(l1c_tif)
    angle_1 = float(torch.atan2(d[0] - a[0], d[1] - a[1]) * 180 / np.pi)

    M = cv2.getRotationMatrix2D((int(d[1]), int(d[0])), angle_1, 1.0)
    rotated = cv2.warpAffine(
        l1c_tif.detach().numpy(),
        M,
        (int(d[1]), int(d[0])),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    rotated_torch = torch.from_numpy(rotated).to(l1c_tif.device)
    non_zero_last = rotated_torch[:, -1, 0].nonzero()[0]

    return rotated_torch[non_zero_last:, :, :]


def get_l1c_file_names(path):
    """Get l1c tif file list in path.

    Args:
        path (str): path to l1c_cropped_tif.

    Returns:
        list: list of l1c files.
    """
    l1c_tif_files = glob(os.path.join(path, "*"))
    l1c_tif_files_clean = []
    for x in l1c_tif_files:
        if ".tif" in x:
            l1c_tif_files_clean.append(x)
    return l1c_tif_files_clean


def get_event_granules_dict(l1c_tif_files):
    """Get events tif list in path.

    Args:
        l1c_tif_files (str): l1c cropped tif list.

    Returns:
        dict: event : granules dictionary..
    """
    events_granule_list = [
        x.split(os.path.sep)[-1].split(".tif")[0] for x in l1c_tif_files
    ]

    events_list = []
    granule_per_event_list = []
    for elem in events_granule_list:
        event_elems = elem.split("_")
        event = event_elems[0] + "_" + event_elems[1]
        granule = event_elems[2]
        if event in events_list:
            granule_per_event_list[events_list.index(event)] += [granule]
        else:
            events_list.append(event)
            granule_per_event_list.append([granule])

    event_granules_dict = dict(zip(events_list, granule_per_event_list))
    return event_granules_dict
