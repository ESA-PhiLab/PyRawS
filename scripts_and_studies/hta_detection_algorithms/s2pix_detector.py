import torch
import numpy as np

# [1] Massimetti, Francesco, et al. "Volcanic hot-spot detection using SENTINEL-2:
# a comparison with MODIS–MIROVA thermal data series." Remote Sensing 12.5 (2020): 820."


def get_thresholds(
    sentinel_img,
    alpha_thr=[1.4, 1.2, 0.15],
    beta_thr=[2, 0.5, 0.5],
    S_thr=[1.2, 1, 1.5, 1],
    gamma_thr=[1, 1, 0.5],
):
    """It returns the alpha, beta, gamma and S threshold maps for each band as described in [1]

    Args:
        sentinel_img (torch.tensor): sentinel image
        alpha_thr (list, optional): pixel-level value for calculation of alpha threshold map. Defaults to [1.4, 1.2, 0.15].
        beta_thr (list, optional): pixel-level value for calculation of beta threshold map. Defaults to [2, 0.5, 0.5].
        S_thr (list, optional): pixel-level value for calculation of S threshold map. Defaults to [1.2, 1, 1.5, 1].
        gamma_thr (list, optional): pixel-level value for calculation of gamma threshold map. Defaults to [1,1,0.5].

    Returns:
        torch.tensor: alpha threshold map.
        torch.tensor: beta threshold map.
        torch.tensor: S threshold map.
        torch.tensor: gamma threshold map.
    """

    def check_surrounded(img):
        conv = torch.nn.Conv2d(1, 1, 3)
        weight = torch.nn.Parameter(
            torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 1.0]]]]),
            requires_grad=False,
        )
        img_pad = torch.nn.functional.pad(img, (1, 1, 1, 1), mode="constant", value=1)
        conv.load_state_dict({"weight": weight, "bias": torch.zeros([1])}, strict=False)

        if sentinel_img.device.type == "cuda":
            conv = conv.cuda()

        with torch.no_grad():
            surrounded = (
                conv(
                    torch.tensor(img_pad, dtype=torch.float32, device=img_pad.device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                .squeeze(0)
                .squeeze(0)
            )
            surrounded[surrounded < 8] = 0
            surrounded[surrounded == 8] = 1
            del weight
            del conv
            del img_pad
            torch.cuda.empty_cache()
            return surrounded

    with torch.no_grad():
        alpha = torch.logical_and(
            torch.where(sentinel_img[:, :, 2] >= alpha_thr[2], 1, 0),
            torch.logical_and(
                torch.where(
                    sentinel_img[:, :, 2] / sentinel_img[:, :, 1] >= alpha_thr[0], 1, 0
                ),
                torch.where(
                    sentinel_img[:, :, 2] / sentinel_img[:, :, 0] >= alpha_thr[1], 1, 0
                ),
            ),
        )
        beta = torch.logical_and(
            torch.where(
                sentinel_img[:, :, 1] / sentinel_img[:, :, 0] >= beta_thr[0], 1, 0
            ),
            torch.logical_and(
                torch.where(sentinel_img[:, :, 1] >= beta_thr[1], 1, 0),
                torch.where(sentinel_img[:, :, 2] >= beta_thr[2], 1, 0),
            ),
        )
        S = torch.logical_or(
            torch.logical_and(
                torch.where(sentinel_img[:, :, 2] >= S_thr[0], 1, 0),
                torch.where(sentinel_img[:, :, 0] <= S_thr[1], 1, 0),
            ),
            torch.logical_and(
                torch.where(sentinel_img[:, :, 1] >= S_thr[2], 1, 0),
                torch.where(sentinel_img[:, :, 0] >= S_thr[3], 1, 0),
            ),
        )
        alpha_beta_logical_surrounded = check_surrounded(torch.logical_or(alpha, beta))
        gamma = torch.logical_and(
            torch.logical_and(
                torch.logical_and(
                    torch.where(sentinel_img[:, :, 2] >= gamma_thr[0], 1, 0),
                    torch.where(sentinel_img[:, :, 2] >= gamma_thr[1], 1, 0),
                ),
                torch.where(sentinel_img[:, :, 0] >= gamma_thr[2], 1, 0),
            ),
            alpha_beta_logical_surrounded,
        )
    return alpha, beta, S, gamma


def get_alert_matrix_and_thresholds(
    sentinel_img,
    alpha_thr=[1.4, 1.2, 0.15],
    beta_thr=[2, 0.5, 0.5],
    S_thr=[1.2, 1, 1.5, 1],
    gamma_thr=[1, 1, 0.5],
):
    """It calculates the alert-matrix for a certain image.

    Args:
        sentinel_img (torch.tensor): sentinel image
        alpha_thr (list, optional): pixel-level value for calculation of alpha threshold map. Defaults to [1.4, 1.2, 0.15].
        beta_thr (list, optional): pixel-level value for calculation of beta threshold map. Defaults to [2, 0.5, 0.5].
        S_thr (list, optional): pixel-level value for calculation of S threshold map. Defaults to [1.2, 1, 1.5, 1].
        gamma_thr (list, optional): pixel-level value for calculation of gamma threshold map. Defaults to [1,1,0.5].

    Returns:
        torch.tensor: alert_matrix threshold map.
        torch.tensor: alpha threshold map.
        torch.tensor: beta threshold map.
        torch.tensor: S threshold map.
        torch.tensor: gamma threshold map.
    """
    with torch.no_grad():
        alpha, beta, S, gamma = get_thresholds(
            sentinel_img, alpha_thr, beta_thr, S_thr, gamma_thr
        )
        alert_matrix = torch.logical_or(
            torch.logical_or(torch.logical_or(alpha, beta), gamma), S
        )
    return alert_matrix, alpha, beta, S, gamma


def cluster_9px(img):
    """It performs the convolution to detect clusters of 9 activate pixels (current pixel and 8 surrounding pixels) are at 1.

    Args:
        img (torch.tensor): input alert-matrix

    Returns:
        torch.tensor: convoluted alert-map
    """

    conv = torch.nn.Conv2d(1, 1, 3)
    weight = torch.nn.Parameter(
        torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
        requires_grad=False,
    )
    img_pad = torch.nn.functional.pad(img, (1, 1, 1, 1), mode="constant", value=1)
    if img.device.type == "cuda":
        conv = conv.cuda()
    conv.load_state_dict({"weight": weight, "bias": torch.zeros([1])}, strict=False)
    with torch.no_grad():
        surrounded = (
            conv(torch.tensor(img_pad, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
            .squeeze(0)
            .squeeze(0)
        )
        del weight
        del conv
        del img_pad
        torch.cuda.empty_cache()
    return surrounded


def s2pix_detector(
    sentinel_img,
    alpha_thr=[1.4, 1.2, 0.15],
    beta_thr=[2, 0.5, 0.5],
    S_thr=[1.2, 1, 1.5, 1],
    gamma_thr=[1, 1, 0.5],
):
    """Implements first step of the one described in [1] by proving a filtered alert-map.

    Args:
        sentinel_img (torch.tensor): sentinel image
        alpha_thr (list, optional): pixel-level value for calculation of alpha threshold map. Defaults to [1.4, 1.2, 0.15].
        beta_thr (list, optional): pixel-level value for calculation of beta threshold map. Defaults to [2, 0.5, 0.5].
        S_thr (list, optional): pixel-level value for calculation of S threshold map. Defaults to [1.2, 1, 1.5, 1].
        gamma_thr (list, optional): pixel-level value for calculation of gamma threshold map. Defaults to [1,1,0.5].

    Returns:
        torch.tensor: binary classification. It is if at least a cluster of 9 hot pixels is found.
        torch.tensor: filtered alert_matrix threshold map.
        torch.tensor: alert_matrix threshold map.
    """
    with torch.no_grad():
        alert_matrix, _, _, _, _ = get_alert_matrix_and_thresholds(
            sentinel_img, alpha_thr, beta_thr, S_thr, gamma_thr
        )
        filtered_alert_matrix = cluster_9px(alert_matrix)
        filtered_alert_matrix[filtered_alert_matrix < 9] = 0
        filtered_alert_matrix[filtered_alert_matrix == 9] = 1
        return (
            torch.tensor(
                float(torch.any(filtered_alert_matrix != 0)),
                device=filtered_alert_matrix.device,
            ),
            filtered_alert_matrix,
            alert_matrix,
        )


def filter_bbox_list(
    alert_matrix, props, event_bbox_coordinates_list=None, num_pixels_threshold=9
):
    """Filters bounding box lists found in an alert matrix by takking only the bounding boxes having at least
      ""num_pixels_threshold"" active pixels.

    Args:
        alert_matrix (torch.tensor): alert matrix
        props (list): bounding box list.
        event_bbox_coordinates_list (list, optional): bounding box coordinates. Defaults to None.
        num_pixels_threshold (int,9): number of active pixels in a bounding box. Defaults to 9.

    Returns:
        list: filtered bounding box list.
        list: filtered coordinates list.
    """
    bbox_filtered_list = []
    event_bbox_coordinates_filtered_list = []
    if event_bbox_coordinates_list is not None:
        for prop, coords in zip(props, event_bbox_coordinates_list):
            bbox = prop.bbox
            bbox_rounded = [int(np.round(x)) for x in bbox]
            if (
                torch.sum(
                    alert_matrix[
                        bbox_rounded[0] : bbox_rounded[2] + 1,
                        bbox_rounded[1] : bbox_rounded[3] + 1,
                    ]
                )
                >= num_pixels_threshold
            ):
                bbox_filtered_list.append(prop)
                event_bbox_coordinates_filtered_list.append(coords)
    else:
        for prop in props:
            bbox = prop.bbox
            bbox_rounded = [int(np.round(x)) for x in bbox]
            if (
                torch.sum(
                    alert_matrix[
                        bbox_rounded[0] : bbox_rounded[2] + 1,
                        bbox_rounded[1] : bbox_rounded[3] + 1,
                    ]
                )
                >= num_pixels_threshold
            ):
                bbox_filtered_list.append(prop)

    return bbox_filtered_list, event_bbox_coordinates_filtered_list
