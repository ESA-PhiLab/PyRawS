from pyraws.utils.img_processing_utils import (
    is_2D_image,
    three_channel_to_grayscale_img,
)
import numpy as np
import cv2


def get_matched_keypoints_sift(
    img_1: np.ndarray,
    img_2: np.ndarray,
    min_matches: int = 10,
    threshold_lowe_ratio: float = 0.7,
    max_val_grayscale: int = 255,
):
    """Detect keypoints in two different images and get keypoints that match using SIFT and Lowe's ratio test.

    Args:
        img_1 (np.ndarray): first image.
        img_2 (np.ndarray): second image.
        min_matches (int, optional): Minimum number of matches to get. Defaults to 10.
        threshold_lowe_ratio (float, optional): Threshold for Lowe's ratio test. It should be a number in [0, 1]. Defaults to 0.7.
        max_val_grayscale (int, optional): Maximum image grayscale value. Defaults to 255.

    Raises:
        ValueError: There are less than 'min_matches' keypoints matches.

    Returns:
        (np.ndarray, np.ndarray): coordinates of matching keypoints.
    """

    # Convert images to grayscale if they are not already.
    if not is_2D_image(img_1):
        img_1 = three_channel_to_grayscale_img(img_1)
    if not is_2D_image(img_2):
        img_2 = three_channel_to_grayscale_img(img_2)

    sift = cv2.SIFT_create()

    img_1 = (img_1 - np.min(img_1)) * max_val_grayscale / np.max(img_1)
    img_2 = (img_2 - np.min(img_2)) * max_val_grayscale / np.max(img_2)
    img_1 = img_1.astype(np.uint8)
    img_2 = img_2.astype(np.uint8)
    # Find keypoints and descriptors.
    kp1, d1 = sift.detectAndCompute(img_1, None)
    kp2, d2 = sift.detectAndCompute(img_2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with Euclidean distance as measurement mode.
    matcher = cv2.BFMatcher()

    # Match the two sets of descriptors.
    matches = matcher.knnMatch(d1, d2, k=2)
    # Store all the good matches as per Lowe's ratio test.
    lowe_filtered_matches = []
    for m, n in matches:
        if m.distance < threshold_lowe_ratio * n.distance:
            lowe_filtered_matches.append(m)
    if len(lowe_filtered_matches) < min_matches:
        raise ValueError("Not enough matches between keypoints were found.")

    no_of_filtered_matches = len(lowe_filtered_matches)

    # Define empty matrices of shape no_of_matches * len_coordinates.
    len_coordinates = 2
    p1 = np.zeros((no_of_filtered_matches, len_coordinates))
    p2 = np.zeros((no_of_filtered_matches, len_coordinates))

    for i in range(len(lowe_filtered_matches)):
        p1[i, :] = kp1[lowe_filtered_matches[i].queryIdx].pt
        p2[i, :] = kp2[lowe_filtered_matches[i].trainIdx].pt

    return p1, p2
