from pyraws.utils.img_processing_utils import (
    is_2D_image,
    three_channel_to_grayscale_img,
)
import numpy as np
import cv2


def get_matched_keypoints_sift(
    img_to_match: np.ndarray,
    reference_img: np.ndarray,
    perc_matched_kp: float,
    max_val_grayscale: int = 255,
):

    # Convert images to grayscale if they are not already.
    if not is_2D_image(img_to_match):
        img_to_match = three_channel_to_grayscale_img(img_to_match)
    if not is_2D_image(reference_img):
        reference_img = three_channel_to_grayscale_img(reference_img)

    sift = cv2.SIFT_create()

    img_to_match = (
        (img_to_match - np.min(img_to_match)) * max_val_grayscale / np.max(img_to_match)
    )
    reference_img = (
        (reference_img - np.min(reference_img))
        * max_val_grayscale
        / np.max(reference_img)
    )
    img_to_match = img_to_match.astype(np.uint8)
    reference_img = reference_img.astype(np.uint8)
    # Find keypoints and descriptors.
    kp1, d1 = sift.detectAndCompute(img_to_match, None)
    kp2, d2 = sift.detectAndCompute(reference_img, None)

    # Match features between the two images.
    # We create a Brute Force matcher with Hamming distance as measurement mode.
    matcher = cv2.BFMatcher()

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the top 'perc_matched_kp' % matches forward.
    matches = matches[: int(len(matches) * perc_matched_kp)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * len_coordinates.
    len_coordinates = 2
    p1 = np.zeros((no_of_matches, len_coordinates))
    p2 = np.zeros((no_of_matches, len_coordinates))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    return p1, p2
