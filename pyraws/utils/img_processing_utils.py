import numpy as np
import cv2


def three_channel_to_grayscale_img(img: np.ndarray) -> np.ndarray:
    """Converts a 3-channel image to grayscale (i.e. 1-channel)

    Args:
        img (np.ndarray): 3-channel image.

    Returns:
        np.ndarray: grayscale image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def save_img(img: np.ndarray, path: str):
    """Saves an image to path.

    Args:
        img (np.ndarray): image to save.
        path (str): path where to save the image.
    """
    cv2.imwrite(path, img)


def is_2D_image(img: np.ndarray) -> bool:
    """True if an image has just two dimensions. False otherwise.

    Args:
        img (np.ndarray): image.

    Returns:
        bool: True if an image has just two dimensions. False otherwise.
    """
    return img.shape == 2
