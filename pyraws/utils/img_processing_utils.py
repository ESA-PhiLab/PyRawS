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


def normalize_img(
    img: np.ndarray,
    min: float,
    max: float,
    a: float = 0,
    b: float = 1,
    tol: float = 1e-6,
) -> np.ndarray:
    """Normalizes an image from range [min, max] to range [a, b].

    Args:
        img (np.ndarray): image to normalize.
        min (float): minimum of the previous range.
        max (float): maximum of the previous range.
        a (float, optional): minimum of the new range. Defaults to 0.
        b (float, optional): maximum of the new range. Defaults to 1.
        tol (float, optional): tolerance. Defaults to 1e-6.

    Returns:
        np.ndarray: normalized image.
    """
    assert max != min
    img_norm = ((b - a) * ((img - min) / (max - min))) + a
    assert np.count_nonzero(img_norm < a - tol) <= 0
    assert np.count_nonzero(img_norm > b + tol) <= 0
    return img_norm
