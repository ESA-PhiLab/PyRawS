import sys
import os

sys.path.insert(1, os.path.join("..", ".."))
sys.path.insert(
    1, os.path.join("..", "..", "scripts_and_studies", "hta_detection_algorithms")
)
from pyraws.raw.raw_event import Raw_event
from shapely.geometry import Polygon
import torch
import matplotlib.pyplot as plt

# CONFIG_
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

plt.rcParams["figure.figsize"] = [10, 10]


def mulitBox(bbox_list):
    if any(isinstance(i, list) for i in bbox_list):
        return True

    return False


def sliding_window(image, window_size, overlap):
    """
    Generate sub-images of the input image using a sliding window.
    Parameters:
         - image (2D numpy array): The input image.
         - window_size (int): The size (in pixels) of the sliding window.
         - overlap (float): The degree of overlap between two consecutive window movements.

    Returns:
         - sub_images (dict): A dictionary with the top-left and bottom-right corner positions (x1, y1, x2, y2)
                              as keys and the sub-images as values.
    """
    # Get the dimensions of the input image
    h, w = image.shape[:2]

    # Calculate the step size (in pixels) for the sliding window
    step_size = int(window_size * (1 - overlap))

    # Initialize the dictionary of sub-images
    sub_images = {}

    # Slide the window across the image
    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            # Extract the sub-image
            sub_image = image[y : y + window_size, x : x + window_size]
            klam = window_size - sub_image.shape[1]
            teta = window_size - sub_image.shape[0]
            sub_image = image[
                y - teta : y - teta + window_size, x - klam : x - klam + window_size
            ]

            # Add the sub-image to the dictionary with the positions as the key
            sub_images[
                (x - klam, y - teta, x - klam + window_size, y - teta + window_size)
            ] = sub_image
    return sub_images


# Event config:
def find_granule_dim(event_name: str, key_granule):
    requested_bands = ["B04", "B8A", "B11", "B12"]
    raw_event = Raw_event(device=device)
    raw_event.from_database(event_name, requested_bands)

    raw_granule_registered = raw_event.coarse_coregistration(
        [key_granule],
        use_complementary_granules=True,
        crop_empty_pixels=True,
        verbose=False,
    )
    raw_granule_tensor = raw_granule_registered.as_tensor()[:, :, :]
    return [raw_granule_tensor.size(0), raw_granule_tensor.size(1)]


def pos2Shap(pos, offY):
    x1, y1, x2, y2 = pos
    # Only for calculating the polygon and the intersection.
    ShapP1 = (x1, offY - y1)
    ShapP2 = (x1, offY - y2)
    ShapP3 = (x2, offY - y2)
    ShapP4 = (x2, offY - y1)
    ShapP5 = ShapP1

    Shapatch = Polygon([ShapP1, ShapP2, ShapP3, ShapP4, ShapP5]).convex_hull
    return Shapatch


def main():
    pass


if __name__ == "__main__":
    main()
