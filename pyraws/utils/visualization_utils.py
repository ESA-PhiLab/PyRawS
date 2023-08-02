import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.patches as patches


def image_histogram_equalization(image, number_bins=255):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = (number_bins - 1) * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def equalize_tensor(raw_granule_tensor, n_std=2):
    """Equalizes a tensor for a better visualization by clipping outliers of a histogram higher and lower than
    pixels value mean *- n_std times the standarda deviation.

    Args:
        raw_granule_tensor (torch.tensor): tensor to equalize.
        n_std (int, optional): Number of times the standard deviation. Defaults to 2.

    Returns:
        torch.tensor: equalized tensor.
    """
    raw_granule_tensor_equalized = raw_granule_tensor.clone()
    for n in range(raw_granule_tensor.shape[2]):
        band = raw_granule_tensor_equalized[:, :, n]
        band_mean, band_std = band.mean(), band.std()
        #   Histogram clipping:
        band[band < band_mean - n_std * band_std] = band_mean - n_std * band_std
        band[band > band_mean + n_std * band_std] = band_mean + n_std * band_std

        band, cdf = image_histogram_equalization(band.numpy(), number_bins=2 ** 16)
        band = torch.from_numpy(band)
        #   band_clahe = clahe.apply((band.numpy() * CONVERSION ).astype(np.uint8))
        #   raw_granule_tensor_equalized[:,:,n]= torch.from_numpy(band_clahe/CONVERSION)
        raw_granule_tensor_equalized[:, :, n] = band

    return raw_granule_tensor_equalized


def plot_img1_vs_img2_bands(
    img1_band,
    img2_band,
    img_name_list,
    alert_matrix=None,
    alert_matrix_unregistered=None,
    save_path=None,
):
    """Util function to visualize and compare the bands of two different images. It also allows adding an alert matrix.

    Args:
        img1_band (torch.tensor): first image band.
        img2_band (torch.tensor): second image band.
        img_name_list (list): list of names of different images.
        alert_matrix (torch.tensor, opional): if not None, the hotmap of normal band is shown. Defaults to None.
        alert_matrix_unregistered (torch.tensor, opional): if not None, the hotmap of unregstered band is shown.
                                                           Defaults to None.
        save_path (string, optional): if not None, the image is saved at save_path. Defaults to None.
    """
    cmap = "bone"
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1_band.detach().cpu().numpy(), cmap=cmap)
    if alert_matrix is not None:
        ax1.contour(alert_matrix.detach().cpu().numpy(), colors="r")
    ax1.grid(False)
    ax1.axis("off")
    ax1.title.set_text(img_name_list[0])
    ax2.imshow(img2_band.detach().cpu().numpy(), cmap=cmap)
    if alert_matrix_unregistered is not None:
        ax2.contour(alert_matrix_unregistered.detach().cpu().numpy(), colors="r")
    ax2.grid(False)
    ax2.axis("off")
    ax2.title.set_text(img_name_list[1])
    fig.tight_layout()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def plot_event(img, img_name, bbox_list, alert_matrix=None, save_path=None):
    """Util function to visualize and compare the bands of two different images. It also allows adding an alert matrix.

    Args:
        img (torch.tensor): img.
        img_name (string): image_name.
        bbox_list (skimage properties): bbox list.
        alert_matrix (torch.tensor, opional): if not None, the hotmap of normal band is shown. Defaults to None.
        save_path (string, optional): if not None, the image is saved at save_path. Defaults to None.
    """
    cmap = "bone"
    fig, ax = plt.subplots()
    ax.imshow(img.detach().cpu().numpy(), cmap=cmap)
    if alert_matrix is not None:
        ax.contour(alert_matrix.detach().cpu().numpy(), colors="r")
    ax.grid(False)
    ax.axis("off")
    ax.title.set_text(img_name)
    for prop in bbox_list:
        bbox = prop.bbox  # x, y, width, height
        rect = patches.Rectangle(
            (bbox[1], bbox[0]),
            abs(bbox[1] - bbox[3]),
            abs(bbox[0] - bbox[2]),
            linewidth=2,
            edgecolor="y",
            facecolor="none",
        )
        ax.add_patch(rect)

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
