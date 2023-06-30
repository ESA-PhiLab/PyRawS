import torch
from termcolor import colored
from torch.nn import Upsample
from torch.nn.functional import interpolate
from .constants import BAND_NAMES, BAND_SPATIAL_RESOLUTION_DICT


def image_band_upsample(img_band, band_name, target_spatial_resolution, upsample_mode):
    """Upsample an image band to a target spatial resolution through an upsample mode.

    Args:
        img_band (torch.tensor): image band.
        band_name (string): band name.
        target_spatial_resolution (float): target resolution (m).
        upsample_mode (string): band name.

    Raises:
        ValueError: unsupported band name.
        ValueError: unsupported upsample mode.

    Returns:
        torch.tensor: upsampled band.
    """

    if not (band_name in BAND_NAMES):
        raise ValueError("Unsupported band name: " + colored(band_name, "red") + ".")

    # print("Upsampling band: "+colored(band_name, "blue")+".")
    upsample_factor = (
        BAND_SPATIAL_RESOLUTION_DICT[band_name] / target_spatial_resolution
    )
    if not (upsample_mode in ["nearest", "bilinear", "bicubic"]):
        raise ValueError(
            "Upsample mode "
            + colored(upsample_mode, "blue")
            + " not supported. Please, choose among: "
            "nearest"
            ", "
            "bilinear"
            ", "
            "bicubic"
            "."
        )

    if upsample_factor <= 1.0:
        print(
            colored("Warnings", "red")
            + ". The requested target resolution ("
            + colored(target_spatial_resolution, "blue")
            + ") is lower or equal to the orginal band resolution ("
            + colored(band_name, "red")
            + ","
            + colored(BAND_SPATIAL_RESOLUTION_DICT[band_name], "green")
            + ")."
        )
        return img_band

    if upsample_factor != int(upsample_factor):
        print(
            colored("Warnings", "red")
            + ". Upsample factor truncanted from "
            + upsample_factor
            + " to "
            + int(upsample_factor)
            + "."
        )
    # else:
    # print("Upsample factor: "+colored(upsample_factor, "blue")+".")
    upsample_factor = int(upsample_factor)
    upsample_method = Upsample(
        scale_factor=upsample_factor, mode=upsample_mode, align_corners=True
    )
    with torch.no_grad():
        return upsample_method(img_band.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


def image_band_resize(
    img_upsample_band, band_name, upsampled_img_spatial_resolution, interpolate_mode
):
    """Resize an upsampled image band to a the orginal spatial resolution through an interpolate mode.

    Args:
        img_upsample_band (torch.tensor): upsampled image to resize
        band_name (string): band name
        upsampled_img_spatial_resolution (float): spatial resolution of the input upsampled image.
        interpolate_mode (string): interpolated mode.

    Raises:
        ValueError: unsupported band name.
        ValueError: unsupported interpolated mode.

    Returns:
        torch.tensor: resized image.
    """
    if not (band_name in BAND_NAMES):
        raise ValueError("Unsupported band name: " + colored(band_name, "red") + ".")

    # print("Downsampling band: "+colored(band_name, "blue")+".")
    downsample_factor = (
        BAND_SPATIAL_RESOLUTION_DICT[band_name] / upsampled_img_spatial_resolution
    )

    if not (interpolate_mode in ["nearest", "bilinear", "bicubic"]):
        raise ValueError(
            "Interpolate mode "
            + colored(interpolate_mode, "blue")
            + " not supported. Please, choose among: "
            "nearest"
            ", "
            "bilinear"
            ", "
            "bicubic"
            "."
        )

    if downsample_factor <= 1.0:
        print(
            colored("Warnings", "red")
            + ". The upsampled image resolution ("
            + colored(upsampled_img_spatial_resolution, "blue")
            + ") is lower or equal to the orginal band resolution ("
            + colored(band_name, "red")
            + ","
            + colored(BAND_SPATIAL_RESOLUTION_DICT[band_name], "green")
            + ")."
        )
        return img_upsample_band

    if downsample_factor != int(downsample_factor):
        print(
            colored("Warnings", "red")
            + ". Upsample factor truncanted from "
            + downsample_factor
            + " to "
            + int(downsample_factor)
            + "."
        )
    # else:
    # print("Downsample factor: "+colored(downsample_factor, "blue")+".")
    downsample_factor = int(downsample_factor)

    size = (
        int(img_upsample_band.shape[0] / downsample_factor),
        int(img_upsample_band.shape[1] / downsample_factor),
    )
    with torch.no_grad():
        return (
            interpolate(
                img_upsample_band.unsqueeze(0).unsqueeze(0),
                size=size,
                mode=interpolate_mode,
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )
