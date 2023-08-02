import sys
import os

sys.path.insert(1, os.path.join("..", ".."))
from pyraws.raw.raw_event import Raw_event
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bands",
        type=str,
        help='bands in format ""[Bxx,Byy,...,Bzz]"" ',
        default="[B04, B8A, B11]",
    )
    parser.add_argument(
        "--granule_index", type=int, help="Event name to start.", default=2
    )
    parser.add_argument(
        "--event_name",
        type=str,
        help="If specified, only the events of that type will be used.",
        default="Etna_00",
    )

    # Bands to open.
    pargs = parser.parse_args()
    requested_bands_str = pargs.bands
    requested_bands_str = requested_bands_str.replace(" ", "")[1:-1]
    bands_list = [x for x in requested_bands_str.split(",")]

    # Instantiate an empty Raw_event
    raw_event = Raw_event()

    # Read "Etna_00"  from THRAWS
    raw_event.from_database(  # Database ID_EVENT
        id_event=pargs.event_name,
        # Bands to open. Leave to None to use all the bands.
        bands_list=bands_list,
        # If True, verbose mode is on.
        verbose=False,
        # Database name
        database="THRAWS",
    )

    raw_coregistered_granule_1 = (
        raw_event.coarse_coregistration(  # granule index to coregister.
            granules_idx=[pargs.granule_index]
        )
    )

    # Perform the corase coregistration of the "Etna_00" event.
    # Missing pixels will be cropped.
    raw_coregistered_granule_1_with_crop = raw_event.coarse_coregistration(  # granule index to coregister.
        granules_idx=[pargs.granule_index],
        # Cropping missing pixels.
        crop_empty_pixels=True,
    )

    # Perform the corase coregistration of the "Etna_00" event.
    # Missing pixels will be cropped.
    raw_coregistered_granule_1_with_fill = raw_event.coarse_coregistration(  # granule index to coregister.
        granules_idx=[pargs.granule_index],
        # Search for filling elements among adjacent L0 granules
        use_complementary_granules=True,
        # Cropping missing pixels when compatible L0 granules are not available
        crop_empty_pixels=True,
    )

    raw_granule_1 = raw_event.get_granule(pargs.granule_index)
    raw_granule_1_t = raw_granule_1.as_tensor()
    raw_coregistered_granule_1_t = raw_coregistered_granule_1.as_tensor()
    raw_coregistered_granule_1_with_crop_t = (
        raw_coregistered_granule_1_with_crop.as_tensor()
    )
    raw_coregistered_granule_1_with_fill_t = (
        raw_coregistered_granule_1_with_fill.as_tensor()
    )

    raw_granule_1_t = raw_granule_1_t / raw_granule_1_t.max()
    raw_coregistered_granule_1_t = (
        raw_coregistered_granule_1_t / raw_coregistered_granule_1_t.max()
    )
    raw_coregistered_granule_1_with_crop_t = (
        raw_coregistered_granule_1_with_crop_t
        / raw_coregistered_granule_1_with_crop_t.max()
    )
    raw_coregistered_granule_1_with_fill_t = (
        raw_coregistered_granule_1_with_fill_t
        / raw_coregistered_granule_1_with_fill_t.max()
    )

    plt.rcParams["figure.figsize"] = [10, 10]
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(raw_granule_1_t)
    ax[0, 0].set_title("Original")
    ax[0, 1].imshow(raw_coregistered_granule_1_t)
    ax[0, 1].set_title("Coregistration")
    ax[1, 0].imshow(raw_coregistered_granule_1_with_crop_t)
    ax[1, 0].set_title("Coregistration with crop")
    ax[1, 1].imshow(raw_coregistered_granule_1_with_fill_t)
    ax[1, 1].set_title("Coregistration with fill")

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()


if __name__ == "__main__":
    main()
