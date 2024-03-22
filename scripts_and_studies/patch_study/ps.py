import pandas as pd
import cv2
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np
import imageio
from matplotlib.path import Path
import os
from IPython.display import clear_output
import argparse
from ast import literal_eval


parser = argparse.ArgumentParser(
    description="Create a gif with the windows by sliding the window on the granule."
)
# add arguments to the parser
parser.add_argument("--event", type=str, help="Event to parse")

parser.add_argument(
    "--img_size",
    type=str,
    default="(1400, 1400)",
    help="Size of the image in pixels, in the format (height, width)",
)
parser.add_argument(
    "--overlap",
    type=float,
    default=0.25,
    help="The proportion of overlap between windows",
)
parser.add_argument(
    "--windowSize",
    type=str,
    default="(256, 256)",
    help="Size of the window in pixels, in the format (height, width)",
)
parser.add_argument("--plot", type=str, default="True", help="Plot or not windows")

args = parser.parse_args()

event_to_parse = args.event
image_size = literal_eval(args.img_size)  # (1400, 1400)
overlap = args.overlap  # 0.25
windowSize = literal_eval(args.windowSize)  # (256, 256)
plotter = literal_eval(args.plot)

print("Plotter set to:", plotter)


df = pd.read_csv("thraws_db.csv", index_col=0)
columns = (
    df.columns.tolist()
)  # ['ID_event', 'Start', 'End', 'Sat', 'Coords (Lat, Lon)', 'class', 'Raw_useful_granules', 'Raw_complementary_granules', 'Polygon_square', 'Raw_files', 'l1c_files', 'bbox_list', 'bbox_list_merged', 'Polygon']
# create dict with ID_evnet as key and bbox_list_merged as values:
bbox_dict = dict(zip(df.ID_event, df.bbox_list_merged))
print("Bbox_dict created", len(bbox_dict))
# create events list
events = df.ID_event.tolist()


def sliding_window(image_size: tuple, overlap: float, windowSize: tuple):
    """
    Slide a window across the image.

    Parameters
    ----------
    image_size : tuple
        Size of the image in pixels, in the format (height, width)
    overlap : float
        The proportion of overlap between windows
    windowSize : tuple
        Size of the window in pixels, in the format (height, width)

    Yields
    ------
    tuple
        The coordinates of the current window, in the format (x1, y1, x2, y2)

    """
    # Calculate the number of steps needed in each dimension
    num_steps_y = int(
        np.ceil((image_size[0] - windowSize[0]) / (windowSize[0] * (1 - overlap))) + 1
    )
    num_steps_x = int(
        np.ceil((image_size[1] - windowSize[1]) / (windowSize[1] * (1 - overlap))) + 1
    )

    # Generate the starting coordinates for the y-axis and x-axis
    y_steps = np.linspace(0, image_size[0] - windowSize[0], num_steps_y).astype(int)
    x_steps = np.linspace(0, image_size[1] - windowSize[1], num_steps_x).astype(int)

    for y in y_steps:
        for x in x_steps:
            # yield the current window, just the coords:
            yield (x, y, x + windowSize[1], y + windowSize[0])


def pointsReformat(points):
    """
    Reformat the coordinates of a window from the format (x1, y1, x2, y2) to the format point1, point2, point3, point4.

    Parameters
    ----------
    points : tuple
        The coordinates of the current window, in the format (x1, y1, x2, y2)

    Returns
    -------
    tuple
        The coordinates of the current window reformatted as point1, point2, point3, point4.
    """
    (x1, y1, x2, y2) = points
    point1 = (x1, y1)
    point2 = (x2, y1)
    point3 = (x2, y2)
    point4 = (x1, y2)
    return (point1, point2, point3, point4)


def fill_polygon_in_granule(granule, polygon):
    """
    Fill a polygon in a granule.

    Parameters
    ----------
    granule : 2D numpy.ndarray
        A granule to fill.
    polygon : list of tuples
        A polygon to fill.

    Returns
    -------
    2D numpy.ndarray
        The granule filled with the polygon.
    """
    path = Path(polygon)
    x, y = np.meshgrid(np.arange(granule.shape[1]), np.arange(granule.shape[0]))
    co_ordinates = np.column_stack((x.ravel(), y.ravel()))
    mask = path.contains_points(co_ordinates).reshape(granule.shape)
    granule[mask] = 1
    return granule


# plot the points reformatted on a squadre of size granule:
def plot_points(
    points,
    granule_size=(1661, 2585),
    idx=0,
    count=0,
    total_windows=240,
    boxes=None,
    plot=True,
):
    """
    Plot the points of the window reformatted on a squadre of size granule.
    :param points: tuple
        The coordinates of the current window, in the format (x1, y1, x2, y2)
    :param granule_size: tuple
        Size of the granule in pixels, in the format (height, width)
    :return: count
        The number of events in the granule

    """

    (point1, point2, point3, point4) = points
    # display a square of size granule_size:
    granule = np.zeros((granule_size[0], granule_size[1]))

    if boxes is not None:
        for box in boxes:
            # Define coordinates of the polygon
            polygon = np.array(box)
            # Fill the polygon in the granule
            granule = fill_polygon_in_granule(granule, polygon)

    # given the coordinates of the current window, in the format (x1, y1, x2, y2), take the window on granule:
    granule_slided_window = granule[point1[1] : point3[1], point1[0] : point3[0]]
    # prompt the number of 1 pixel of granule_slided_window:
    area = np.count_nonzero(granule_slided_window)

    if area > 5:
        count += 1

    if plot:
        # plot the granule:
        plt.figure(figsize=(10, 10), dpi=300)
        plt.imshow(granule, cmap="gray")
        # plot the points:
        plt.plot(
            [point1[0], point2[0], point3[0], point4[0], point1[0]],
            [point1[1], point2[1], point3[1], point4[1], point1[1]],
        )
        # add title on right of the image, inside the image:
        plt.title(
            f"Window #:{idx+1}/{total_windows}, N.NotEvent:{total_windows-count}, N.Events:{count}"
        )

        # set visualisation at granule size area -100 pixels on each side:
        plt.xlim(-100, granule_size[1] + 100)
        plt.ylim(granule_size[0] + 100, -100)

        # save the figure:
        plt.savefig("frames/window_{}.png".format(idx))
        # Close the figure to free up memory
        plt.close()
    return count


def get_granule_boxes(id_event, bbox_dict):
    """
    Given a granule id, returns the bounding boxes of the events
    in the granule.
    :param id_event: int
        id of the granule
    :param bbox_dict: dict
        dictionary of granule ids and event bounding boxes
    :return: list
        list of bounding boxes
    """
    return literal_eval(bbox_dict[id_event])


def parse_event(id_event, bbox_dict, plot=True):
    """
    Parse an event given its id and a dictionary of bounding boxes.

    Parameters
    ----------
    id_event : str
        The id of the event to parse.
    bbox_dict : dict
        A dictionary of bounding boxes.

    Returns
    -------
    result_dic : dict
        A dictionary of granule ids and the number of events in each granule.
    """
    result_dic = {}

    granule_boxes = get_granule_boxes(id_event, bbox_dict)
    granule_idxs = list(granule_boxes.keys())
    print("Granule idxs:", granule_idxs)
    for g_idx in granule_idxs:
        bboxes = granule_boxes[g_idx]

        count = 0
        # get the points of the first window:
        for idx in range(len(windows)):
            points = windows[idx]
            window_reformatted = pointsReformat(points)
            # plot the points:
            try:
                count = plot_points(
                    points=window_reformatted,
                    granule_size=image_size,
                    idx=idx,
                    count=count,
                    total_windows=len(windows),
                    boxes=bboxes,
                    plot=plot,
                )
            except ValueError:
                count = plot_points(
                    points=window_reformatted,
                    granule_size=image_size,
                    idx=idx,
                    count=count,
                    total_windows=len(windows),
                    boxes=bboxes[0],
                    plot=plot,
                )

            result_dic[g_idx] = count
        ### # create a gif with the windows by sliding the window on the granule:
        # Create a GIF using the frames
        if plot:
            filenames = [f"frames/window_{x}.png" for x in range(len(windows))]
            with imageio.get_writer(
                f"gif/{id_event}_g{g_idx}_count_{count}.gif", mode="I", duration=1500
            ) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            # Remove files:
            for filename in set(filenames):
                os.remove(filename)

    return result_dic


# get all the windows:
windows = list(sliding_window(image_size, overlap, windowSize))
print("Number of windows per granule:", len(windows))
N_windows = len(windows)


start_event = event_to_parse
start_index = events.index(start_event)

# create a folder in study_analysis folder with the name of the configuration:
config_name = f"img_size_{image_size}_overlap_{overlap}_windowSize_{windowSize}"
os.makedirs(f"study_analysis/{config_name}", exist_ok=True)


for idx, event in enumerate(events[start_index : start_index + 1]):
    print(f"Event {start_index}/{len(events)}:", event)
    # if NE in event skip:
    if "NE" in event:
        result_dic = {0: 0}
        continue
    try:
        result_dic = parse_event(event, bbox_dict, plot=plotter)
        clear_output(wait=True)
        print("result_dic:", result_dic)
    except:
        print("Error with event:", event)
        break

    # save the result_dic in a csv file:
    # the key is the granule idx while the value is the number of events in the granule, add a columns with the notevents = (len(windows)-events of the granule idx):
    df = pd.DataFrame.from_dict(result_dic, orient="index", columns=["events"])
    df["not_events"] = N_windows - df["events"]
    # add column event name:
    df["event"] = event
    # add column granule idx:
    df["granule_idx"] = df.index
    # save the df in a csv file:
    df.to_csv(f"study_analysis/{config_name}/{event}_count.csv")
