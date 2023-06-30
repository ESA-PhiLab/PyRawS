import sys
import os

sys.path.insert(1, os.path.join("..", ".."))
import numpy as np
import matplotlib.pyplot as plt


def extract_test_time(test_name):
    """Extracts test results.

    Args:
        test_name (str): test path.

    Returns:
        dict: results dictionary.
    """
    f = open(test_name, "r")
    results_txt = f.read()
    f.close()
    results = results_txt.split("\n")[3].split(" ")

    results_clean = []
    for x in results:
        if x != "":
            results_clean.append(x)

    n_images = int(test_name.split("_n_images_")[1].split("_")[0])
    coreg_type = test_name.split("_coreg_")[1].split("_")[0]
    device = test_name.split("_device_")[1].split("_")[0]
    equalize = test_name.split("_equalize_")[1].split("_")[0]
    test_iteration = test_name.split("test_iteration_")[1][:-4]

    if results_clean[-3][-2:] == "ms":
        results = float(results_clean[-3][:-2]) / 1000
    else:
        results = float(results_clean[-3][:-2])

    result_dict = {
        "n_images": n_images,
        "coreg_type": coreg_type,
        "device": device,
        "equalize": equalize,
        "result [s]": results,
    }

    return test_iteration, result_dict


def clean_coregistration_histogram(values):
    """Clean coregistration histogram by removing values outside -std and std.

    Args:
        values (numpy array): values to clean.

    Returns:
        numpy array: cleaned histogram.
    """
    values_mean = values.mean()
    vaues_std = values.std()
    values = values[values <= values_mean + 0.1 * vaues_std]
    values = values[values >= values_mean - 0.1 * vaues_std]
    return values


def generate_histograms(results_df, satellite, detector, path="histograms", bins=25):
    """Generate histogram for coregistration study results.

    Args:
        results_df (pandas df): coregistration study results as pandas dataframe.
        satellite (str): satellite.
        detector (int): detector number.
        path (str, optional): output path to save histogram. Defaults to "histograms".
        bins (int, optional): Number of histogram beams. Defaults to 25.
    """
    os.makedirs(path, exist_ok=True)
    results_df_satellite = results_df[results_df["Sat"] == satellite]
    results_satellite_detector = results_df_satellite[
        results_df_satellite["detector_number"] == detector
    ]
    results_satellite_detector_N_v = np.array(results_satellite_detector.N_v.to_list())
    results_satellite_detector_N_h = np.array(results_satellite_detector.N_h.to_list())

    plt.figure()
    plt.hist(results_satellite_detector_N_v, bins=bins)
    plt.xlabel("N_v")
    plt.ylabel("Number of occurrences")
    plt.title("Hist_" + str(satellite) + "_" + str(detector) + "_Nv")
    plt.savefig(
        os.path.join(path, "hist_" + str(satellite) + "_" + str(detector) + "_Nv.png")
    )
    plt.figure()
    plt.hist(results_satellite_detector_N_h, bins=bins)
    plt.xlabel("N_h")
    plt.ylabel("Number of occurrences")
    plt.title("Hist_" + str(satellite) + "_" + str(detector) + "_Nh")
    plt.savefig(
        os.path.join(path, "hist_" + str(satellite) + "_" + str(detector) + "_Nh.png")
    )
    return
