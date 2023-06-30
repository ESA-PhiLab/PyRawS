import torch

# -------------------------------------------------------------------------------------------------
# Database constants
# -------------------------------------------------------------------------------------------------
# Database file names
DATABASE_FILE_DICTIONARY = {
    "THRAWS": "thraws_db.csv",
    "vessel_detection": "vessel_detection_db.csv",
}

# -------------------------------------------------------------------------------------------------
# General constants
# -------------------------------------------------------------------------------------------------
# Band names
BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]
# Band resolution along-track dict
BAND_SPATIAL_RESOLUTION_DICT = {
    "B01": 60.0,
    "B02": 10.0,
    "B03": 10.0,
    "B04": 10.0,
    "B05": 20.0,
    "B06": 20.0,
    "B07": 20.0,
    "B08": 10.0,
    "B8A": 20.0,
    "B09": 60.0,
    "B10": 60.0,
    "B11": 20.0,
    "B12": 20.0,
}
# Band names physical order
BAND_NAMES_REAL_ORDER = [
    "B02",
    "B08",
    "B03",
    "B10",
    "B04",
    "B05",
    "B11",
    "B06",
    "B07",
    "B8A",
    "B12",
    "B01",
    "B09",
]  # Band names in the order they compare in the detectors.
# SWIR bands
SWIR_BANDS = ["B10", "B11", "B12"]

# -------------------------------------------------------------------------------------------------
# Raw constants
# -------------------------------------------------------------------------------------------------
# Bands shape dictionary
BANDS_RAW_SHAPE_DICT = {
    "B02": torch.Size([2304, 2592]),
    "B08": torch.Size([2304, 2592]),
    "B03": torch.Size([2304, 2592]),
    "B10": torch.Size([384, 1296]),
    "B04": torch.Size([2304, 2592]),
    "B05": torch.Size([1152, 1296]),
    "B11": torch.Size([1152, 1296]),
    "B06": torch.Size([1152, 1296]),
    "B07": torch.Size([1152, 1296]),
    "B8A": torch.Size([1152, 1296]),
    "B12": torch.Size([1152, 1296]),
    "B01": torch.Size([384, 1296]),
    "B09": torch.Size([384, 1296]),
}
# Band resolution across-track dict
BAND_SPATIAL_RESOLUTION_DICT_ACROSS = {
    "B01": 20.0,
    "B02": 10.0,
    "B03": 10.0,
    "B04": 10.0,
    "B05": 20.0,
    "B06": 20.0,
    "B07": 20.0,
    "B08": 10.0,
    "B8A": 20.0,
    "B09": 20.0,
    "B10": 20.0,
    "B11": 20.0,
    "B12": 20.0,
}

# -------------------------------------------------------------------------------------------------
# L1-c constants
# -------------------------------------------------------------------------------------------------
# Default quantification value
S2_DEFAULT_QUANTIFICATION_VALUE = (
    10000  # Default value used in case of missing metadata.
)
