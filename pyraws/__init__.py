import logging

from .raw.raw_granule import Raw_granule
from .raw.raw_event import Raw_event
from .l1.l1_tile import L1C_tile
from .l1.l1_event import L1C_event
from .utils.constants import (
    BAND_NAMES,
    BANDS_RAW_SHAPE_DICT,
    BAND_NAMES_REAL_ORDER,
    BAND_SPATIAL_RESOLUTION_DICT,
    BAND_SPATIAL_RESOLUTION_DICT_ACROSS,
    DATABASE_FILE_DICTIONARY,
)
from .utils.date_utils import get_timestamp, parse_string_date
from .utils.database_utils import (
    get_event_info,
    get_events_list,
    get_event_granule_bb_dict,
)
from .utils.shape_utils import get_granules_polys, create_rectangular_polygon
from .utils.visualization_utils import (
    equalize_tensor,
    plot_event,
    plot_img1_vs_img2_bands,
)
from .utils.parse_csv_utils import parse_csv

__all__ = [
    "BAND_NAMES",
    "BAND_NAMES_REAL_ORDER",
    "BANDS_RAW_SHAPE_DICT",
    "BAND_SPATIAL_RESOLUTION_DICT",
    "BAND_SPATIAL_RESOLUTION_DICT_ACROSS",
    "create_rectangular_polygon",
    "DATABASE_FILE_DICTIONARY",
    "equalize_tensor",
    "get_event_bounding_box",
    "get_event_granule_bb_dict",
    "get_events_list",
    "get_event_info",
    "get_l1C_image_default_path",
    "get_granules_polys",
    "get_timestamp",
    "L1C_event",
    "L1C_tile",
    "parse_csv",
    "parse_string_date",
    "plot_event",
    "plot_img1_vs_img2_bands",
    "parse_string_date",
    "Raw_granule",
    "Raw_event",
    "read_L1C_event",
    "read_L1C_image_from_tif",
]

# Initialize logger
logger = logging.getLogger(__name__)


def set_log_level(level=logging.WARN):
    """Allows setting global log level for the application.

    Args:
        level (logging.level, optional): Level to set, available are
                                        (logging.DEBUG,logging.INFO,logging.WARN,logging.ERROR).
                                        Defaults to logging.WARN.
    """
    logger.setLevel(level)
    if level == 10:
        logger.info("Log level set to debug")
    elif level == 20:
        logger.info("Log level set to info")
    # we still store it in case we might write some logfile or sth later
    elif level == 30:
        logger.info("Log level set to warn")
    elif level == 40:
        logger.info("Log level set to error")
