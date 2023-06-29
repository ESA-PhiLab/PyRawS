from datetime import datetime


def get_timestamp():
    """Return current timestamp.

    Returns:
        str: timestamp
    """
    timestamp = datetime.now()
    return (
        str(timestamp.year)
        + "_"
        + str(timestamp.month)
        + "_"
        + str(timestamp.day)
        + "_"
        + str(timestamp.hour)
        + "_"
        + str(timestamp.minute)
        + "_"
        + str(timestamp.second)
    )


def parse_string_date(date_string):
    """Parse a date in string format and returns a datetime.

    Args:
        date_string (string): date string

    Returns:
        datetime: correspondent datetime.
    """
    year = date_string[:4]
    month = date_string[4:6]
    day = date_string[6:8]
    hour = date_string[9:11]
    minute = date_string[11:13]
    second = date_string[13:]
    return datetime(
        int(year), int(month), int(day), int(hour), int(minute), int(second)
    )
