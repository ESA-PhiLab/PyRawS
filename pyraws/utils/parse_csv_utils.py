import csv


def parse_csv(csv_name):
    """Parse a CSV file and return a list of rows.


    :csv_name: CSV name.
    :return: (list of rows. Each row is a dictionary.)
    """
    row_list = []
    try:
        with open(csv_name) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                row_list.append(row)
    except:  # noqa: E722
        return ValueError("Impossible to parse CSV file: ", csv_name)

    return row_list
