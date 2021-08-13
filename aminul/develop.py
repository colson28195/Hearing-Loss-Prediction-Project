import pandas as pd

from tyty import processing


def run_pipeline():
    data = processing.import_data("data")
    data = processing.clean_empty(data)
    data = processing.remove_columns(
        data,
        [
            "Participant ID",
            "Protocol",
            "Created",
            "Data",
            "Pressure",
            "AdultAbsorbanceData",
            "AgeCategory",
            "PressureCategory",
        ],
    )
    return data


# Explanation for removing these column:
# participant id can be deleted. it does not required to make any prediction.
# protocol colunm has no vlaue/ empty column. so it can be deleted.
# Created column is insignificant. Because it a time when the measurement is taken.
# data column have similar value in each raw. so it is not significant.
# Adult absorbance and pressure is required for appropriate raw selection . but as a column , they are not significant.
# As we are using age directly as numerical value, so AgeCategory just duplicate of that age features.
# Pressure Category only contain single char data which is "ambient ". so it is not significant.
