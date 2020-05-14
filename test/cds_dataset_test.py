from io import StringIO
import pandas as pd
import numpy as np
import pytest

from libs.datasets.sources import cds_dataset
from test.dataset_utils_test import to_dict

# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


def test_remove_duplicate_city_data():
    input_df = pd.read_csv(
        StringIO(
            "city,county,state,fips,date,metric_a\n"
            "Smithville,,ZZ,97123,2020-03-23,march23-removed\n"
            "Smithville,,ZZ,97123,2020-03-22,march22-kept\n"
            "New York City,,ZZ,97324,2020-03-22,march22-ny-patched\n"
            ",North County,ZZ,97001,2020-03-22,county-not-touched\n"
            ",North County,ZZ,97001,2020-03-23,county-not-touched\n"
        )
    )

    output_df = cds_dataset.CDSDataset.remove_duplicate_city_data(input_df)
    expected_df = pd.read_csv(
        StringIO(
            "city,county,state,fips,date,metric_a\n"
            "Smithville,Smithville,ZZ,97123,2020-03-22,march22-kept\n"
            "New York City,New York,ZZ,97324,2020-03-22,march22-ny-patched\n"
            ",North County,ZZ,97001,2020-03-22,county-not-touched\n"
            ",North County,ZZ,97001,2020-03-23,county-not-touched\n"
        )
    )

    assert to_dict(["fips", "date"], output_df) == to_dict(["fips", "date"], expected_df)
