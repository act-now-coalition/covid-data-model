from typing import List, Optional, Union, TextIO
import pathlib

from more_itertools import first

from libs import us_state_abbrev
import pandas as pd
from libs.datasets.dataset_utils import AggregationLevel, make_rows_key
from libs.datasets import dataset_utils
from libs.datasets import dataset_base
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields
from libs.datasets.dataset_utils import DatasetType


class LatestValuesDataset(dataset_base.DatasetBase):

    INDEX_FIELDS = [
        CommonIndexFields.AGGREGATE_LEVEL,
        CommonIndexFields.COUNTRY,
        CommonIndexFields.STATE,
        CommonIndexFields.FIPS,
    ]
    STATE_GROUP_KEY = [
        CommonIndexFields.AGGREGATE_LEVEL,
        CommonIndexFields.COUNTRY,
        CommonIndexFields.STATE,
    ]
    COMMON_INDEX_FIELDS = [CommonFields.FIPS]
