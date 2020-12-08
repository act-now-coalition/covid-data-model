from libs.datasets import dataset_base
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields


class LatestValuesDataset(dataset_base.DatasetBase):

    INDEX_FIELDS = [
        CommonIndexFields.AGGREGATE_LEVEL,
        CommonIndexFields.COUNTRY,
        CommonIndexFields.STATE,
        CommonIndexFields.FIPS,
    ]
    COMMON_INDEX_FIELDS = [CommonFields.FIPS]
