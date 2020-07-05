import warnings
from typing import List, Optional
import pandas as pd
from libs import us_state_abbrev
from libs.datasets import dataset_utils
from libs.datasets import dataset_base
from libs.datasets import custom_aggregations
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields
from libs.datasets.dataset_utils import AggregationLevel


class TimeseriesDataset(dataset_base.DatasetBase):
    """Represents timeseries dataset.

    To make a data source compatible with the timeseries, it must have the required
    fields in the Fields class below + metrics. The other fields are generated
    in the `from_source` method.
    """

    INDEX_FIELDS = [
        CommonIndexFields.DATE,
        CommonIndexFields.AGGREGATE_LEVEL,
        CommonIndexFields.COUNTRY,
        CommonIndexFields.STATE,
        CommonIndexFields.FIPS,
    ]

    def __init__(self, data: pd.DataFrame):
        self.data = data

    @property
    def states(self) -> List:
        return self.data[CommonFields.STATE].dropna().unique().tolist()

    @property
    def state_data(self) -> pd.DataFrame:
        return self.get_subset(AggregationLevel.STATE).data

    @property
    def county_data(self) -> pd.DataFrame:
        return self.get_subset(AggregationLevel.COUNTY).data

    def county_keys(self) -> List:
        """Returns a list of all (country, state, county) combinations."""
        # Check to make sure all values are county values
        warnings.warn(
            "Tell Tom you are using this, I'm going to delete it soon.",
            DeprecationWarning,
            stacklevel=2,
        )
        county_values = self.data[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        county_data = self.data[county_values]

        data = county_data.set_index(
            [CommonFields.COUNTRY, CommonFields.STATE, CommonFields.COUNTY, CommonFields.FIPS,]
        )
        values = set(data.index.to_list())
        return sorted(values)

    def latest_values(self, aggregation_level=None) -> pd.DataFrame:
        """Gets the most recent values.

        Args:
            aggregation_level: If specified, only gets latest values for that aggregation,
                otherwise returns values for entire aggretation.

        Return: DataFrame
        """
        if not aggregation_level:
            county = self.latest_values(aggregation_level=AggregationLevel.COUNTY)
            state = self.latest_values(aggregation_level=AggregationLevel.STATE)
            return pd.concat([county, state])

        if aggregation_level == AggregationLevel.COUNTY:
            group = [CommonFields.COUNTRY, CommonFields.STATE, CommonFields.FIPS]
        if aggregation_level == AggregationLevel.STATE:
            group = [CommonFields.COUNTRY, CommonFields.STATE]
        if aggregation_level == AggregationLevel.COUNTRY:
            group = [CommonFields.COUNTRY]

        data = self.data[
            self.data[CommonFields.AGGREGATE_LEVEL] == aggregation_level.value
        ].reset_index()
        # If the groupby raises a ValueError check the dtype of date. If it was loaded
        # by read_csv did you set parse_dates=["date"]?
        return data.iloc[data.groupby(group).date.idxmax(), :]

    def get_subset(
        self,
        aggregation_level,
        country=None,
        fips: Optional[str] = None,
        state: Optional[str] = None,
        states: Optional[List[str]] = None,
        on: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> "TimeseriesDataset":
        """Fetch a new TimeseriesDataset with a subset of the data in `self`.

        Some parameters are only used in ipython notebooks."""
        row_binary_array = dataset_utils.make_binary_array(
            self.data,
            aggregation_level=aggregation_level,
            country=country,
            fips=fips,
            state=state,
            states=states,
            on=on,
            after=after,
            before=before,
        )
        return self.__class__(self.data.loc[row_binary_array, :])

    def get_records_for_fips(self, fips) -> List[dict]:
        """Get data for FIPS code.

        Args:
            fips: FIPS code.

        Returns: List of dictionary records with NA values replaced to be None
        """
        subset = self.get_subset(AggregationLevel.COUNTY, fips=fips)
        return subset.records

    def get_records_for_state(self, state) -> List[dict]:
        """Get data for state.

        Args:
            state: 2 letter state abbrev.

        Returns: List of dictionary records with NA values replaced to be None.
        """
        subset = self.get_subset(AggregationLevel.STATE, state=state)
        return subset.records

    @property
    def records(self) -> List[dict]:
        """Returns rows in current data."""
        data = self.data
        return data.where(pd.notnull(data), None).to_dict(orient="records")

    def get_data(
        self,
        aggregation_level,
        country=None,
        fips: Optional[str] = None,
        state: Optional[str] = None,
        states: Optional[List[str]] = None,
        on: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        columns_slice: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        rows_binary_array = dataset_utils.make_binary_array(
            self.data,
            aggregation_level=aggregation_level,
            country=country,
            fips=fips,
            state=state,
            states=states,
            on=on,
            after=after,
            before=before,
        )
        if columns_slice is None:
            columns_slice = slice(None, None, None)
        return self.data.loc[rows_binary_array, columns_slice]

    @classmethod
    def from_source(
        cls, source: "DataSource", fill_missing_state: bool = True
    ) -> "TimeseriesDataset":
        """Loads data from a specific datasource.

        Args:
            source: DataSource to standardize for timeseries dataset
            fill_missing_state: If True, backfills missing state data by
                calculating county level aggregates.

        Returns: Timeseries object.
        """
        data = source.data
        # TODO(tom): Do this renaming upstream, when the source is loaded or when first copied from the third party.
        to_common_fields = {value: key for key, value in source.all_fields_map().items()}
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]
        group = [
            CommonFields.DATE,
            CommonFields.COUNTRY,
            CommonFields.AGGREGATE_LEVEL,
            CommonFields.STATE,
        ]
        data = custom_aggregations.update_with_combined_new_york_counties(
            data, group, are_boroughs_zero=source.HAS_AGGREGATED_NYC_BOROUGH
        )

        if fill_missing_state:
            state_groupby_fields = [
                CommonFields.DATE,
                CommonFields.COUNTRY,
                CommonFields.STATE,
            ]
            non_matching = dataset_utils.aggregate_and_get_nonmatching(
                data, state_groupby_fields, AggregationLevel.COUNTY, AggregationLevel.STATE,
            ).reset_index()
            data = pd.concat([data, non_matching])

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_county_using_fips(data, fips_data)
        is_state = data[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.STATE.value
        state_fips = data.loc[is_state, CommonFields.STATE].map(us_state_abbrev.ABBREV_US_FIPS)
        data.loc[is_state, CommonFields.FIPS] = state_fips

        # Choosing to sort by date
        data = data.sort_values(CommonFields.DATE)
        return cls(data)

    @classmethod
    def build_from_data_source(cls, source):
        """Build TimeseriesDataset from a data source."""
        if set(source.INDEX_FIELD_MAP.keys()) != set(cls.INDEX_FIELDS):
            raise ValueError("Index fields must match")

        return cls.from_source(source, fill_missing_state=source.FILL_MISSING_STATE_LEVEL_DATA)

    def to_latest_values_dataset(self):
        from libs.datasets.latest_values_dataset import LatestValuesDataset

        return LatestValuesDataset(self.latest_values())

    def summarize(self):
        dataset_utils.summarize(
            self.data,
            AggregationLevel.COUNTY,
            [CommonFields.DATE, CommonFields.COUNTRY, CommonFields.STATE, CommonFields.FIPS,],
        )

        print()
        dataset_utils.summarize(
            self.data,
            AggregationLevel.STATE,
            [CommonFields.DATE, CommonFields.COUNTRY, CommonFields.STATE],
        )
