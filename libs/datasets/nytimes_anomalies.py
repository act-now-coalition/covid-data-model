import enum
from functools import lru_cache
from typing import List
import dataclasses
import pathlib

import pandas as pd
import numpy as np
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import GetByValueMixin
from covidactnow.datapublic.common_fields import ValueAsStrMixin
from covidactnow.datapublic.common_fields import PdFields

from libs.datasets import taglib
from libs.datasets import timeseries
from libs.datasets import dataset_utils

MultiRegionDataset = timeseries.MultiRegionDataset


NYTIMES_ANOMALIES_CSV = dataset_utils.REPO_ROOT / "data/nyt_anomalies.csv"


@enum.unique
class NYTimesFields(GetByValueMixin, ValueAsStrMixin, FieldName, enum.Enum):
    """Fields used in the NYTimes anomalies file"""

    DATE = "date"
    END_DATE = "end_date"
    COUNTY = "county"
    STATE = "state"
    GEOID = "geoid"
    TYPE = "type"
    OMIT_FROM_ROLLING_AVERAGE = "omit_from_rolling_average"
    OMIT_FROM_ROLLING_AVERAGE_ON_SUBGEOGRAPHIES = "omit_from_rolling_average_on_subgeographies"
    DESCRIPTION = "description"


@lru_cache(None)
def read_nytimes_anomalies():
    df = pd.read_csv(
        NYTIMES_ANOMALIES_CSV, parse_dates=[NYTimesFields.DATE, NYTimesFields.END_DATE]
    )

    # Extract fips from geoid column.
    df[CommonFields.FIPS] = df[NYTimesFields.GEOID].str.replace("USA-", "")

    # Denormalize data so that each row represents a single date+location+metric anomaly
    df = _denormalize_nyt_anomalies(df)

    # Add LOCATION_ID column (must happen after denormalizing since denormalizing can add additional
    # rows for subgeographies).
    df[CommonFields.LOCATION_ID] = df[CommonFields.FIPS].map(dataset_utils.get_fips_to_location())

    # A few locations (e.g. NYC aggregated FIPS 36998) don't have location IDs. That's okay, just remove them.
    df = df.loc[df[CommonFields.LOCATION_ID].notna()]

    # Convert "type" column into "variable" column using new_cases / new_deaths as the variable.
    assert df[NYTimesFields.TYPE].isin(["cases", "deaths"]).all()
    df[PdFields.VARIABLE] = df[NYTimesFields.TYPE].map(
        {"cases": CommonFields.NEW_CASES, "deaths": CommonFields.NEW_DEATHS}
    )

    # Add demographic bucket (all) to make it more compatible with our dataset structure.
    df[PdFields.DEMOGRAPHIC_BUCKET] = "all"

    return df


# TODO(mikelehen): This should probably live somewhere more central, but I'm not sure where.
def _get_county_fips_codes_for_state(state_fips_code: str) -> List[str]:
    """Helper to get county FIPS codes for all counties in a given state."""
    geo_data = dataset_utils.get_geo_data()
    state = geo_data.set_index("fips").at[state_fips_code, "state"]
    counties_df = geo_data.loc[
        (geo_data["state"] == state) & (geo_data["aggregate_level"] == "county")
    ]
    counties_fips = counties_df["fips"].to_list()
    return counties_fips


def _denormalize_nyt_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    The NYT anomaly data is normalized such that each row can represent an
    anomaly for multiple dates, locations, and metrics. We want to denormalize
    it so that each row represents a single date+location+metric anomaly.
    """

    # Look for rows with an end_date and create separate rows for each date in the [date, end_date] range.
    def date_range_for_row(row: pd.Series):
        return pd.date_range(
            row[NYTimesFields.DATE],
            row[NYTimesFields.DATE]
            if pd.isna(row[NYTimesFields.END_DATE])
            else row[NYTimesFields.END_DATE],
            freq="d",
        )

    df = df.assign(date=df.apply(date_range_for_row, axis=1), end_date=pd.NaT,).explode(
        NYTimesFields.DATE
    )

    # Look for state rows with omit_from_rolling_average_on_subgeographies and add new rows for each county within the state
    omit_subgeographies_mask = (
        df[NYTimesFields.OMIT_FROM_ROLLING_AVERAGE_ON_SUBGEOGRAPHIES] == "yes"
    ) & (df[CommonFields.FIPS].str.len() == 2)
    df_subgeography_rows = df.loc[omit_subgeographies_mask].copy()
    df_subgeography_rows = df_subgeography_rows.assign(
        fips=df_subgeography_rows.apply(
            lambda row: _get_county_fips_codes_for_state(row[CommonFields.FIPS]), axis=1
        ),
        omit_from_rolling_average="yes",
    ).explode(CommonFields.FIPS)
    df = df.append(df_subgeography_rows)

    # Look for rows with type=='both' and replace with two rows ('cases' and 'deaths')
    df = df.assign(
        type=df.apply(
            lambda row: ["cases", "deaths"]
            if row[NYTimesFields.TYPE] == "both"
            else row[NYTimesFields.TYPE],
            axis=1,
        )
    ).explode(NYTimesFields.TYPE)

    return df.reset_index()


def filter_by_nyt_anomalies(dataset: MultiRegionDataset) -> MultiRegionDataset:
    """Applies NYT anomalies data to case and death series.

    NYT Anomalies marked with omit_from_rolling_average are removed and tagged.
    """
    nyt_anomalies = read_nytimes_anomalies()

    new_tags = taglib.TagCollection()
    timeseries_copy = dataset.timeseries_bucketed.copy()

    # Find the applicable anomalies by intersecting indexes.
    assert timeseries_copy.index.names == [
        CommonFields.LOCATION_ID,
        PdFields.DEMOGRAPHIC_BUCKET,
        CommonFields.DATE,
    ]
    nyt_anomalies_indexed = nyt_anomalies.set_index(
        [CommonFields.LOCATION_ID, PdFields.DEMOGRAPHIC_BUCKET, CommonFields.DATE,]
    )
    overlapping_index = timeseries_copy.index.intersection(nyt_anomalies_indexed.index)
    if overlapping_index.size == 0:
        return dataset

    # Get the applicable anomalies.
    applicable_nyt_anomalies = nyt_anomalies_indexed.loc[overlapping_index].reset_index()

    for index, row in applicable_nyt_anomalies.iterrows():
        if row[NYTimesFields.OMIT_FROM_ROLLING_AVERAGE] == "yes":
            location_id = row[CommonFields.LOCATION_ID]
            bucket = row[PdFields.DEMOGRAPHIC_BUCKET]
            date = row[NYTimesFields.DATE]
            description = "NYTimes: {}".format(row["description"])
            variable = row[PdFields.VARIABLE]
            assert location_id and bucket and date and description and variable

            new_tags.add(
                taglib.KnownIssue(public_note=description, date=date.date()),
                location_id=location_id,
                variable=variable,
                bucket=bucket,
            )
            timeseries_copy.at[(location_id, bucket, date), variable] = np.nan

    # Return dataset with filtered timeseries.
    return dataclasses.replace(dataset, timeseries_bucketed=timeseries_copy).append_tag_df(
        new_tags.as_dataframe()
    )
