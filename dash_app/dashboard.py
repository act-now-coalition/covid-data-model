import enum
from dataclasses import dataclass
from typing import Collection

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import git
import more_itertools
import pandas as pd
import numpy as np
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import PdFields
from covidactnow.datapublic.common_fields import ValueAsStrMixin
from dash.dependencies import Input
from dash.dependencies import Output
from pandas.core.dtypes.common import is_numeric_dtype
from plotly import express as px

from libs import pipeline
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets
from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.taglib import TagField
from libs.datasets.tail_filter import TagType

EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# These columns match the OneRegion tag attribute. Unlike timeseries.TAG_INDEX_FIELDS it does
# not contain LOCATION_ID.
TAG_TABLE_COLUMNS = [TagField.VARIABLE, TagField.TYPE, TagField.CONTENT]


# TODO(tom): Move all the aggregation and statistics stuff to a different module and test it.


def _location_id_to_agg(loc_id):
    """Turns a location_id into a label used for aggregation. For now this is only the
    AggregationLevel but future UI changes could let the user aggregate regions by state etc."""
    region = pipeline.Region.from_location_id(loc_id)
    return region.level.value


def _location_id_to_agg_and_state(loc_id):
    region = pipeline.Region.from_location_id(loc_id)
    if region.is_county():
        return region.state
    else:
        return region.level.value


@enum.unique
class RegionAggregationMethod(ValueAsStrMixin, str, enum.Enum):
    LEVEL = "level"
    LEVEL_AND_COUNTY_BY_STATE = "level_and_county_by_state"


@enum.unique
class VariableAggregationMethod(ValueAsStrMixin, str, enum.Enum):
    FIELD_GROUP = "field_group"
    NONE = "none"


def _agg_wide_var_counts(
    wide_vars: pd.DataFrame,
    location_id_group_by: RegionAggregationMethod,
    var_group_by: VariableAggregationMethod,
) -> pd.DataFrame:
    """Aggregate wide variable counts to make a smaller table."""
    assert wide_vars.index.names == [CommonFields.LOCATION_ID]
    assert wide_vars.columns.names == [PdFields.VARIABLE]
    assert is_numeric_dtype(more_itertools.one(set(wide_vars.dtypes)))

    if location_id_group_by == RegionAggregationMethod.LEVEL:
        axis0_groupby = wide_vars.groupby(_location_id_to_agg)
    elif location_id_group_by == RegionAggregationMethod.LEVEL_AND_COUNTY_BY_STATE:
        axis0_groupby = wide_vars.groupby(_location_id_to_agg_and_state)
    else:
        raise ValueError("Bad location_id_group_by")

    agg_counts = axis0_groupby.sum().rename_axis(index=CommonFields.AGGREGATE_LEVEL)

    if var_group_by == VariableAggregationMethod.FIELD_GROUP:
        agg_counts = agg_counts.groupby(
            common_fields.COMMON_FIELD_TO_GROUP, axis=1, sort=False
        ).sum()
        # Reindex columns to match order of FieldGroup enum.
        agg_counts = agg_counts.reindex(
            columns=pd.Index(common_fields.FieldGroup).intersection(agg_counts.columns)
        )

    return agg_counts


@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class AggregatedStats:
    """Aggregated statistics, where index are regions and columns are variables. Either axis may
    be filtered to keep only a subset and/or aggregated."""

    # TODO(tom): Move all these into one DataFrame so one vector operation can apply to all of them.

    # A table of count of timeseries
    has_timeseries: pd.DataFrame
    # A table of count of URLs
    has_url: pd.DataFrame
    # A table of count of annotations
    annotation_count: pd.DataFrame


@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class PerRegionStats(AggregatedStats):
    """Instances of AggregatedStats where each row represents one region."""

    @staticmethod
    def make(ds: timeseries.MultiRegionDataset) -> "PerRegionStats":
        wide_var_has_url = ds.tag.loc[:, :, TagType.SOURCE_URL].unstack(PdFields.VARIABLE).notnull()
        # Need to use pivot_table instead of unstack to aggregate using sum.
        wide_var_annotation_count = pd.pivot_table(
            ds.tag.loc[:, :, timeseries.ANNOTATION_TAG_TYPES].notnull().reset_index(),
            values=TagField.CONTENT,
            index=CommonFields.LOCATION_ID,
            columns=PdFields.VARIABLE,
            aggfunc=np.sum,
            fill_value=0,
        )

        return PerRegionStats(
            has_timeseries=ds.wide_var_has_timeseries,
            has_url=wide_var_has_url,
            annotation_count=wide_var_annotation_count,
        )

    def aggregate(
        self, regions: RegionAggregationMethod, variables: VariableAggregationMethod
    ) -> AggregatedStats:
        return AggregatedStats(
            has_timeseries=_agg_wide_var_counts(self.has_timeseries, regions, variables),
            has_url=_agg_wide_var_counts(self.has_url, regions, variables),
            annotation_count=_agg_wide_var_counts(self.annotation_count, regions, variables),
        )

    def subset_variables(self, variables: Collection[CommonFields]) -> "PerRegionStats":
        """Returns a new PerRegionStats with only `variables` in the columns."""
        return PerRegionStats(
            has_timeseries=self.has_timeseries.loc[
                :, self.has_timeseries.columns.intersection(variables).rename(PdFields.VARIABLE)
            ],
            has_url=self.has_url.loc[
                :, self.has_url.columns.intersection(variables).rename(PdFields.VARIABLE)
            ],
            annotation_count=self.annotation_count.loc[
                :, self.annotation_count.columns.intersection(variables).rename(PdFields.VARIABLE)
            ],
        )

    def stats_for_locations(self, location_ids: pd.Index) -> pd.DataFrame:
        """Returns a DataFrame of statistics with `location_ids` as the index."""
        assert location_ids.names == [CommonFields.LOCATION_ID]
        # The stats likely don't have a value for every region. Replace any NAs with 0 so that
        # subtracting them produces a real value.
        df = pd.DataFrame(
            {
                "annotation_count": self.annotation_count.sum(axis=1),
                "url_count": self.has_url.sum(axis=1),
                "timeseries_count": self.has_timeseries.sum(axis=1),
            },
            index=location_ids,
        ).fillna(0)
        df["no_url_count"] = df["timeseries_count"] - df["url_count"]
        return df


def region_table(stats: PerRegionStats, dataset: timeseries.MultiRegionDataset) -> pd.DataFrame:
    # Use an index to preserve the order while keeping only columns actually present.
    static_columns = pd.Index(
        [
            CommonFields.LOCATION_ID,
            CommonFields.COUNTY,
            CommonFields.STATE,
            CommonFields.POPULATION,
        ]
    ).intersection(dataset.static.columns)
    regions = dataset.static.loc[:, static_columns]

    regions = regions.join(stats.stats_for_locations(regions.index))

    regions = regions.reset_index()  # Move location_id from the index to a regular column
    # Add location_id as the row id, used by DataTable. Maybe it makes more sense to rename the
    # DataFrame index to "id" and call reset_index() just before to_dict("records"). I dunno.
    regions["id"] = regions[CommonFields.LOCATION_ID]
    return regions


def init(server):
    dash_app = dash.Dash(__name__, server=server, external_stylesheets=EXTERNAL_STYLESHEETS)

    # Enable offline use.
    dash_app.css.config.serve_locally = True
    dash_app.scripts.config.serve_locally = True

    commit = git.Repo(dataset_utils.REPO_ROOT).head.commit
    commit_str = (
        f"commit {commit.hexsha} at {commit.committed_datetime.isoformat()}: {commit.summary}"
    )

    ds = combined_datasets.load_us_timeseries_dataset().get_subset(exclude_county_999=True)
    ds = timeseries.make_source_url_tags(ds)

    variable_groups = ["all"] + list(common_fields.FieldGroup)

    per_region_stats_all_vars = PerRegionStats.make(ds)

    agg_stats = per_region_stats_all_vars.aggregate(
        RegionAggregationMethod.LEVEL, VariableAggregationMethod.FIELD_GROUP
    )

    source_url_value_counts = (
        ds.tag.loc[:, :, TagType.SOURCE_URL].value_counts().rename_axis(index="URL").rename("count")
    )

    counties = ds.get_subset(aggregation_level=AggregationLevel.COUNTY)
    county_stats = PerRegionStats.make(counties)
    county_variable_population_ratio = pd.DataFrame(
        {
            "has_url": population_ratio_by_variable(counties, county_stats.has_url),
            "has_timeseries": population_ratio_by_variable(counties, county_stats.has_timeseries),
        }
    )

    region_df = region_table(per_region_stats_all_vars, ds)

    dash_app.layout = html.Div(
        children=[
            html.H1(children="CAN Data Pipeline Dashboard"),
            html.P(commit_str),
            html.H2("Time-series count"),
            dash_table_from_data_frame(agg_stats.has_timeseries, id="agg_has_timeseries"),
            html.H2("Source URLs"),
            dash_table_from_data_frame(
                source_url_value_counts, id="source_url_counts", page_size=8
            ),
            html.Br(),  # Give table above some space for page action controls
            html.Br(),  # Give table above some space for page action controls
            html.Br(),  # Give table above some space for page action controls
            dash_table_from_data_frame(agg_stats.has_url, id="agg_has_url"),
            html.P("Ratio of population in county data with a URL, by variable"),
            dash_table_from_data_frame(
                county_variable_population_ratio, id="county_variable_population_ratio"
            ),
            html.H2("Regions"),
            html.Div(
                [
                    html.Div("Select variables: "),
                    dcc.Dropdown(
                        id="regions-variable-dropdown",
                        options=[{"label": n, "value": n} for n in variable_groups],
                        value="all",
                        clearable=False,
                        # From https://stackoverflow.com/a/55755387/341400
                        style=dict(width="40%"),
                    ),
                ],
                style=dict(display="flex"),
            ),
            dash_table.DataTable(
                id="datatable-regions",
                columns=[{"name": i, "id": i} for i in region_df.columns if i != "id"],
                cell_selectable=False,
                page_size=8,
                row_selectable="single",
                data=region_df.to_dict("records"),
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_action="native",
                style_table={"height": "330px", "overflowY": "auto"},
                # selected_row_ids=[df_regions[CommonFields.POPULATION].idxmax()],
                selected_rows=[region_df[CommonFields.POPULATION].idxmax()],
                # selected_rows=[0],
                sort_by=[{"column_id": CommonFields.POPULATION, "direction": "desc"}],
            ),
            html.P(),
            html.Hr(),  # Stop graph drawing over table pageination control.
            dcc.Graph(id="region-graph",),
            dash_table.DataTable(
                id="region-tag-table", columns=[{"name": i, "id": i} for i in TAG_TABLE_COLUMNS]
            ),
        ]
    )

    _init_callbacks(dash_app, ds, per_region_stats_all_vars, region_df["id"])

    return dash_app.server


def dash_table_from_data_frame(df: pd.DataFrame, *, id, **kwargs):
    """Returns a dash_table.DataTable that will render `df` in a simple HTML table."""
    df_all_columns = df.reset_index()
    return dash_table.DataTable(
        id=id,
        columns=[{"name": i, "id": i} for i in df_all_columns.columns],
        cell_selectable=False,
        data=df_all_columns.to_dict("records"),
        editable=False,
        page_action="native",
        **kwargs,
    )


def population_ratio_by_variable(
    dataset: timeseries.MultiRegionDataset, df: pd.DataFrame
) -> pd.DataFrame:
    """Finds the ratio of the population where df is True, broken down by column/variable."""
    assert df.index.names == [CommonFields.LOCATION_ID]
    assert df.columns.names == [PdFields.VARIABLE]
    population_indexed = dataset.static[CommonFields.POPULATION].reindex(df.index)
    population_total = population_indexed.sum()
    # Make a DataFrame that is like df but filled with zeros.
    zeros = pd.DataFrame(0, index=df.index, columns=df.columns)
    # Where df is True add the population, otherwise add zero. The result is a series with
    # PdFields.VARIABLE index
    population_where_true = zeros.mask(df, population_indexed, axis=0).sum(axis=0)
    population_ratio = population_where_true / population_total
    return population_ratio.rename("population_ratio")


def _init_callbacks(
    dash_app,
    ds: timeseries.MultiRegionDataset,
    per_region_stats_all_vars: PerRegionStats,
    region_id_series: pd.Series,
):
    @dash_app.callback(
        [Output("datatable-regions", "data"), Output("datatable-regions", "columns")],
        [Input("regions-variable-dropdown", "value")],
        prevent_initial_call=True,
    )
    def update_regions_table_variables(variable_dropdown_value):
        if variable_dropdown_value == "all":
            per_region_stats = per_region_stats_all_vars
        else:
            selected_variables = common_fields.FIELD_GROUP_TO_LIST_FIELDS[variable_dropdown_value]
            per_region_stats = per_region_stats_all_vars.subset_variables(selected_variables)

        region_df = region_table(per_region_stats, ds)
        columns = [{"name": i, "id": i} for i in region_df.columns if i != "id"]
        data = region_df.to_dict("records")
        return data, columns

    # Work-around to get initial selection, from
    # https://github.com/plotly/dash-table/issues/707#issuecomment-626890525
    @dash_app.callback(
        [Output("datatable-regions", "selected_row_ids")],
        [Input("datatable-regions", "selected_rows")],
        prevent_initial_call=False,
    )
    def update_selected_rows(selected_rows):
        selected_row = more_itertools.one(selected_rows)
        return [region_id_series.iat[selected_row]]

    # Input not in a list raises dash.exceptions.IncorrectTypeException: The input argument
    # `location-dropdown.value` must be a list or tuple of `dash.dependencies.Input`s.
    # but doesn't in the docs at https://dash.plotly.com/basic-callbacks. Odd.
    @dash_app.callback(
        [Output("region-graph", "figure"), Output("region-tag-table", "data")],
        [
            Input("datatable-regions", "selected_row_ids"),
            Input("regions-variable-dropdown", "value"),
        ],
        prevent_initial_call=False,
    )
    def update_figure(selected_row_ids, variable_dropdown_value):
        # Not sure why this isn't consistent but oh well
        if isinstance(selected_row_ids, str):
            selected_row_id = selected_row_ids
        else:
            selected_row_id = more_itertools.one(selected_row_ids)
        one_region = ds.get_one_region(pipeline.Region.from_location_id(selected_row_id))
        interesting_ts = one_region.data.set_index(CommonFields.DATE).select_dtypes(
            include="number"
        )
        interesting_ts = interesting_ts.dropna(axis=1, how="all")

        tag_df = one_region.tag.reset_index()
        assert list(tag_df.columns) == TAG_TABLE_COLUMNS

        # Keep only variables that have been selected in regions-variable-dropdown.
        if variable_dropdown_value != "all":
            selected_variables = common_fields.FIELD_GROUP_TO_LIST_FIELDS[variable_dropdown_value]
            interesting_ts = interesting_ts.loc[
                :, interesting_ts.columns.intersection(selected_variables)
            ]

            tag_df = tag_df.loc[tag_df[TagField.VARIABLE].isin(selected_variables)]

        fig = px.scatter(interesting_ts.reset_index(), x="date", y=interesting_ts.columns.to_list())
        return fig, tag_df.to_dict("records")
