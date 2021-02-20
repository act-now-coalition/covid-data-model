import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import git
import more_itertools
import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import PdFields
from dash.dependencies import Input
from dash.dependencies import Output
from pandas.core.dtypes.common import is_numeric_dtype
from plotly import express as px

from libs import pipeline
from libs.datasets import combined_datasets
from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.taglib import TagField
from libs.datasets.tail_filter import TagType

EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# These columns match the OneRegion tag attribute. Unlike timeseries.TAG_INDEX_FIELDS it does
# not contain LOCATION_ID.
TAG_TABLE_COLUMNS = [TagField.VARIABLE, TagField.TYPE, TagField.CONTENT]


def _location_id_to_agg(loc_id, just_levels=True):
    region = pipeline.Region.from_location_id(loc_id)

    if just_levels:
        return region.level.value

    if region.is_county():
        return region.state
    else:
        return region.level.value


def _agg_wide_var_counts(wide_vars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate wide variable counts to make a smaller table."""
    assert wide_vars.index.names == [CommonFields.LOCATION_ID]
    assert wide_vars.columns.names == [PdFields.VARIABLE]
    assert is_numeric_dtype(more_itertools.one(set(wide_vars.dtypes)))

    agg_counts = (
        wide_vars.groupby(_location_id_to_agg).sum().rename_axis(index=CommonFields.AGGREGATE_LEVEL)
    )
    agg_counts = agg_counts.groupby(common_fields.COMMON_FIELD_TO_GROUP, axis=1, sort=False).sum()
    # Reindex columns to match order of FieldGroup enum.
    agg_counts = agg_counts.reindex(
        columns=pd.Index(common_fields.FieldGroup).intersection(agg_counts.columns)
    )

    return agg_counts


def init(server):
    dash_app = dash.Dash(__name__, server=server, external_stylesheets=EXTERNAL_STYLESHEETS)

    # Enable offline use.
    dash_app.css.config.serve_locally = True
    dash_app.scripts.config.serve_locally = True

    ds = combined_datasets.load_us_timeseries_dataset().get_subset(exclude_county_999=True)

    commit = git.Repo(dataset_utils.REPO_ROOT).head.commit
    commit_str = (
        f"commit {commit.hexsha} at {commit.committed_datetime.isoformat()}: {commit.summary}"
    )

    # A table of regions in a DataFrame.
    df_regions = ds.static.copy()
    df_regions["annotation_count"] = (
        ds.tag.loc[:, :, timeseries.ANNOTATION_TAG_TYPES]
        .index.get_level_values(CommonFields.LOCATION_ID)
        .value_counts()
    )
    df_regions = df_regions.reset_index()  # Move location_id from the index to a regular column
    df_regions["id"] = df_regions[CommonFields.LOCATION_ID]

    df_popular_urls = (
        ds.tag.loc[:, :, TagType.SOURCE_URL]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "URL", "content": "count"})
    )

    wide_var_has_timeseries = (
        ds.timeseries_wide_dates()
        .notnull()
        .any(1)
        .unstack(PdFields.VARIABLE, fill_value=False)
        .astype(int)
    )
    agg_has_timeseries = _agg_wide_var_counts(wide_var_has_timeseries).reset_index()

    wide_var_has_url = (
        ds.tag.loc[:, :, TagType.SOURCE_URL].unstack(PdFields.VARIABLE).notnull().astype(int)
    )
    agg_has_url = _agg_wide_var_counts(wide_var_has_url).reset_index()

    dash_app.layout = html.Div(
        children=[
            html.H1(children="CAN Data Pipeline Dashboard"),
            html.P(commit_str),
            html.H2("Time-series count"),
            dash_table.DataTable(
                id="agg_has_timeseries",
                columns=[{"name": i, "id": i} for i in agg_has_timeseries.columns],
                cell_selectable=False,
                data=agg_has_timeseries.to_dict("records"),
                editable=False,
                page_action="native",
            ),
            html.H2("Source URLs"),
            dash_table.DataTable(
                id="source_url_counts",
                columns=[{"name": i, "id": i} for i in df_popular_urls.columns],
                page_size=8,
                data=df_popular_urls.to_dict("records"),
                editable=False,
                page_action="native",
            ),
            html.Br(),  # Give table above some space for page action controls
            html.Br(),  # Give table above some space for page action controls
            html.Br(),  # Give table above some space for page action controls
            dash_table.DataTable(
                id="agg_has_url",
                columns=[{"name": i, "id": i} for i in agg_has_url.columns],
                cell_selectable=True,
                data=agg_has_url.to_dict("records"),
                editable=False,
                page_action="native",
            ),
            html.H2("Regions"),
            dash_table.DataTable(
                id="datatable-regions",
                columns=[
                    {"name": i, "id": i}
                    for i in [
                        CommonFields.LOCATION_ID,
                        CommonFields.COUNTY,
                        CommonFields.STATE,
                        CommonFields.POPULATION,
                        "annotation_count",
                    ]
                ],
                cell_selectable=False,
                page_size=8,
                row_selectable="single",
                data=df_regions.to_dict("records"),
                editable=False,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                page_action="native",
                style_table={"height": "300px", "overflowY": "auto"},
                # selected_row_ids=[df_regions[CommonFields.POPULATION].idxmax()],
                selected_rows=[df_regions[CommonFields.POPULATION].idxmax()],
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

    _init_callbacks(dash_app, ds, df_regions)

    return dash_app.server


def _init_callbacks(dash_app, ds: timeseries.MultiRegionDataset, df_regions: pd.DataFrame):

    # Work-around to get initial selection, from
    # https://github.com/plotly/dash-table/issues/707#issuecomment-626890525
    @dash_app.callback(
        [Output("datatable-regions", "selected_row_ids")],
        [Input("datatable-regions", "selected_rows")],
        prevent_initial_call=False,
    )
    def update_selected_rows(selected_rows):
        return [df_regions["id"].iat[more_itertools.one(selected_rows)]]

    # Input not in a list raises dash.exceptions.IncorrectTypeException: The input argument
    # `location-dropdown.value` must be a list or tuple of `dash.dependencies.Input`s.
    # but doesn't in the docs at https://dash.plotly.com/basic-callbacks. Odd.
    @dash_app.callback(
        [Output("region-graph", "figure"), Output("region-tag-table", "data")],
        [Input("datatable-regions", "selected_row_ids")],
        prevent_initial_call=False,
    )
    def update_figure(selected_row_ids):
        print(f"update figure: {selected_row_ids}")
        # Not sure why this isn't consistent but oh well
        if isinstance(selected_row_ids, str):
            selected_row_id = selected_row_ids
        else:
            selected_row_id = more_itertools.one(selected_row_ids)
        one_region = ds.get_one_region(pipeline.Region.from_location_id(selected_row_id))
        interesting_ts = one_region.data.set_index(CommonFields.DATE).select_dtypes(
            include="number"
        )
        fig = px.scatter(interesting_ts.reset_index(), x="date", y=interesting_ts.columns.to_list())
        tag_df = one_region.tag.reset_index()
        assert list(tag_df.columns) == TAG_TABLE_COLUMNS
        return fig, tag_df.to_dict("records")
