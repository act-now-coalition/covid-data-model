import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_pivottable
import dash_table
import git
import more_itertools
import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import PdFields
from dash.dependencies import Input
from dash.dependencies import Output
from plotly import express as px

from libs import pipeline
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets
from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.taglib import TagField
from libs.datasets.tail_filter import TagType
from libs.qa import timeseries_stats

EXTERNAL_STYLESHEETS = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# These columns match the OneRegion tag attribute. Unlike timeseries.TAG_INDEX_FIELDS it does
# not contain LOCATION_ID.
TAG_TABLE_COLUMNS = [
    TagField.VARIABLE,
    TagField.DEMOGRAPHIC_BUCKET,
    TagField.TYPE,
    TagField.CONTENT,
]


def region_table(
    stats: timeseries_stats.PerTimeseries, dataset: timeseries.MultiRegionDataset
) -> pd.DataFrame:
    # Use an index to preserve the order while keeping only columns actually present.
    static_columns = pd.Index(
        [
            CommonFields.LOCATION_ID,
            CommonFields.COUNTY,
            CommonFields.STATE,
            CommonFields.POPULATION,
        ]
    ).intersection(dataset.static_and_geo_data.columns)
    regions = dataset.static_and_geo_data.loc[:, static_columns]

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

    per_timeseries_stats = timeseries_stats.PerTimeseries.make(ds)

    agg_level_and_field_group = per_timeseries_stats.aggregate(
        CommonFields.AGGREGATE_LEVEL, timeseries_stats.FIELD_GROUP
    )

    source_url_value_counts = (
        ds.tag_all_bucket.loc[:, :, TagType.SOURCE_URL]
        .value_counts()
        .rename_axis(index="URL")
        .rename("count")
    )

    counties = ds.get_subset(aggregation_level=AggregationLevel.COUNTY)
    county_stats = timeseries_stats.PerTimeseries.make(counties)
    county_variable_population_ratio = pd.DataFrame(
        {
            "has_url": population_ratio_by_variable(counties, county_stats.has_url),
            "has_timeseries": population_ratio_by_variable(counties, county_stats.has_timeseries),
        }
    )

    region_df = region_table(per_timeseries_stats, ds)

    df = per_timeseries_stats.stats.reset_index()
    pivottable_data = [df.columns.tolist()] + df.values.tolist()

    dash_app.layout = html.Div(
        children=[
            html.H1(children="CAN Data Pipeline Dashboard"),
            html.P(commit_str),
            html.H2("Time-series pivot table"),
            dcc.Markdown(
                "Drag attributes to explore information about time-series in "
                "this dataset. See an animated demo in the [Dash Pivottable docs]("
                "https://github.com/plotly/dash-pivottable#readme)."
            ),
            dash_pivottable.PivotTable(
                id="pivot_table",
                data=pivottable_data,
                rows=[CommonFields.AGGREGATE_LEVEL],
                cols=[timeseries_stats.FIELD_GROUP, timeseries_stats.DISTRIBUTION],
            ),
            html.H2("Source URLs"),
            dash_table_from_data_frame(
                source_url_value_counts, id="source_url_counts", page_size=8
            ),
            html.Br(),  # Give table above some space for page action controls
            html.Br(),  # Give table above some space for page action controls
            html.Br(),  # Give table above some space for page action controls
            dash_table_from_data_frame(agg_level_and_field_group.has_url, id="agg_has_url"),
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

    _init_callbacks(dash_app, ds, per_timeseries_stats, region_df["id"])

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
    population_where_true = zeros.mask(df.astype(bool), population_indexed, axis=0).sum(axis=0)
    population_ratio = population_where_true / population_total
    return population_ratio.rename("population_ratio")


def _init_callbacks(
    dash_app,
    ds: timeseries.MultiRegionDataset,
    per_timeseries_stats: timeseries_stats.PerTimeseries,
    region_id_series: pd.Series,
):
    @dash_app.callback(
        [Output("datatable-regions", "data"), Output("datatable-regions", "columns")],
        [Input("regions-variable-dropdown", "value")],
        prevent_initial_call=True,
    )
    def update_regions_table_variables(variable_dropdown_value):
        if variable_dropdown_value == "all":
            selected_var_stats = per_timeseries_stats
        else:
            selected_variables = common_fields.FIELD_GROUP_TO_LIST_FIELDS[variable_dropdown_value]
            selected_var_stats = per_timeseries_stats.subset_variables(selected_variables)

        region_df = region_table(selected_var_stats, ds)
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
