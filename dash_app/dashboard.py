import enum

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
from libs.datasets import combined_datasets
from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.taglib import TagField
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


VARIABLE_GROUPS = ["all"] + list(common_fields.FieldGroup)


def _remove_prefix(text, prefix):
    assert text.startswith(prefix)
    return text[len(prefix) :]


@enum.unique
class TimeSeriesPivotTablePreset(enum.Enum):
    SOURCES = (
        "Per source count",
        {
            "rows": [CommonFields.AGGREGATE_LEVEL, timeseries_stats.FIELD_GROUP],
            "cols": [timeseries_stats.SOURCE_TYPE_SET],
        },
    )
    SOURCE_BY_STATE = (
        "Sources by state",
        {
            "rows": [CommonFields.AGGREGATE_LEVEL, CommonFields.STATE],
            "cols": [timeseries_stats.FIELD_GROUP],
            "aggregatorName": "List Unique Values",
            "vals": [timeseries_stats.SOURCE_TYPE_SET],
        },
    )
    COUNTY_DEMO = (
        "County vaccines by demographic attributes",
        {
            "rows": [CommonFields.AGGREGATE_LEVEL, CommonFields.STATE],
            "cols": [timeseries_stats.FIELD_GROUP, timeseries_stats.DISTRIBUTION],
            "valueFilter": {timeseries_stats.DISTRIBUTION: {"all": False}},
        },
    )

    def __new__(cls, description, pivot_table_parameters):
        # Make a unique string _value_ for each preset, mostly copied from
        # https://docs.python.org/3/library/enum.html#using-a-custom-new
        value = str(len(cls.__members__) + 1)
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        obj.pivot_table_parameters = pivot_table_parameters
        return obj

    @classmethod
    def get_by_btn_id(cls, btn_id) -> "TimeSeriesPivotTablePreset":
        # Not sure why lint complains but ... pylint: disable=no-member
        return cls._value2member_map_[_remove_prefix(btn_id, "pivot_table_btn_")]

    @property
    def btn_id(self):
        return f"pivot_table_btn_{self._value_}"

    @property
    def tbl_id(self):
        return f"pivot_table_{self._value_}"


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

    recent_wide_dates = dataset.timeseries_bucketed_wide_dates.iloc(axis=1)[-14:]
    recent_vaccinations_completed = recent_wide_dates.xs(
        axis=0,
        level=[PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET],
        key=[CommonFields.VACCINATIONS_COMPLETED, "all"],
    )
    recent_vaccinations_initiated = recent_wide_dates.xs(
        axis=0,
        level=[PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET],
        key=[CommonFields.VACCINATIONS_INITIATED, "all"],
    )
    if not recent_vaccinations_completed.empty and not recent_vaccinations_initiated.empty:
        completed_initiated_ratio = recent_vaccinations_completed / recent_vaccinations_initiated
        regions = regions.join(completed_initiated_ratio.max(axis=1).rename("recent_vac_ratio_max"))
        regions = regions.join(
            completed_initiated_ratio.idxmax(axis=1).rename("recent_vac_ratio_max_date")
        )

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

    per_timeseries_stats = timeseries_stats.PerTimeseries.make(ds)

    region_df = region_table(per_timeseries_stats, ds)

    dash_app.layout = html.Div(
        children=[
            html.H1(children="CAN Data Pipeline Dashboard"),
            html.P(commit_str),
            html.H2("Time series pivot table"),
            html.P("Preset views:"),
            *[
                html.Button(preset.description, id=preset.btn_id)
                for preset in TimeSeriesPivotTablePreset
            ],
            dcc.Markdown(
                "Drag attributes to explore information about time series in "
                "this dataset. See an animated demo in the [Dash Pivottable docs]("
                "https://github.com/plotly/dash-pivottable#readme)."
            ),
            # PivotTable `rows` and `cols` properties can not be modified by dash on an existing
            # object, see
            # https://github.com/plotly/dash-pivottable/blob/master/README.md#references. As a
            # work around `pivot_table_parent` is updated to add a new PivotTable when a button
            # is clicked.
            dcc.Loading(id="pivot_table_parent"),
            html.H2("Regions"),
            html.Div(
                [
                    html.Div("Select variables: "),
                    dcc.Dropdown(
                        id="regions-variable-dropdown",
                        options=[{"label": n, "value": n} for n in VARIABLE_GROUPS],
                        value="all",
                        clearable=False,
                        # From https://stackoverflow.com/a/55755387/341400
                        style=dict(width="40%"),
                    ),
                ],
                style=dict(display="flex"),
            ),
            dcc.Markdown(
                "Select a region using the radio button at the left of this table to "
                "view its data below. recent_vac_ratio is completed / initiated and is "
                "expected to be about 0.60 to 0.95 and never more than 1."
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
                # Default to the first row of `region_df`.
                # As a work around for https://github.com/plotly/dash-table/issues/707 pass the
                # selected row offset integer (for the UI) and row id string (for `update_figure`).
                selected_rows=[0],
                selected_row_ids=[region_df["id"].iat[0]],
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

    @dash_app.callback(
        Output("pivot_table_parent", "children"),
        [Input(preset.btn_id, "n_clicks") for preset in TimeSeriesPivotTablePreset],
    )
    def time_series_pivot_table_preset_btn_clicked(*inputs_ignored):
        """Make a new pivot table according to the most recently clicked button.

        We can't have a callback function for each button because
        https://dash.plotly.com/callback-gotchas "A component/property pair can only be the Output
        of one callback"."""

        # From https://dash.plotly.com/advanced-callbacks "Determining which Input has fired"
        ctx = dash.callback_context
        if not ctx.triggered:
            # Use the first as the default when page is loaded.
            preset = more_itertools.first(TimeSeriesPivotTablePreset)
        else:
            preset = TimeSeriesPivotTablePreset.get_by_btn_id(
                ctx.triggered[0]["prop_id"].split(".")[0]
            )

        # Each PivotTable needs a unique id as a work around for
        # https://github.com/plotly/dash-pivottable/issues/10
        return dash_pivottable.PivotTable(
            id=preset.tbl_id,
            data=per_timeseries_stats.pivottable_data,
            **preset.pivot_table_parameters,
        )

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
        selected_row_id = more_itertools.one(selected_row_ids)
        assert isinstance(selected_row_id, str)
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
        # Default to comparing values on the same date.
        fig.update_layout(hovermode="x")
        return fig, tag_df.to_dict("records")
