import dataclasses
import enum
from functools import lru_cache
from typing import List
from typing import Mapping

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_pivottable
import dash_table
import git
import more_itertools
import pandas as pd
from datapublic.common_fields import CommonFields
from datapublic import common_fields
from datapublic.common_fields import GetByValueMixin
from datapublic.common_fields import PdFields
from datapublic.common_fields import ValueAsStrMixin
from dash.dependencies import Input
from dash.dependencies import Output
from plotly import express as px

from libs import pipeline
from libs.datasets import combined_datasets
from libs.datasets import dataset_utils
from libs.datasets import new_cases_and_deaths
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
class Id(ValueAsStrMixin, str, enum.Enum):
    """Dash component ids. Use members of this enum as the string id."""

    def _generate_next_value_(name, start, count, last_values):  # pylint: disable=no-self-argument
        """Returns the name of the enum as it's value"""
        return name

    DATASET_PAGE_CONTENT = enum.auto()
    URL = enum.auto()
    PIVOT_TABLE_PARENT = enum.auto()
    REGION_GRAPH = enum.auto()
    REGION_TAG_TABLE = enum.auto()
    DATATABLE_REGIONS = enum.auto()
    DATASETS_DROPDOWN = enum.auto()
    REGIONS_VARIABLE_DROPDOWN = enum.auto()
    PIVOT_TABLE_PRESETS_DROPDOWN = enum.auto()


@enum.unique
class DashboardFile(GetByValueMixin, str, enum.Enum):
    """Enum of files that may be displayed in the dashboard.

    This is very de-coupled from the code that creates these files and will likely remain that
    way unless/until DatasetPointer is cleaned up. See also the TODO in dataset_utils.py."""

    # Define a custom __new__ so DashboardFile instances use 'file_key' as their value and have a
    # description.
    def __new__(cls, file_key, description):
        # Initialize super class (str) with file_key.
        obj = super().__new__(cls, file_key)
        # _value_ is a special name of enum. Set it here so enum code doesn't attempt to call
        # str(file_key, description).
        obj._value_ = file_key
        obj.description = description
        obj.file_key = file_key
        return obj

    COMBINED_DATA = "multiregion-wide-date", "Combined data"
    MANUAL_FILTER_REMOVED = "manual_filter_removed", "Manual filter removed timeseries"
    COMBINED_RAW = "combined-raw", "Combined data, raw from datasources"


@dataclasses.dataclass(frozen=True)
class RepoWrapper:
    # If you read files using _repo watch out for a nasty API where streams must be read in
    # order. See https://github.com/gitpython-developers/GitPython/issues/642#issuecomment-614588349
    _repo: git.Repo
    head_commit_str: str
    _working_copy_dataset: timeseries.MultiRegionDataset

    @staticmethod
    def make() -> "RepoWrapper":
        repo = git.Repo(dataset_utils.REPO_ROOT)
        commit = repo.head.commit
        commit_str = (
            f"commit {commit.hexsha} at {commit.committed_datetime.isoformat()}: {commit.summary}"
        )

        return RepoWrapper(repo, commit_str, combined_datasets.load_us_timeseries_dataset())

    @lru_cache(None)
    def get_stats(self, dataset_name: DashboardFile) -> timeseries_stats.PerTimeseries:
        if dataset_name is DashboardFile.COMBINED_DATA:
            dataset = self._working_copy_dataset.get_subset(exclude_county_999=True)
        elif dataset_name is DashboardFile.MANUAL_FILTER_REMOVED:
            dataset = timeseries.MultiRegionDataset.from_wide_dates_csv(
                dataset_utils.MANUAL_FILTER_REMOVED_WIDE_DATES_CSV_PATH
            ).add_static_csv_file(dataset_utils.MANUAL_FILTER_REMOVED_STATIC_CSV_PATH)
            dataset = new_cases_and_deaths.add_new_cases(dataset)
        elif dataset_name is DashboardFile.COMBINED_RAW:
            dataset = timeseries.MultiRegionDataset.from_compressed_pickle(
                dataset_utils.COMBINED_RAW_PICKLE_GZ_PATH
            )
        else:
            raise ValueError(f"Bad {dataset_name}")

        dataset = timeseries.make_source_url_tags(dataset)
        return timeseries_stats.PerTimeseries.make(dataset)


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


def region_table(stats: timeseries_stats.PerTimeseries) -> pd.DataFrame:
    dataset = stats.dataset
    # `geo_data` includes all locations in stats. `static` may have only a subset of locations.
    static_and_geo_data = dataset.geo_data.join(dataset.static)
    # Use an index to preserve the order while keeping only columns actually in static_and_geo_data.
    static_columns = pd.Index(
        [
            CommonFields.LOCATION_ID,
            CommonFields.COUNTY,
            CommonFields.STATE,
            CommonFields.POPULATION,
        ]
    ).intersection(static_and_geo_data.columns)
    regions = static_and_geo_data.loc[:, static_columns]

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

    repo = RepoWrapper.make()

    dash_app.layout = html.Div(
        [
            # TODO(tom): Add a mechanism to modify the URL, perhaps using
            #  https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
            dcc.Location(id=Id.URL, refresh=False),
            html.H1(children="CAN Data Pipeline Dashboard"),
            html.P(repo.head_commit_str),
            dropdown_select(
                "Select dataset: ",
                id=Id.DATASETS_DROPDOWN,
                options=[{"label": n.description, "value": n.file_key} for n in DashboardFile],
            ),
            html.Div(id=Id.DATASET_PAGE_CONTENT),
        ]
    )

    _init_callbacks(dash_app, repo)

    return dash_app.server


def dropdown_select(text: str, *, id: Id, options: List[Mapping]) -> html.Div:
    """Returns `text` next to a dropdown with options mappings having 'label' and 'value' keys."""
    return html.Div(
        [
            html.Div(text),
            dcc.Dropdown(
                id=id,
                options=options,
                value=more_itertools.first(options)["value"],
                clearable=False,
                # From https://stackoverflow.com/a/55755387/341400
                style=dict(width="40%"),
            ),
        ],
        style=dict(display="flex"),
    )


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


def _init_callbacks(dash_app, repo: RepoWrapper):
    @dash_app.callback(
        Output(Id.DATASET_PAGE_CONTENT, "children"), [Input(Id.DATASETS_DROPDOWN, "value")]
    )
    def update_dataset_page_content(datasets_dropdown_value):
        dashboard_file = DashboardFile.get(datasets_dropdown_value)
        per_timeseries_stats = repo.get_stats(dashboard_file)
        region_df = region_table(per_timeseries_stats)
        if region_df.index.empty:
            raise ValueError("Unexpected empty dataset")

        return html.Div(
            [
                html.H2("Time series pivot table"),
                dropdown_select(
                    "Preset views:",
                    id=Id.PIVOT_TABLE_PRESETS_DROPDOWN,
                    options=[
                        {"label": preset.description, "value": preset.btn_id}
                        for preset in TimeSeriesPivotTablePreset
                    ],
                ),
                html.P(),
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
                dcc.Loading(id=Id.PIVOT_TABLE_PARENT),
                html.H2("Regions"),
                dropdown_select(
                    "Select variables: ",
                    id=Id.REGIONS_VARIABLE_DROPDOWN,
                    options=[{"label": n, "value": n} for n in VARIABLE_GROUPS],
                ),
                dcc.Markdown(
                    "Select a region using the radio button at the left of this table to "
                    "view its data below. recent_vac_ratio is completed / initiated and is "
                    "expected to be about 0.60 to 0.95 and never more than 1."
                ),
                dash_table.DataTable(
                    id=Id.DATATABLE_REGIONS,
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
                dcc.Graph(id=Id.REGION_GRAPH),
                dash_table.DataTable(
                    id=Id.REGION_TAG_TABLE,
                    columns=[{"name": i, "id": i} for i in TAG_TABLE_COLUMNS],
                ),
            ]
        )

    @dash_app.callback(
        [Output(Id.DATATABLE_REGIONS, "data"), Output(Id.DATATABLE_REGIONS, "columns")],
        [Input(Id.REGIONS_VARIABLE_DROPDOWN, "value"), Input(Id.DATASETS_DROPDOWN, "value")],
        prevent_initial_call=True,
    )
    def update_regions_table_variables(variable_dropdown_value, datasets_dropdown_value):
        per_timeseries_stats = repo.get_stats(DashboardFile.get(datasets_dropdown_value))
        if variable_dropdown_value == "all":
            selected_var_stats = per_timeseries_stats
        else:
            selected_variables = common_fields.FIELD_GROUP_TO_LIST_FIELDS[variable_dropdown_value]
            selected_var_stats = per_timeseries_stats.subset_variables(selected_variables)

        region_df = region_table(selected_var_stats)
        columns = [{"name": i, "id": i} for i in region_df.columns if i != "id"]
        data = region_df.to_dict("records")
        return data, columns

    @dash_app.callback(
        Output(Id.PIVOT_TABLE_PARENT, "children"),
        [Input(Id.PIVOT_TABLE_PRESETS_DROPDOWN, "value"), Input(Id.DATASETS_DROPDOWN, "value"),],
    )
    def time_series_pivot_table_preset_btn_clicked(
        pivot_table_presets_dropdown_value, datasets_dropdown_value
    ):
        """Make a new pivot table when the table preset or dataset dropdown is changed."""
        per_timeseries_stats = repo.get_stats(DashboardFile.get(datasets_dropdown_value))
        preset = TimeSeriesPivotTablePreset.get_by_btn_id(pivot_table_presets_dropdown_value)
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
        [Output(Id.REGION_GRAPH, "figure"), Output(Id.REGION_TAG_TABLE, "data")],
        [
            Input(Id.DATATABLE_REGIONS, "selected_row_ids"),
            Input(Id.REGIONS_VARIABLE_DROPDOWN, "value"),
            Input(Id.DATASETS_DROPDOWN, "value"),
        ],
        prevent_initial_call=False,
    )
    def update_figure(selected_row_ids, variable_dropdown_value, datasets_dropdown_value):
        per_timeseries_stats = repo.get_stats(DashboardFile.get(datasets_dropdown_value))
        ds = per_timeseries_stats.dataset
        selected_row_id = more_itertools.one(selected_row_ids)
        assert isinstance(selected_row_id, str)
        try:
            one_region = ds.get_one_region(pipeline.Region.from_location_id(selected_row_id))
        except timeseries.RegionLatestNotFound:
            return dash.no_update, dash.no_update
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
