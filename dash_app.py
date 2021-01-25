import collections
from itertools import chain
from typing import MutableMapping

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import more_itertools

import plotly.express as px
import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields
from dash.dependencies import Input, Output

from libs import pipeline
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets
from libs.datasets import data_source
from libs.datasets import timeseries
from libs.datasets.tail_filter import TagType
from libs.datasets.timeseries import TagField
import pandas as pd

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True


ds = combined_datasets.load_us_timeseries_dataset().get_subset(exclude_county_999=True)

df_regions = ds.static
df_regions["annotation_count"] = (
    ds.tag.loc[:, :, [TagType.CUMULATIVE_LONG_TAIL_TRUNCATED, TagType.CUMULATIVE_TAIL_TRUNCATED]]
    .index.get_level_values(CommonFields.LOCATION_ID)
    .value_counts()
)
df_regions = df_regions.reset_index()  # Move location_id to from index to regular column
df_regions["id"] = df_regions[CommonFields.LOCATION_ID]

# These columns are from OneRegion... missing LOCATION_ID of timeseries.TAG_INDEX_FIELDS
TAG_TABLE_COLUMNS = [TagField.VARIABLE, TagField.TYPE, TagField.CONTENT]

field_def = {
    field: ", ".join(cls.__name__ for cls in clses)
    for field, clses in combined_datasets.ALL_TIMESERIES_FEATURE_DEFINITION.items()
}


field_provider = {field: ", ".join(cls_list) for field, cls_list in combined_datasets.foo().items()}

fields = (
    pd.concat(
        {
            "definition": pd.Series(field_def),
            "source": pd.Series(field_provider),
            "provenance": pd.Series({k: ", ".join(v) for k, v in ds.provenance_map().items()}),
        },
        axis=1,
    )
    .rename_axis(index=PdFields.VARIABLE)
    .reset_index()
)


app.layout = html.Div(
    children=[
        html.H1(children="CAN Data"),
        dash_table.DataTable(
            id="datatable-sources",
            columns=[{"name": i, "id": i} for i in fields.columns],
            cell_selectable=False,
            page_size=10,
            row_selectable=False,
            data=fields.reset_index().to_dict("records"),
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            page_action="native",
            style_table={"height": "300px", "overflowY": "auto"},
        ),
        html.Hr(),  # Stop graph drawing over table pageination control.
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
            page_size=10,
            row_selectable="single",
            data=df_regions.to_dict("records"),
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            page_action="native",
            style_table={"height": "300px", "overflowY": "auto"},
            selected_rows=[0],
            sort_by=[{"column_id": CommonFields.POPULATION, "direction": "desc"}],
        ),
        html.Hr(),  # Stop graph drawing over table pageination control.
        dcc.Graph(id="example-graph",),
        dash_table.DataTable(id="table", columns=[{"name": i, "id": i} for i in TAG_TABLE_COLUMNS]),
    ]
)


# Input not in a list raises
# dash.exceptions.IncorrectTypeException: The input argument `location-dropdown.value` must be a
# list or tuple of `dash.dependencies.Input`s.
# but doesn't in the docs at https://dash.plotly.com/basic-callbacks. Odd.
@app.callback(
    [Output("example-graph", "figure"), Output("table", "data")],
    [Input("datatable-regions", "selected_row_ids")],
    prevent_initial_call=True,
)
def update_figure(selected_rows):
    one_region = ds.get_one_region(
        pipeline.Region.from_location_id(more_itertools.one(selected_rows))
    )
    interesting_ts = one_region.data.set_index(CommonFields.DATE).select_dtypes(include="number")
    fig = px.scatter(interesting_ts.reset_index(), x="date", y=interesting_ts.columns.to_list())
    tag_df = one_region.tag.reset_index()
    assert list(tag_df.columns) == TAG_TABLE_COLUMNS
    return fig, tag_df.to_dict("records")


if __name__ == "__main__":
    app.run_server(debug=True)
