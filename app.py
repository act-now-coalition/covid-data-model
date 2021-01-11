import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.express as px
import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from dash.dependencies import Input, Output

from libs import pipeline
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets
from libs.datasets import timeseries

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.css.config.serve_locally = True
app.scripts.config.serve_locally = True


ds = combined_datasets.load_us_timeseries_dataset().get_subset(
    exclude_county_999=True, aggregation_level=AggregationLevel.COUNTY,
)

dropdown_df = ds.static.reset_index().loc[:, [CommonFields.LOCATION_ID, CommonFields.COUNTY]]
dropdown_df = dropdown_df.rename(
    columns={CommonFields.LOCATION_ID: "value", CommonFields.COUNTY: "label"}
)
dropdown_options = dropdown_df.to_dict("records")


app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(
            children="""
        Dash: A web application framework for Python.
    """
        ),
        html.Hr(),
        dcc.Dropdown(
            id="location-dropdown", options=dropdown_options, value=dropdown_options[0]["value"]
        ),
        dcc.Graph(id="example-graph",),
    ]
)


# Input not in a list raises
# dash.exceptions.IncorrectTypeException: The input argument `location-dropdown.value` must be a
# list or tuple of `dash.dependencies.Input`s.
# but doesn't in the docs at https://dash.plotly.com/basic-callbacks. Odd.
@app.callback(Output("example-graph", "figure"), [Input("location-dropdown", "value")])
def update_figure(selected_location):
    one_region = ds.get_one_region(pipeline.Region.from_location_id(selected_location))
    interesting_ts = one_region.data.set_index(CommonFields.DATE)[
        [c for c in one_region.data.columns if ("test" in c or c in ("cases", "deaths"))]
    ]
    fig = px.scatter(
        interesting_ts.diff().reset_index(), x="date", y=interesting_ts.columns.to_list()
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
