from typing import List, Optional
import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs import google_sheet_helpers
from libs.datasets.sources import fips_population
from libs import notebook_helpers
from libs.datasets import combined_datasets
from libs.datasets.latest_values_dataset import LatestValuesDataset
import gspread
import gspread_formatting

LOCATION_GROUP_KEY = "location_group"
COMBINED_DATA_KEY = "Combined Data"


def load_all_latest_sources():
    latest = combined_datasets.load_us_latest_dataset()
    sources = notebook_helpers.load_data_sources_by_name()
    sources_latest = {name: source.latest_values() for name, source in sources.items()}
    combined_latest_data = latest.data.copy()
    combined_latest_data["source"] = "Combined"
    sources_latest[COMBINED_DATA_KEY] = combined_latest_data

    for source_name in sources_latest:
        data = sources_latest[source_name]

        # Drop unknown sources
        invalid_locations = (data[CommonFields.FIPS].str.endswith("999")) | (
            data[CommonFields.FIPS].str.startswith("90")
        )
        data = data.loc[~invalid_locations]
        sources_latest[source_name] = LatestValuesDataset(data)

    return sources_latest


def build_data_availability_report(latest: LatestValuesDataset) -> pd.DataFrame:
    """Builds report containing counts of locations with values.

    Args:
        latest: Dataset to summarize.

    Returns:
    """

    data = latest.data.copy()

    if "population" not in data.columns:
        pop = fips_population.FIPSPopulation.local()
        pop_map = pop.data.set_index("fips")["population"]
        data["population"] = data["fips"].map(pop_map)

    def classify_row(row):

        if row.aggregate_level == "state":
            return "state data"
        return row.state

    def count_with_values(x):
        return x.apply(lambda y: sum(~y.isna()))

    data[LOCATION_GROUP_KEY] = data.apply(classify_row, axis=1)

    counts_per_location = data.groupby(LOCATION_GROUP_KEY).apply(count_with_values)

    columns_to_drop = [
        CommonFields.STATE,
        CommonFields.COUNTY,
        CommonFields.AGGREGATE_LEVEL,
    ]
    columns_to_drop = counts_per_location.columns.intersection(columns_to_drop)

    counts_per_location = counts_per_location.drop(columns_to_drop, axis="columns")
    counts_per_location["total_population"] = data.groupby(LOCATION_GROUP_KEY).population.sum()
    counts_per_location = counts_per_location.sort_values("total_population", ascending=False).drop(
        ["total_population"], axis="columns"
    )

    return counts_per_location.rename({LOCATION_GROUP_KEY: "num_locations"}, axis="columns")


def build_data_availability_for_field(
    latest_dataset: LatestValuesDataset, field: str
) -> pd.DataFrame:
    """Builds data availability report for a specific field.

    Args:
        latest_dataset: Dataset with multiple data sources
        field: Name of field to select.

    """
    data = latest_dataset.data
    data = data.set_index(
        [CommonFields.FIPS, CommonFields.AGGREGATE_LEVEL, CommonFields.STATE, "source"]
    )
    series = data[field]
    field_by_source = series.unstack(level=-1)
    field_by_source.columns = field_by_source.columns.get_level_values(0).values
    field_by_source = field_by_source.reset_index()
    field_by_source = LatestValuesDataset(field_by_source)
    return build_data_availability_report(field_by_source)


def update_google_sheet_with_data(sheet, data: pd.DataFrame, worksheet_name):

    sheet_data = [data.columns.values.tolist()] + data.where(~data.isna(), None).values.tolist()

    worksheet = google_sheet_helpers.create_or_replace_worksheet(sheet, worksheet_name)
    update_results = worksheet.update(sheet_data)

    # Rotate first column and resize to make more readable
    # TODO: Make this range dynamic
    num_columns = update_results["updatedColumns"]
    start = gspread.utils.rowcol_to_a1(1, 1)
    end = gspread.utils.rowcol_to_a1(1, num_columns)

    worksheet.format(f"{start}:{end}", {"textRotation": {"angle": 75}})
    sheet.batch_update(
        {
            "requests": [
                {
                    "autoResizeDimensions": {
                        "dimensions": {
                            "sheetId": worksheet.id,
                            "dimension": "COLUMNS",
                            "startIndex": 0,
                            "endIndex": len(data.columns),
                        }
                    }
                }
            ]
        }
    )
    return worksheet


def _build_color_scale_row_rule(worksheet, cell_range):
    rule = gspread_formatting.ConditionalFormatRule(
        ranges=[gspread_formatting.GridRange.from_a1_range(cell_range, worksheet)],
        gradientRule=gspread_formatting.GradientRule(
            minpoint=gspread_formatting.InterpolationPoint(
                type="NUMBER", value="0", color=gspread_formatting.Color(0.9, 0.48, 0.46)
            ),
            midpoint=gspread_formatting.InterpolationPoint(
                type="PERCENT", value="50", color=gspread_formatting.Color(0.98, 0.73, 0.01)
            ),
            maxpoint=gspread_formatting.InterpolationPoint(
                type="MAX", color=gspread_formatting.Color(0.34, 0.73, 0.54)
            ),
        ),
    )
    return rule


def update_multi_field_availability_report(
    sheet: gspread.Spreadsheet,
    dataset: pd.DataFrame,
    name: str,
    columns_to_drop: Optional[List[str]] = None,
):
    columns_to_drop = columns_to_drop or []
    # Reorder dataset indices to put location group and num_locations first
    # + sort columns for consistency.
    dataset = dataset.reset_index()
    columns = [column for column in dataset.columns.values if column not in columns_to_drop]
    special_order = {LOCATION_GROUP_KEY: -2, "num_locations": -1}
    columns = sorted(dataset.columns.values, key=lambda x: (special_order.get(x, 0), x))
    dataset = dataset[columns]

    worksheet = update_google_sheet_with_data(sheet, dataset, name)

    # Add conditional formatting
    rules = gspread_formatting.get_conditional_format_rules(worksheet)
    for i in range(len(dataset) + 1):
        row = i + 1
        end_column = gspread.utils.rowcol_to_a1(row, len(columns))
        rule = _build_color_scale_row_rule(worksheet, f"B{row}:{end_column}")

        rules.append(rule)
    rules.save()
