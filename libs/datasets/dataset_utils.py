import pandas as pd
import pathlib
from libs import build_params

LOCAL_PUBLIC_DATA_PATH = (
    pathlib.Path(__file__).parent.parent / ".." / ".." / "covid-data-public"
)


def strip_whitespace(data: pd.DataFrame) -> pd.DataFrame:
    """Removes all whitespace from string values.

    Note: Does not modify column names.

    Args:
        data: DataFrame

    Returns: New DataFrame with no whitespace.
    """
    # Remove all whitespace
    return data.applymap(lambda x: x.strip() if type(x) == str else x)


def standardize_county(series: pd.Series):
    return series.apply(lambda x: x if pd.isnull(x) else x.replace(" County", ""))


def parse_county_from_state(state):
    if pd.isnull(state):
        return state
    state_split = [val.strip() for val in state.split(",")]
    if len(state_split) == 2:
        return state_split[0]
    return None


def parse_state(state):
    if pd.isnull(state):
        return state
    state_split = [val.strip() for val in state.split(",")]
    state = state_split[1] if len(state_split) == 2 else state_split[0]
    return build_params.US_STATE_ABBREV.get(state, state)



def plot_grouped_data(data, group, series='source', values='cases'):
    data_by_source = data.groupby(group).sum().reset_index()
    cases_by_source = data_by_source.pivot_table(
        index=['date'], columns=series, values=values
    ).fillna(0)
    cases_by_source.plot(
        kind='bar',
        figsize=(15,7),
        title=f"{values} by data source vs date"
    )


def build_aggregate_county_data_frame(jhu_data_source, cds_data_source):
    """Combines JHU and CDS county data."""
    data = jhu_data_source.to_common(county_only=True)
    jhu_usa_data = data.get_country('USA').get_date(after='2020-03-01').data

    data = cds_data_source.to_common(county_only=True)
    cds_usa_data = data.get_country('USA').get_date(after='2020-03-01').data

    # TODO(chris): Better handling of counties that are not consistent.

    # Before 3-22, CDS has mostly consistent county level numbers - except for
    # 3-12, where there are no numbers reported. Still need to fill that in.
    return pd.concat([
        cds_usa_data[cds_usa_data.date < '2020-03-22'],
        jhu_usa_data[jhu_usa_data.date >= '2020-03-22']
    ])
