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
