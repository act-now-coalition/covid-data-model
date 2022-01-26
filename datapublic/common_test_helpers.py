import numbers
from typing import List

import numpy as np
import pandas as pd


def _is_empty(v):
    if v is None:
        return True
    if v == "":
        return True
    if not isinstance(v, numbers.Number):
        return False
    return np.isnan(v)


def to_dict(keys: List[str], df: pd.DataFrame):
    """Transforms df into a dict mapping columns `keys` to a dict of the record/row in df.

    Use this to extract the values from a DataFrame for easier comparisons in assert statements.
    """
    try:
        if df.empty:
            return {}
        if any(df.index.names):
            df = df.reset_index()
        df = df.set_index(keys)
        records_without_nas = {}
        for key, values in df.to_dict(orient="index").items():
            records_without_nas[key] = {k: v for k, v in values.items() if not _is_empty(v)}
        return records_without_nas
    except Exception:
        # Print df to provide more context when the above code raises.
        print(f"Problem with {df}")
        raise
