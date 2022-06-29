from typing import List, Optional
import enum
import os
import pathlib
import csv
import io
import logging
import pandas as pd
from functools import partial

from libs.parallel_utils import parallel_map

_logger = logging.getLogger(__name__)


class DatasetDeployer(object):
    """Common deploy operations for persisting files to a local folder.

    """

    def __init__(self, key="filename.csv", body="a random data", output_dir="."):
        self.key = key
        self.body = body
        self.output_dir = output_dir

    def _persist_to_local(self):
        """Persists specific data onto an s3 bucket.
        This method assumes versioned is handled on the bucket itself.
        """
        _logger.info(f"persisting {self.key} {self.output_dir}")

        with open(os.path.join(self.output_dir, self.key), "wb") as f:
            # hack to allow the local writer to take either bytes or a string
            # note this assumes that all strings are given in utf-8 and not,
            # like, ASCII
            f.write(self.body.encode("UTF-8") if isinstance(self.body, str) else self.body)

    def persist(self):
        self._persist_to_local()


def upload_csv(key_name: str, csv: str, output_dir: str):
    blob = {
        "key": f"{key_name}.csv",
        "body": csv,
        "output_dir": output_dir,
    }
    obj = DatasetDeployer(**blob)
    obj.persist()
    _logger.info(f"Generated csv for {key_name}")


def flatten_dict(data: dict, level_separator: str = ".") -> dict:
    """Flattens a nested dictionary, separating nested keys by separator.

    Args:
        data: data to flatten
        level_separator: separator to use when combining keys from nested dictionary.
    """
    flattened = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            flattened[key] = value
            continue

        value = flatten_dict(value)
        new_data = {
            f"{key}{level_separator}{nested_key}": nested_value
            for nested_key, nested_value in value.items()
        }
        flattened.update(new_data)

    return flattened


def write_nested_csv(
    data: List[dict], output_path: pathlib.Path, header: Optional[List[str]] = None,
):
    """Writes list of data as a nested csv.

    Args:
        data: list of data to write.
        output_path: Path of file to write to.
        header: Columns to output.  If specified, header will be in order of columns.
    """
    if not data:
        raise ValueError("Cannot upload a 0 length list.")

    first_row = data[0]
    first_row = {key: value for key, value in first_row.items()}

    all_columns = list(flatten_dict(first_row).keys())
    header = header or all_columns

    header_set = set(header)
    _logger.info(f"Writing to {output_path}")
    with output_path.open("w") as csvfile:
        writer = csv.DictWriter(csvfile, header)
        writer.writeheader()

        flattened_data = parallel_map(flatten_dict, data)
        flattened_rows = parallel_map(partial(_flatten_row, header_set), flattened_data)
        [writer.writerow(row) for row in flattened_rows]


def _flatten_row(header_set, flattened_row):
    # if a nested key is optional (i.e. {a: Optional[dict]}) and there is no
    # value for a, (i.e. {a: None}), don't write a, as it's not in the header.
    flattened_row = {k: v for k, v in flattened_row.items() if k in header_set}
    flattened_row = {k: v for k, v in flattened_row.items() if not pd.isnull(v)}
    flattened_row = {
        k: v.value if isinstance(v, enum.Enum) else v for k, v in flattened_row.items()
    }
    return flattened_row


def upload_json(key_name, json: str, output_dir: str):
    DatasetDeployer(f"{key_name}.json", json, output_dir).persist()


def deploy_shape_files(
    output_dir: str, key: str, shp_bytes: io.BytesIO, shx_bytes: io.BytesIO, dbf_bytes: io.BytesIO,
):
    """Deploys shape files to specified output dir.

    Args:
        output_dir: Output directory to save shapefiles to.
        key: stem of filename to save shapefiles to.
        shp_bytes:
        shx_bytes:
        dbf_bytes:
    """
    DatasetDeployer(key=f"{key}.shp", body=shp_bytes.getvalue(), output_dir=output_dir).persist()
    DatasetDeployer(key=f"{key}.shx", body=shx_bytes.getvalue(), output_dir=output_dir).persist()
    DatasetDeployer(key=f"{key}.dbf", body=dbf_bytes.getvalue(), output_dir=output_dir).persist()


def remove_root_wrapper(obj: dict):
    """Removes __root__ and replaces with __root__ value.

    When pydantic models are used to wrap lists this is done using a property __root__.
    When this is serialized using `model.json()`, it will return a json list. However,
    calling `model.dict()` will return a dictionary with a single key `__root__`.
    This function removes that __root__ key (and all sub pydantic models with a
    similar structure) to have a similar hierarchy to the json output.

    A dictionary {"__root__": []} will return [].

    Args:
        obj: pydantic model as dict.

    Returns: object with __root__ removed.
    """
    # Objects with __root__ should have it as the only key.
    if len(obj) == 1 and "__root__" in obj:
        return obj["__root__"]

    results = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            value = remove_root_wrapper(value)

        results[key] = value

    return results
