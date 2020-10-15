from typing import Optional, List, Callable, Dict, Any
import time
import datetime
import logging

import gspread
import boto3

from libs import google_sheet_helpers

_logger = logging.getLogger(__name__)


TOTAL_USERS_QUERY = """
SELECT MIN(DATE(timestamp)) AS "signupDate",
         MAX(timestamp) AS "latestDate",
         count(distinct date(timestamp)) AS "daysActive",
         count(*) AS "totalRequests",
         email
FROM default.{table}
GROUP BY email;
"""


ATHENA_BUCKET = "s3://covidactnow-athena-results"


class CloudWatchQueryError(Exception):
    """Raised on a failed query to CloudWatch"""


def _run_query(database: str, query: str,) -> List[dict]:
    """Runs athena query.

    Args:
        database: Name of Athena database.
        query: Query to run.

    Returns: List of {<field_name>: <value>, ...} records.
    """
    client = boto3.client("athena")
    start_query_response = client.start_query_execution(
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": ATHENA_BUCKET},
        QueryString=query,
    )

    query_id = start_query_response["QueryExecutionId"]

    response = client.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]
    completed_states = ["FAILED", "CANCELLED", "SUCCEEDED"]

    while response["Status"]["State"] not in completed_states:
        _logger.info("Waiting for query to complete ...")
        time.sleep(1)
        response = client.get_query_execution(QueryExecutionId=query_id)["QueryExecution"]

    if response["Status"]["State"] != "SUCCEEDED":
        raise CloudWatchQueryError()

    response_query_result = client.get_query_results(QueryExecutionId=query_id)

    # Parse query results
    rows = response_query_result["ResultSet"]["Rows"]
    rows = [row["Data"] for row in rows]
    # All items in each row has a dictionary of {"VarCharValue": <value>}.
    rows = [[item["VarCharValue"] for item in row] for row in rows]

    header = rows[0]

    data = rows[1:]

    results = []
    for row in data:
        record = {}
        for i, value in enumerate(header):
            record[value] = row[i]

        results.append(record)

    return results


def _prepare_results(
    rows, field_transformations: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    field_transformations = field_transformations or {}

    for row in rows:
        for field, value in row.items():

            if field_transformations.get(field):
                value = field_transformations[field](value)

            row[field] = value

    return rows


def run_user_activity_summary_query(table_name: str, database) -> List[Dict[str, Any]]:
    query = TOTAL_USERS_QUERY.format(table=table_name)

    # Transforms datestring to YYY-MM-DD format
    to_date_string = lambda x: datetime.datetime.fromisoformat(x).date().isoformat()

    # Field names match those in `TOTAL_USERS_QUERY` defined above.
    field_transformations = {
        "signupDate": to_date_string,
        "daysActive": int,
        "totalRequests": int,
        "latestDate": to_date_string,
    }

    results = _run_query(database, query)
    return _prepare_results(results, field_transformations=field_transformations)


def update_google_sheet(
    sheet: gspread.Spreadsheet, worksheet_name: str, data: List[Dict[str, Any]]
):
    """Updates Google Sheet with latest data.

    Args:
        sheet: Google Sheet to update
        worksheet_name: Name of worksheet to update.
        data: List of rows containing user activity.
    """
    worksheet = google_sheet_helpers.create_or_clear_worksheet(sheet, worksheet_name)
    header = list(data[0].keys())
    rows = [header]
    for result in data:
        rows.append([result[column] for column in header])

    # Setting raw=False allows Google Sheets to parse date strings as dates.
    worksheet.update(rows, raw=False)
