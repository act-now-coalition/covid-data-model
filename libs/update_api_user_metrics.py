from typing import Optional, List, Callable, Dict, Any
import time
import datetime
import logging

import gspread
import boto3

from libs import google_sheet_helpers

_logger = logging.getLogger(__name__)


TOTAL_USERS_QUERY = """
fields @timestamp, @message
| stats
    min(datefloor(@timestamp, 1d)) as signupDate,
    max(datefloor(@timestamp, 1d)) as latestDate,
    count_distinct(datefloor(@timestamp, 1d)) as daysActive,
    count() as totalRequests
  by email
| sort daysActive desc
"""


START_DATE = datetime.datetime(year=2020, month=9, day=15)


class CloudWatchQueryError(Exception):
    """Raised on a failed query to CloudWatch"""


def _run_query(
    log_group: str,
    query: str,
    start_time: datetime.datetime,
    end_time: Optional[datetime.datetime] = None,
) -> List[dict]:
    """Runs query on log group, applying transformations from `field_transformations`.

    Args:
        log_group: Name of CloudWatch log group.
        query: Query to run.
        start_time: Time to start query from.
        end_time: end time for query.  If not specified, uses current time.

    Returns: List of {<field_name>: <value>, ...} records.
    """
    client = boto3.client("logs")
    end_time = end_time if end_time else datetime.datetime.utcnow()
    start_query_response = client.start_query(
        logGroupName=log_group,
        startTime=int(start_time.timestamp()),
        endTime=int(end_time.timestamp()),
        queryString=query,
    )

    query_id = start_query_response["queryId"]

    response = client.get_query_results(queryId=query_id)

    while response["status"] == "Running":
        _logger.info("Waiting for query to complete ...")
        time.sleep(1)
        response = client.get_query_results(queryId=query_id)

    if response["status"] != "Complete":
        raise CloudWatchQueryError()

    return response


def _prepare_results(
    response, field_transformations: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    field_transformations = field_transformations or {}

    rows = []
    for result in response["results"]:
        row = {}
        for column in result:
            field = column["field"]
            value = column["value"]
            if field_transformations.get(field):
                value = field_transformations[field](value)

            row[field] = value

        rows.append(row)

    return rows


def run_user_activity_summary_query(log_group: str) -> List[Dict[str, Any]]:
    query = TOTAL_USERS_QUERY
    start_time = START_DATE
    # Transforms datestring to YYY-MM-DD format
    to_date_string = lambda x: datetime.datetime.fromisoformat(x).date().isoformat()

    # Field names match those in `TOTAL_USERS_QUERY` defined above.
    field_transformations = {
        "signupDate": to_date_string,
        "daysActive": int,
        "totalRequests": int,
        "latestDate": to_date_string,
    }

    results = _run_query(log_group, query, start_time)
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
