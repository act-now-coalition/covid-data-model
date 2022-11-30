from typing import Optional, List, Callable, Dict, Any
import os
import time
import datetime
from datetime import timedelta
import logging
import requests
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
GROUP BY email
"""


ATHENA_BUCKET = "s3://covidactnow-athena-results"
HUBSPOT_AUTH_TOKEN = os.getenv("HUBSPOT_AUTH_TOKEN")


class CloudWatchQueryError(Exception):
    """Raised on a failed query to CloudWatch"""


def update_hubspot_activity(email, latest_active_at, days_active):
    """Updates Hubspot contact with latest activity dates."""

    # Hubspot date field should be at UTC midnight. When the %z directive is provided to the
    # strptime() method, a TZ aware datetime object will be produced.
    date = datetime.datetime.strptime(latest_active_at + " +0000", "%Y-%m-%d %z")
    url = f"https://api.hubapi.com/contacts/v1/contact/createOrUpdate/email/{email}"

    response = requests.post(
        url,
        headers={"Authorization": f"Bearer {HUBSPOT_AUTH_TOKEN}"},
        json={
            "email": email,
            "properties": [
                {"property": "last_api_request_at", "value": int(date.timestamp()) * 1000},
                {"property": "api_days_active", "value": days_active},
            ],
        },
    )
    if not response.ok:
        _logger.warning(f"Failed to update hubspot activity for {email}")
        return
    else:
        _logger.info(f"Updated hubspot activity for {email} with status {response.status_code}")

    _logger.info(f"Successfully updated {email}")


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

    results_paginator = client.get_paginator("get_query_results")
    results_iter = results_paginator.paginate(
        QueryExecutionId=query_id, PaginationConfig={"PageSize": 1000}
    )

    results = []
    header = None
    for results_page in results_iter:
        # Parse query results
        rows = results_page["ResultSet"]["Rows"]
        rows = [row["Data"] for row in rows]
        # All items in each row has a dictionary of {"VarCharValue": <value>}.
        rows = [[item["VarCharValue"] for item in row] for row in rows]

        # For the first page, set header
        if not header:
            header = rows[0]
            data = rows[1:]
        else:
            # For pages without header, use all rows
            data = rows

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


def update_hubspot_users(data: List[Dict[str, Any]], only_update_recent: bool = True):
    """Updates hubspot users with usage activity.

    Args:
        data: List of query results.
        only_update_recent: If True only updates users with usage in the past 2 days.
    """
    if not HUBSPOT_AUTH_TOKEN:
        _logger.warning("Hubspot API key not provided, skipping hubspot update")
        return

    recent_activity_date = datetime.datetime.now() - timedelta(days=2)
    for row in data:
        last_activity_date = datetime.datetime.strptime(row["latestDate"], "%Y-%m-%d")
        if only_update_recent and last_activity_date < recent_activity_date:
            _logger.info(f"{row['email']} has no recent usage, skipping")
            continue

        update_hubspot_activity(row["email"], row["latestDate"], row["daysActive"])
