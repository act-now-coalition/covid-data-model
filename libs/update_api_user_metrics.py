from typing import Optional, List, Callable, Dict
import os
from libs import google_sheet_helpers
import time
import boto3
import datetime

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


def run_query(
    log_group: str,
    query,
    start_time: datetime.datetime,
    end_time: Optional[datetime.datetime] = None,
    field_transformations: Optional[Dict[str, Callable]] = None,
) -> List[dict]:
    field_transformations = field_transformations or {}
    client = boto3.client("logs")
    end_time = end_time if end_time else datetime.datetime.utcnow()
    start_query_response = client.start_query(
        logGroupName=log_group,
        startTime=int(start_time.timestamp()),
        endTime=int(end_time.timestamp()),
        queryString=query,
    )

    query_id = start_query_response["queryId"]

    response = None

    while response is None or response["status"] == "Running":
        print("Waiting for query to complete ...")
        time.sleep(1)
        response = client.get_query_results(queryId=query_id)

    if response["status"] != "Complete":
        raise Exception("ahhhh")

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


def run_users_query():
    log_group = os.environ["API_LOG_GROUP"]
    query = TOTAL_USERS_QUERY
    start_time = START_DATE
    to_date_string = lambda x: datetime.datetime.fromisoformat(x).date().isoformat()
    field_transformations = {
        "signupDate": to_date_string,
        "daysActive": int,
        "totalRequests": int,
        "latestDate": to_date_string,
    }
    return run_query(log_group, query, start_time, field_transformations=field_transformations)


def update_google_sheet(sheet, worksheet_name, data):

    worksheet = google_sheet_helpers.create_or_clear_worksheet(sheet, worksheet_name)

    header = list(data[0].keys())
    rows = [header]
    for result in data:
        rows.append([result[column] for column in header])

    worksheet.update(rows)
