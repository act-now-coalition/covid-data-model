from datetime import datetime
from libs import update_api_user_metrics


def test_activity_summary_query(mocker):
    query_mock = mocker.patch.object(update_api_user_metrics, "_run_query")
    now = datetime.utcnow()
    example_rows = [
        {"signupDate": now.isoformat(), "daysActive": "10"},
    ]
    query_mock.return_value = example_rows

    results = update_api_user_metrics.run_user_activity_summary_query("table", "database")
    assert results == [{"signupDate": now.date().isoformat(), "daysActive": 10}]
