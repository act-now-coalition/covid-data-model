from datetime import datetime
from libs import update_api_user_metrics


def test_activity_summary_query(mocker):
    query_mock = mocker.patch.object(update_api_user_metrics, "_run_query")
    now = datetime.utcnow()
    example_boto_response = {
        "results": [
            [
                {"field": "signupDate", "value": now.isoformat()},
                {"field": "daysActive", "value": "10"},
            ]
        ]
    }
    query_mock.return_value = example_boto_response

    results = update_api_user_metrics.run_user_activity_summary_query("test_log_group")
    assert results == [{"signupDate": now.date().isoformat(), "daysActive": 10}]
