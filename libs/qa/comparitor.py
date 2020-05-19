import csv
import requests
from datetime import datetime, timedelta
from sentry_sdk import capture_event

from libs.qa.metrics import CURRENT_METRICS, TIMESERIES_METRICS, QAMetric

from dataclasses import dataclass, asdict


@dataclass
class MetricDiff:
    state: str
    county: str
    fips: str
    date: str
    metric_name: str

    snapshot1: int
    value1: float

    snapshot2: int
    value2: float

    difference: float
    threshold: float

    def __lt__(self, other):
        return self.difference < other.difference

    def __str__(self):
        return f"{self.difference} for {self.metric_name} compared {self.snapshot1} to {self.snapshot2} for {self.state}."


class Comparitor(object):
    def __init__(self, snapshot1, snapshot2, api1, api2, state, intervention, fips=None):
        self.snapshot1 = snapshot1
        self.snapshot2 = snapshot2
        self.state = state
        self.fips = fips
        self.county = None
        self.intervention = intervention
        # TODO (sgoldblatt) swap out here if it's counties
        self.api1 = api1
        self.api2 = api2
        self._date_to_data = {}
        self._results = []
        self._generate_date_dictionary()

    def _generate_helper(self, full_data, path, snapshot):
        for data in full_data[path]:
            date_of_data = data["date"]

            current_data = self._date_to_data.get(date_of_data, {})
            current_data[snapshot] = current_data.get(snapshot, {})

            current_data[snapshot][path] = data
            self._date_to_data[date_of_data] = current_data

    def _generate_last_updated_helper(self, full_data, snapshot):
        current_data = self._date_to_data.get("current", {})
        current_data[snapshot] = current_data.get(snapshot, {})

        current_data[snapshot]["projections"] = full_data["projections"]
        current_data[snapshot]["actuals"] = full_data["actuals"]
        self._date_to_data["current"] = current_data

    def _generate_date_dictionary(self):
        """
            {"2020-01-27": {
                271: {
                    "actualTimeseries": {'population': None, 'intervention': 'STRONG_INTERVENTION', 'cumulativeConfirmedCases': 2, 'cumulativePositiveTests': None, 'cumulativeNegativeTests': None, 'cumulativeDeaths': None, 'hospitalBeds': {'capacity': None, 'totalCapacity': None, 'currentUsageCovid': None, 'currentUsageTotal': None, 'typicalUsageRate': None
                    }, 'ICUBeds': {'capacity': None, 'totalCapacity': None, 'currentUsageCovid': None, 'currentUsageTotal': None, 'typicalUsageRate': None
                    }, 'date': '2020-01-26'
                    }, 
                    ...
                    "timeseries": {...}
                }, 
                286: {},
            }}
        """
        for api, snapshot in [(self.api1, self.snapshot1), (self.api2, self.snapshot2)]:
            self._generate_helper(api, "actualsTimeseries", snapshot)
            self._generate_helper(api, "timeseries", snapshot)

        for api, snapshot in [(self.api1, self.snapshot1), (self.api2, self.snapshot2)]:
            self._generate_last_updated_helper(api, snapshot)

    def _getMetricValue(self, metric: QAMetric, date: str, snapshot: int, isProjected: bool):
        metric_path = metric.projection_path if isProjected else metric.actual_path
        if not metric_path:
            return None
        data_at_date = self._date_to_data.get(date)
        if not data_at_date:
            capture_event(
                {"state": self.state, "snapshot": snapshot, "metric": metric.name, "date": date}
            )
            return None
        data = data_at_date.get(snapshot)
        if not data:
            return None

        current_data = data
        for path_segment in metric_path:
            current_data = current_data.get(path_segment)
            if current_data is None:
                return None
        if isinstance(current_data, float):
            current_data = round(current_data, 4)
        return current_data

    def get_metric_projected_value(self, metric, date, snapshot):
        return self._getMetricValue(metric, date, snapshot, isProjected=True)

    def get_metric_actual_value(self, metric, date, snapshot):
        return self._getMetricValue(metric, date, snapshot, isProjected=False)

    def _get_min_and_max_date(self, array_with_dates):
        dates_only = [x["date"] for x in array_with_dates]
        return min(*dates_only), max(*dates_only)

    @property
    def dates(self):
        min_date = datetime.now() - timedelta(days=3)
        max_date = datetime.now() + timedelta(days=3)

        delta = max_date - min_date

        dates = []
        for i in range(delta.days + 1):
            day = min_date + timedelta(days=i)
            dates.append(day.strftime("%Y-%m-%d"))

        return dates

    def compareMetric(self, date, metric):
        actual_name = f"actual-{metric.name}"
        projection_name = f"projected-{metric.name}"

        projected_value1 = self.get_metric_projected_value(metric, date, self.snapshot1)
        projected_value2 = self.get_metric_projected_value(metric, date, self.snapshot2)

        actual_value1 = self.get_metric_actual_value(metric, date, self.snapshot1)
        actual_value2 = self.get_metric_actual_value(metric, date, self.snapshot2)

        if projected_value1 and projected_value2:
            if metric.isAboveThreshold(projected_value1, projected_value2):
                self._results.append(
                    MetricDiff(
                        self.state,
                        self.county,
                        self.fips,
                        date,
                        projection_name,
                        self.snapshot1,
                        projected_value1,
                        self.snapshot2,
                        projected_value2,
                        metric.diff(projected_value1, projected_value2),
                        metric.threshold,
                    )
                )

        if actual_value1 and actual_value2:
            if metric.isAboveThreshold(actual_value1, actual_value2):
                self._results.append(
                    MetricDiff(
                        self.state,
                        self.county,
                        self.fips,
                        date,
                        actual_name,
                        self.snapshot1,
                        actual_value1,
                        self.snapshot2,
                        actual_value2,
                        metric.diff(actual_value1, actual_value2),
                        metric.threshold,
                    )
                )
        return self._results

    def compareMetrics(self):
        for date in self.dates:
            for metric in TIMESERIES_METRICS:
                self.compareMetric(date, metric)
        for metric in CURRENT_METRICS:
            self.compareMetric("current", metric)
        return self._results

    def generate_report(self):
        unique_items = set([metric.metric_name for metric in self._results])
        unique_dates = set([metric.date for metric in self._results])
        differences = [metric.difference for metric in self._results]
        report = f"{self.state} has {len(self._results)} over threshold metrics ({len(unique_items)} unique metrics)."
        report += (
            f' Errors were found in: {", ".join(unique_items)} over the past 3 and next 3 days.'
        )
        max_difference = max(self._results)
        report += f" The biggest difference was {max_difference.difference} on {max_difference.date} for {max_difference.metric_name}"
        report += f"More detailed report can be found in a csv generated by this report."
        return report

    @classmethod
    def dict_results(cls, results):
        return [asdict(result) for result in results]
