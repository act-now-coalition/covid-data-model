import csv
import requests
from datetime import datetime, timedelta

from dictdiffer import diff

from qa.metrics import METRICS

class MetricDiff(object):
    state = None
    county = None
    fips = None
    date = None
    metric_name = None

    snapshot1 = None
    snapshot2 = None

    value1 = None
    value2 = None

    threshold = None
    difference = None

    def __init__(self, state, county, fips, date, metric_name, snapshot1, snapshot2, value1, value2, difference, threshold):
        self.state  = state
        self.county = county
        self.fips = fips
        self.date = date
        self.metric_name = metric_name
        self.snapshot1 = snapshot1
        self.snapshot2 = snapshot2
        self.value1 = value1
        self.value2 = value2
        self.threshold = threshold
        self.difference = difference


class Comparitor(object): 
    snapshot1 = None
    snapshot2 = None

    state = None
    fips = None

    api1 = None
    api2 = None

    _date_to_data = {}
    _results = {}
    
    def _get_state_data(self, snapshot1, snapshot2, state_abbrev, intervention): 
        # counties_url_1 = f'https://data.covidactnow.org/snapshot/{snapshot1}/us/counties.{intervention}.timeseries.json'
        # counties_url_2= f'https://data.covidactnow.org/snapshot/{snapshot2}/us/counties.{intervention}.timeseries.json'

        states_url_1 = f'https://data.covidactnow.org/snapshot/{snapshot1}/us/states/{state_abbrev}.{intervention}.timeseries.json'
        states_url_2 = f'https://data.covidactnow.org/snapshot/{snapshot2}/us/states/{state_abbrev}.{intervention}.timeseries.json'

        # counties_json_1 = requests.get(counties_url_1).json()
        # counties_json_2 = requests.get(counties_url_2).json()

        states_json_1 = requests.get(states_url_1).json()
        states_json_2 = requests.get(states_url_2).json()

        return states_json_1, states_json_2

    def __init__(self, snapshot1, snapshot2, state, intervention, fips = None):
        self.snapshot1 = snapshot1
        self.snapshot2 = snapshot2
        self.state = state
        self.fips = fips
        self.intervention = intervention
        # TODO (sgoldblatt) swap out here if it's counties
        self.api1, self.api2 = self._get_state_data(snapshot1, snapshot2, state, intervention)
        self._generate_date_dictionary()

    def _generate_helper(self, series, path, snapshot): 
        for data in series: 
            date_of_data = data["date"]
            
            current_data = self._date_to_data.get(date_of_data, {})
            current_data[path] = current_data.get(path, {})
            current_data[path][snapshot] = data
            self._date_to_data[date_of_data] = current_data

    def _generate_date_dictionary(self):
        self._generate_helper(self.api1["actualsTimeseries"], "actualsTimeseries", self.snapshot1)
        self._generate_helper(self.api2["actualsTimeseries"], "actualsTimeseries", self.snapshot2)
        self._generate_helper(self.api1["timeseries"], "timeseries", self.snapshot1)
        self._generate_helper(self.api2["timeseries"], "timeseries", self.snapshot2)

    def getMetricProjectedValue(self, metric, date, snapshot): 
        timeseries = self._date_to_data[date].get("timeseries", None) 
        if not timeseries:
            return None
        return timeseries[snapshot][metric.projection_name]

    def getMetricActualValue(self, metric, date, snapshot):
        actuals_timeseries = self._date_to_data.get("actualTimeseries")
        if not actuals_timeseries: 
            return None
        return actuals_timeseries[snapshot][metric.actual_name]

    def _get_min_and_max_date(self, array_with_dates):
        dates_only = [x["date"] for x in array_with_dates]
        return min(*dates_only), max(*dates_only)

    def getDates(self):
        timeseries1_min_date, timeseries1_max_date = self._get_min_and_max_date(self.api2["timeseries"])
        timeseries2_min_date, timeseries2_max_date = self._get_min_and_max_date(self.api2["timeseries"])

        actual1_min_date, actual1_max_date = self._get_min_and_max_date(self.api1["actualsTimeseries"])
        actual2_min_date, actual2_max_date= self._get_min_and_max_date(self.api2["actualsTimeseries"])

        min_date_string = min(timeseries1_min_date, timeseries2_min_date, actual1_min_date, actual2_min_date)
        max_date_string = max(timeseries1_max_date, timeseries2_max_date, actual1_max_date, actual2_max_date)

        min_date = datetime.strptime(min_date_string, "%Y-%m-%d")
        max_date = datetime.strptime(max_date_string, "%Y-%m-%d")

        delta = max_date - min_date

        dates = []
        for i in range(delta.days + 1): 
            day = min_date + timedelta(days=i)
            dates.append(day.strftime("%Y-%m-%d"))
        
        return dates
    
    def compareMetrics(self):
        for date in self.getDates(): 
            for metric in METRICS: 
                actual_name = metric.actual_path[-1] if metric.actual_path else None
                projection_name = metric.projection_path[-1] if metric.projection_path else None
                
                projected_value1 = self.getMetricProjectedValue(metric, date, self.snapshot1)
                projected_value2 = self.getMetricProjectedValue(metric, date, self.snapshot2)

                actual_value1 = self.getMetricActualValue(metric, date, self.snapshot1)
                actual_value2 = self.getMetricActualValue(metric, date, self.snapshot2)

                if (projected_value1 and projected_value2): 
                    if metric.isAboveThreshold(projected_value1, projected_value2): 
                        self._results[projection_name] = MetricDiff(
                            self.state, 
                            self.county, 
                            self.fips, 
                            date, 
                            projection_name, 
                            self.snapshot1, 
                            self.snapshot2, 
                            projected_value1, 
                            projected_value2, 
                            metric.diff(actual_value1, actual_value2), 
                            metric.threshold)

                if (actual_value1 and actual_value2): 
                    if metric.isAboveThreshold(actual_value1, actual_value2): 
                        self._results[actual_name] = MetricDiff(
                            self.state, 
                            self.county, 
                            self.fips, 
                            date, 
                            actual_name, 
                            self.snapshot1, 
                            self.snapshot2, 
                            actual_value1, 
                            actual_value2, 
                            metric.diff(actual_value1, actual_value2), 
                            metric.threshold)
        return self.results
            
    @property
    def results(self): 
        return self._results.values()

    def write_to_csv(self, file_name):
        with open(file_name, 'w+') as csv_file: 
            write = csv.writer(csv_file) 
            write.writerows(self.results)  
