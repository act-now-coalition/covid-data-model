from abc import ABC, abstractmethod
from datetime import datetime


class QAMetric(ABC):
    def __init__(self):
        # ...
        pass

    @property
    @abstractmethod
    def projection_path(self):
        pass

    @property
    @abstractmethod
    def actual_path(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def threshold(self):
        pass

    @classmethod
    def diff(cls, value1, value2):
        return abs(value2 - value1)

    @classmethod
    def isAboveThreshold(cls, value1, value2):
        return cls.diff(value1, value2) > cls.threshold


class HospitalBedsRequiredTS(QAMetric):
    name = "HospitalBedsRequiredTS"
    projection_path = ["timeseries", "hospitalBedsRequired"]
    actual_path = None
    threshold = 1


class HospitalBedCapacityTS(QAMetric):
    name = "HospitalBedCapacityTS"
    projection_path = ["timeseries", "hospitalBedCapacity"]
    actual_path = None
    threshold = 1


class ICUBedCovidUsageTS(QAMetric):
    name = "ICUBedCovidUsageTS"
    projection_path = ["timeseries", "ICUBedsInUse"]
    actual_path = ["actualsTimeseries", "ICUBeds", "currentUsageCovid"]
    threshold = 1


class ICUBedTypicalUsageRateTS(QAMetric):
    name = "ICUBedTypicalUsageRateTS"
    projection_path = None
    actual_path = ["actualsTimeseries", "ICUBeds", "typicalUsageRate"]
    threshold = 1


class ICUBedTotalUsageTS(QAMetric):
    name = "ICUBedTotalUsageTS"
    projection_path = None
    actual_path = ["actualsTimeseries", "ICUBeds", "currentUsageTotal"]
    threshold = 1


class ICUBedTotalCapacityTS(QAMetric):
    name = "ICUBedTotalCapacityTS"
    projection_path = ["timeseries", "ICUBedCapacity"]
    actual_path = ["actualsTimeseries", "ICUBeds", "totalCapacity"]
    threshold = 1


class RtIndicatorTS(QAMetric):
    name = "RtIndicatorTS"
    projection_path = ["timeseries", "RtIndicator"]
    actual_path = None
    threshold = 0.1


class RtIndicatorCI90TS(QAMetric):
    name = "RtIndicatorCI90TS"
    projection_path = ["timeseries", "RtIndicatorCI90"]
    actual_path = None
    threshold = 0.1


class CumulativePositiveTestsTS(QAMetric):
    name = "CumulativePositiveTestsTS"
    projection_path = ["timeseries", "cumulativePositiveTests"]
    actual_path = ["actualsTimeseries", "cumulativePositiveTests"]
    threshold = 1


class CumulativeNegativeTestsTS(QAMetric):
    name = "CumulativeNegativeTestsTS"
    projection_path = ["timeseries", "cumulativeNegativeTests"]
    actual_path = ["actualsTimeseries", "cumulativeNegativeTests"]
    threshold = 1


class CumulativeInfectedTS(QAMetric):
    name = "CumulativeInfectedTS"
    projection_path = ["timeseries", "cumulativeInfected"]
    actual_path = None
    threshold = 1


class CumulativeDeathsTS(QAMetric):
    name = "CumulativeDeathsTS"
    projection_path = ["timeseries", "cumulativeDeaths"]
    actual_path = None
    threshold = 1


class CurrentTestPositiveRate(QAMetric):
    name = "CurrentTestPositiveRate"
    projection_path = None
    actual_path = ["actuals", "cumulativePositiveTests"]
    threshold = 1


class Population(QAMetric):
    name = "Population"
    projection_path = None
    actual_path = ["actuals", "population"]
    threshold = 0


class CurrentRT(QAMetric):
    name = "rt"
    projection_path = ["projections", "Rt"]
    actual_path = None
    threshold = 0.1


class HospitalShortageStartDate(QAMetric):
    name = "HospitalShortageStartDate"
    projection_path = ["projections", "totalHospitalBeds", "shortageStartDate"]
    actual_path = None
    threshold = 1

    def diff(cls, value1, value2):
        value1_date = datetime.strptime(value1, "%Y-%m-%d")
        value2_date = datetime.strptime(value2, "%Y-%m-%d")

        delta = value1_date - value1_date
        return abs(delta.day)


CURRENT_METRICS = [
    CurrentTestPositiveRate,
    Population,
    CurrentRT,
    HospitalShortageStartDate,
]

TIMESERIES_METRICS = [
    HospitalBedsRequiredTS,
    HospitalBedCapacityTS,
    ICUBedCovidUsageTS,
    ICUBedTypicalUsageRateTS,
    ICUBedTotalUsageTS,
    ICUBedTotalCapacityTS,
    RtIndicatorTS,
    RtIndicatorCI90TS,
    CumulativePositiveTestsTS,
    CumulativeNegativeTestsTS,
    CumulativeInfectedTS,
    CumulativeDeathsTS,
]
