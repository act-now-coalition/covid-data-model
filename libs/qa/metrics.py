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
        """ Can be either a percent or a raw value cutoff. If it's a raw cut off we
        need to overwrite the method isAboveThreshold to handle correctly
        """
        pass

    @classmethod
    def diff(cls, value1, value2):
        return round(abs(value2 - value1), 4)

    @classmethod
    def threshold_diff_value(cls, value1, value2):
        average = (value2 + value1) / 2
        if average < 20:
            # small numbers changes are kind of confusing handle them a lil different
            return False
        return (abs(average - value1) / value1) * 100

    @classmethod
    def isAboveThreshold(cls, value1, value2):
        return cls.threshold_diff_value(value1, value2) > cls.threshold


class HospitalBedsRequiredTS(QAMetric):
    name = "HospitalBedsRequiredTS"
    projection_path = ["timeseries", "hospitalBedsRequired"]
    actual_path = None
    threshold = 10


class HospitalBedCapacityTS(QAMetric):
    name = "HospitalBedCapacityTS"
    projection_path = ["timeseries", "hospitalBedCapacity"]
    actual_path = None
    threshold = 10


class ICUBedCovidUsageTS(QAMetric):
    name = "ICUBedCovidUsageTS"
    projection_path = ["timeseries", "ICUBedsInUse"]
    actual_path = ["actualsTimeseries", "ICUBeds", "currentUsageCovid"]
    threshold = 10


class ICUBedTypicalUsageRateTS(QAMetric):
    name = "ICUBedTypicalUsageRateTS"
    projection_path = None
    actual_path = ["actualsTimeseries", "ICUBeds", "typicalUsageRate"]
    threshold = 10


class ICUBedTotalUsageTS(QAMetric):
    name = "ICUBedTotalUsageTS"
    projection_path = None
    actual_path = ["actualsTimeseries", "ICUBeds", "currentUsageTotal"]
    threshold = 10


class ICUBedTotalCapacityTS(QAMetric):
    name = "ICUBedTotalCapacityTS"
    projection_path = ["timeseries", "ICUBedCapacity"]
    actual_path = ["actualsTimeseries", "ICUBeds", "totalCapacity"]
    threshold = 10


class RtIndicatorTS(QAMetric):
    name = "RtIndicatorTS"
    projection_path = ["timeseries", "RtIndicator"]
    actual_path = None
    threshold = 0.1

    @classmethod
    def threshold_diff_value(cls, value1, value2):
        return cls.diff(value2, value1)


class RtIndicatorCI90TS(QAMetric):
    name = "RtIndicatorCI90TS"
    projection_path = ["timeseries", "RtIndicatorCI90"]
    actual_path = None
    threshold = 0.1

    @classmethod
    def threshold_diff_value(cls, value1, value2):
        return cls.diff(value2, value1)


class CumulativePositiveTestsTS(QAMetric):
    name = "CumulativePositiveTestsTS"
    projection_path = ["timeseries", "cumulativePositiveTests"]
    actual_path = ["actualsTimeseries", "cumulativePositiveTests"]
    threshold = 10


class CumulativeNegativeTestsTS(QAMetric):
    name = "CumulativeNegativeTestsTS"
    projection_path = ["timeseries", "cumulativeNegativeTests"]
    actual_path = ["actualsTimeseries", "cumulativeNegativeTests"]
    threshold = 10


class CumulativeInfectedTS(QAMetric):
    name = "CumulativeInfectedTS"
    projection_path = ["timeseries", "cumulativeInfected"]
    actual_path = None
    threshold = 10


class CumulativeDeathsTS(QAMetric):
    name = "CumulativeDeathsTS"
    projection_path = ["timeseries", "cumulativeDeaths"]
    actual_path = None
    threshold = 10


class CurrentTestPositiveRate(QAMetric):
    name = "CurrentTestPositiveRate"
    projection_path = None
    actual_path = ["actuals", "cumulativePositiveTests"]
    threshold = 10


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

    @classmethod
    def threshold_diff_value(cls, value1, value2):
        return cls.diff(value2, value1)

    @classmethod
    def diff(cls, value1, value2):
        value1_date = datetime.strptime(value1, "%Y-%m-%d")
        value2_date = datetime.strptime(value2, "%Y-%m-%d")

        delta = value1_date - value1_date
        return abs(delta.days)


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
]
