from abc import ABC, abstractmethod

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
        return cls.diff(value1, value2) >= cls.threshold; 

class HospitalBedsRequired(QAMetric):
    name = "HospitalBedsRequired"
    projection_path = ["timeseries", "hospitalBedsRequired"]
    actual_path = None
    threshold = 1

class HospitalBedCapacity(QAMetric):
    name = "HospitalBedCapacity"
    projection_path = ["timeseries", "hospitalBedCapacity"]
    actual_path = None
    threshold = 1

class ICUBedCovidUsage(QAMetric):
    name = "ICUBedCovidUsage"
    projection_path = ["timeseries", "ICUBedsInUse"]
    actual_path = ["actualsTimeseries", "ICUBeds", "currentUsageCovid"]
    threshold = 1 

class ICUBedTotalCapacity(QAMetric):
    name = "ICUBedTotalCapacity"
    projection_path = None
    actual_path = ["actualsTimeseries", "ICUBeds", "totalCapacity"]
    threshold = 1

METRICS = [HospitalBedCapacity, HospitalBedsRequired, ICUBedCovidUsage, ICUBedTotalCapacity]

"""
timeseries_projection_hospitalBedsRequired
timeseries_projection_hospitalBedCapacity
Acutalstimeseries_icubeds_currentUsageTotal
Actualstimeseries_icubeds_totalCapacity
Actualstimeseries_icubeds_currentUsageCovid
Actualstimeseries_icubeds_typicalUsageRate
timeseries_projection_hospitalBedsRequired
timeseries_projection_hospitalBedCapacity
timeseries_projection_ICUBedsInUse
timeseries_projection_ICUBedCapacity
timeseries_projection_RtIndicator
timeseries_projection_RtIndicatorCI90
timeseries_projection_cumulativePositiveTests
timeseries_projection_cumulativeNegativeTests
Projection.rt
projection.currentIcuUtilization
projection.currentTestPositiveRate
projections_totalHospitalBeds_shortageStartDate, 
timeseries_projection_cumulativeInfected, 
actuals_totalPopulation, 
timeseries_projection_cumulativeDeaths
"""