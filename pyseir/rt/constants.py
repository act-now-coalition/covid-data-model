from datetime import datetime

import numpy as np


class InferRtConstants:
    RNG_SEED = 42

    # Don't try to infer Rt for timeseries shorter than this
    MIN_TIMESERIES_LENGTH = 20

    # Settings for outlier removal
    LOCAL_LOOKBACK_WINDOW = 14
    Z_THRESHOLD = 10
    MIN_MEAN_TO_CONSIDER = 5

    # Window size used during smoothing of cases and deaths
    # Originally 14 but odd is better and larger avoids edges that drive R unrealistically
    # Size of the sliding Gaussian window to compute. Note that kernel std sets the width of the
    # kernel weight.
    COUNT_SMOOTHING_WINDOW_SIZE = 19
    COUNT_SMOOTHING_KERNEL_STD = 5

    # Sets the default value for sigma before adjustments
    # Recommend .03 (was .05 before when not adjusted) as adjustment moves up
    # Stdev of the process model. Increasing this allows for larger
    # instant deltas in R_t, shrinking it smooths things, but allows for
    # less rapid change. Can be interpreted as the std of the allowed
    # shift in R_t day-to-day.
    DEFAULT_PROCESS_SIGMA = 0.03

    # Scale sigma up as sqrt(SCALE_SIGMA_FROM_COUNT/current_count)
    # 5000 recommended
    SCALE_SIGMA_FROM_COUNT = 5000.0

    # Maximum increase (from DEFAULT_PROCESS_SIGMA) permitted for low counts
    # Recommend range 20. - 50. 30. appears to be best
    MAX_SCALING_OF_SIGMA = 30.0

    # Override min_cases and min_deaths with this value.
    # Recommend 1. - 5. range.
    # 1. is allowing some counties to run that shouldn't (unphysical results)
    MIN_COUNTS_TO_INFER = 5.0
    # TODO really understand whether the min_cases and/or min_deaths compares to max,
    #  avg, or day to day counts

    # Correct for tail suppression due to case smoothing window converting from centered to lagging
    # as approach current time
    TAIL_SUPPRESSION_CORRECTION = 0.75

    # Smooth RTeff (Rt_MAP_composite) to make less reactive in the short term while retaining long
    # term shape correctly.
    SMOOTH_RT_MAP_COMPOSITE = 1  # number of times to apply soothing
    RT_SMOOTHING_WINDOW_SIZE = 25  # also controls kernel_std

    # Minimum (half) width of confidence interval in composite Rt
    # Avoids too narrow values when averaging over timeseries that already have high confidence
    MIN_CONF_WIDTH = 0.1

    # Serial period = Incubation + 0.5 * Infections
    _sigma = 0.333  # Mirrored from assumptions doc
    _delta = 0.25  # Mirrored from assumptions doc
    SERIAL_PERIOD = 1 / _sigma + 0.5 * 1 / _delta

    # The quantization of the R Buckets
    R_BUCKETS = np.linspace(0, 10, 501).astype(float)

    # Reference date to compute from.
    REF_DATE = datetime(year=2020, month=1, day=1)

    # Confidence interval to compute. 0.95 would be 90% credible intervals from 5% to 95%.
    CONFIDENCE_INTERVALS = (0.68, 0.95)

    # The Possible Cross Correlation Shifts Allowed to Align Cases and Deaths
    XCOR_DAY_RANGE = range(-21, 5)
