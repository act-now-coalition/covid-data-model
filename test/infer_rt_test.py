import pathlib

import pytest
import pandas as pd
import structlog

from libs.datasets import combined_datasets
from covidactnow.datapublic.common_fields import CommonFields
from libs import pipeline
from libs.datasets import timeseries
from pyseir import cli

from pyseir.rt import utils
from pyseir.rt import infer_rt
from test.mocks.inference import load_data
from test.mocks.inference.load_data import RateChange


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error", "ignore::libs.pipeline.BadFipsWarning")


def test_replace_outliers_on_last_day():
    x = pd.Series([10, 10, 10, 500], [0, 1, 2, 3])

    results = utils.replace_outliers(x, structlog.getLogger(), local_lookback_window=3)

    expected = pd.Series([10, 10, 10, 10], [0, 1, 2, 3])
    pd.testing.assert_series_equal(results, expected)


"""
Tests of Rt inference code using synthetically generated data for 100 days where the following are
specified:
1) The starting count of cases (scale)
2) One or two Rate changes - each with
    2a) time at which the change occurs (first should have t0=0)
    2b) The Rt value with which to generate growing (Rt>1) or decaying (Rt<1) values

Note that smoothing of values smears out the transitions +/- window_size/2 days
"""

FAILURE_ERROR_FRACTION = 0.2

# output directory where test artifacts are saved.
TEST_OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "output" / "test_results"


def run_individual(
    fips: str,
    spec: load_data.DataSpec,
    display_name: str,
    output_dir: pathlib.Path = TEST_OUTPUT_DIR,
):
    output_dir.mkdir(exist_ok=True)

    # TODO fails below if deaths not present even if not using
    data_generator = load_data.DataGenerator(spec)
    input_df = load_data.create_synthetic_df(data_generator)
    regional_input = infer_rt.RegionalInput.from_fips(fips)

    # Now apply smoothing and filtering
    collector = {}
    smoothed_df = infer_rt.filter_and_smooth_input_data(
        df=input_df,
        region=regional_input.region,
        include_deaths=False,
        figure_collector=collector,
        log=structlog.getLogger(),
    )

    engine = infer_rt.RtInferenceEngine(
        data=smoothed_df,
        display_name=display_name,
        regional_input=regional_input,
        figure_collector=collector,
    )  # Still Needed to Pipe Output For Now
    output_df = engine.infer_all()

    # output all figures
    for (key, fig) in collector.items():
        plot_path = output_dir / f"{display_name}__fips_{fips}__{key}.pdf"
        fig.savefig(plot_path, bbox_inches="tight")

    rt = output_df["Rt_MAP_composite"].values
    t_switch = spec.ratechange2.t0
    rt1 = spec.ratechange1.reff
    rt2 = spec.ratechange2.reff
    return (rt1, rt2, t_switch, rt)


def check_standard_assertions(rt1, rt2, t_switch, rt):
    OFFSET = 15
    # Check expected values are within 10%
    if abs(rt1 - 1.0) > 0.05:
        assert (
            pytest.approx(rt[t_switch - OFFSET] - 1.0, rel=FAILURE_ERROR_FRACTION) == rt1 - 1.0
        )  # settle into first rate change
    else:
        assert abs(rt[t_switch - OFFSET] - rt1) < 0.1  # settle into first rate change
    assert (
        pytest.approx(rt[t_switch + OFFSET] - 1.0, rel=FAILURE_ERROR_FRACTION) == rt2 - 1.0
    )  # settle into 2nd rate change

    assert (
        pytest.approx(rt[-1] - 1.0, rel=FAILURE_ERROR_FRACTION * 2) == rt2 - 1.0
    ), f"Test {id} Failed: Today Value Not Within Spec: Predicted={round(rt[-1],2)} Observed={rt2}."


@pytest.mark.slow
def test_constant_cases_high_count(tmp_path):
    """Track constant cases (R=1) at low count"""
    data_spec = load_data.DataSpec(
        generator_type=load_data.DataGeneratorType.EXP,
        disable_deaths=True,
        scale=1000.0,
        ratechange1=RateChange(0, 1.0),
        ratechange2=RateChange(80, 1.5),  # To avoid plotting issues
    )
    rt1, rt2, t_switch, rt = run_individual(
        "20", data_spec, "test_constant_cases_high_count", output_dir=tmp_path  # Kansas
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


@pytest.mark.slow
def test_med_scale_strong_growth_and_decay(tmp_path):
    """Track cases growing strongly and then decaying strongly"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "36",  # New York
        load_data.DataSpec(
            generator_type=load_data.DataGeneratorType.EXP,
            disable_deaths=True,
            scale=100.0,
            ratechange1=RateChange(0, 1.5),
            ratechange2=RateChange(50, 0.7),
        ),
        "test_med_scale_strong_growth_and_decay",
        output_dir=tmp_path,
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


@pytest.mark.skip(reason="From Alex: Test is failing rt = .84 instead of rt1")
@pytest.mark.slow
def test_low_cases_weak_growth(tmp_path):
    """Track with low scale (count = 5) and slow growth"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "50",  # Vermont
        load_data.DataSpec(
            generator_type=load_data.DataGeneratorType.EXP,
            disable_deaths=True,
            scale=5.0,
            ratechange1=RateChange(0, 1.0),
            ratechange2=RateChange(70, 1.2),
        ),
        "test_low_cases_weak_growth",
        output_dir=tmp_path,
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


@pytest.mark.slow
def test_high_scale_late_growth(tmp_path):
    """Track decaying from high initial count to low number then strong growth"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "02",  # Alaska
        load_data.DataSpec(
            generator_type=load_data.DataGeneratorType.EXP,
            disable_deaths=True,
            scale=2000.0,
            ratechange1=RateChange(0, 0.95),
            ratechange2=RateChange(70, 1.5),
        ),
        "test_high_scale_late_growth",
        output_dir=tmp_path,
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


@pytest.mark.slow
def test_low_scale_two_decays(tmp_path):
    """Track low scale decay at two different rates"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "06",  # California
        load_data.DataSpec(
            generator_type=load_data.DataGeneratorType.EXP,
            disable_deaths=True,
            scale=50.0,
            ratechange1=RateChange(0, 0.9),
            ratechange2=RateChange(50, 0.7),
        ),
        "test_low_scale_two_decays",
        output_dir=tmp_path,
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


@pytest.mark.slow
def test_smoothing_and_causality(tmp_path):
    run_individual(
        "56",  # Wyoming
        load_data.DataSpec(
            generator_type=load_data.DataGeneratorType.EXP,
            disable_deaths=True,
            scale=1000.0,
            ratechange1=RateChange(0, 1.0),
            ratechange2=RateChange(95, 5.0),
        ),
        "test_smoothing_and_causality",
        output_dir=tmp_path,
    )


@pytest.mark.slow
def test_generate_infection_rate_metric_one_empty():
    FIPS = [
        "51017",  # Bath County VA Almost No Cases. Will be filtered out under any thresholds.
        "51153",  # Prince William VA Lots of Cases
    ]
    regions = [infer_rt.RegionalInput.from_fips(region) for region in FIPS]

    df = pd.concat(infer_rt.run_rt(input) for input in regions)
    returned_fips = df.fips.unique()
    assert "51153" in returned_fips
    assert "51017" not in returned_fips


@pytest.mark.slow
def test_generate_infection_rate_metric_two_aggregate_levels():
    FIPS = ["06", "06075"]  # CA  # San Francisco, CA
    regions = [infer_rt.RegionalInput.from_fips(region) for region in FIPS]

    df = pd.concat(infer_rt.run_rt(input) for input in regions)
    returned_fips = df.fips.unique()
    assert "06" in returned_fips
    assert "06075" in returned_fips


@pytest.mark.slow
def test_generate_infection_rate_new_orleans_patch():
    FIPS = ["22", "22051", "22071"]  # LA, Jefferson and Orleans
    regions = [infer_rt.RegionalInput.from_fips(region) for region in FIPS]

    df = pd.concat(infer_rt.run_rt(input) for input in regions)
    returned_fips = df.fips.unique()
    assert "22" in returned_fips
    assert "22051" in returned_fips
    assert "22071" in returned_fips
    assert not df[CommonFields.DATE].isna().any()
    assert not df[CommonFields.FIPS].isna().any()


def test_generate_infection_rate_metric_fake_fips():
    with structlog.testing.capture_logs() as logs:
        # TX Misc Fips Holder timeseries not found in combined data
        infer_input = infer_rt.RegionalInput.from_fips("48999")
    assert [l["event"] for l in logs] == ["Creating OneRegionTimeseriesDataset with zero regions"]
    assert infer_input.timeseries.empty

    with pytest.raises(timeseries.RegionLatestNotFound):
        # Totally bogus FIPS not even in latest data raises an exception
        infer_rt.RegionalInput.from_fips("48998")


@pytest.mark.xfail(raises=ValueError)
def test_generate_infection_rate_with_nans():
    # Ma Counties is currently failing with a ValueError due to recent period of non-reporting
    FIPS = ["25001"]  # MA lots of NaN
    regions = [infer_rt.RegionalInput.from_fips(region) for region in FIPS]
    df = pd.concat(infer_rt.run_rt(input) for input in regions)
    returned_fips = df.fips.unique()
    assert "25001" in returned_fips


@pytest.mark.slow
def test_patch_substatepipeline_nola_infection_rate():
    nola_fips = [
        "22051",  # Jefferson
        "22071",  # Orleans
    ]
    pipelines = []
    for fips in nola_fips:
        region = pipeline.Region.from_fips(fips)
        infection_rate_df = infer_rt.run_rt(infer_rt.RegionalInput.from_region(region))
        pipelines.append(
            cli.SubStatePipeline(
                region=region,
                infer_df=infection_rate_df,
                _combined_data=combined_datasets.RegionalData.from_region(region),
            )
        )

    patched = cli._patch_substatepipeline_nola_infection_rate(pipelines)

    df = pd.concat(p.infer_df for p in patched)
    returned_fips = df.fips.unique()
    assert "22051" in returned_fips
    assert "51017" not in returned_fips
