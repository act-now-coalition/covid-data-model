from datapublic.common_fields import DemographicBucket

from tests import test_helpers
import pytest
from libs.pipeline import Region
from datapublic.common_fields import CommonFields

from libs.datasets import ca_vaccination_backfill


def test_ca_county_vaccination_assertion():

    region_la = Region.from_fips("06037")

    input_data = {
        region_la: {
            CommonFields.VACCINES_ADMINISTERED: [100, 200],
            CommonFields.VACCINATIONS_INITIATED: [50, 150],
        },
    }
    dataset = test_helpers.build_dataset(input_data)

    with pytest.raises(AssertionError):
        dataset_out = ca_vaccination_backfill.derive_ca_county_vaccine_pct(dataset)


def test_ca_county_vaccination_calculation():

    region_ca = Region.from_state("CA")
    region_la = Region.from_fips("06037")
    region_mo = Region.from_state("MO")

    input_data = {
        region_ca: {
            CommonFields.VACCINES_ADMINISTERED: [100, 200],
            CommonFields.VACCINATIONS_INITIATED: [50, 150],
            CommonFields.VACCINATIONS_COMPLETED: [50, 50],
        },
        region_la: {CommonFields.VACCINES_ADMINISTERED: [100, 200]},
        region_mo: {CommonFields.VACCINATIONS_COMPLETED_PCT: [25.0, 28.0]},
    }
    static_by_region_then_field_name = {region_la: {CommonFields.POPULATION: 1000}}
    dataset = test_helpers.build_dataset(
        input_data, static_by_region_then_field_name=static_by_region_then_field_name
    )

    dataset_out = ca_vaccination_backfill.derive_ca_county_vaccine_pct(dataset)
    expected_data_input = {
        **input_data,
        region_la: {
            CommonFields.VACCINES_ADMINISTERED: [100, 200],
            CommonFields.VACCINATIONS_INITIATED_PCT: [5, 15],
            CommonFields.VACCINATIONS_COMPLETED_PCT: [5, 5],
        },
    }

    expected_dataset = test_helpers.build_dataset(
        expected_data_input, static_by_region_then_field_name=static_by_region_then_field_name
    )

    test_helpers.assert_dataset_like(dataset_out, expected_dataset, drop_na_dates=True)


def test_ca_county_vaccination_calculation_by_bucket():

    region_ca = Region.from_state("CA")
    region_la = Region.from_fips("06037")
    region_mo = Region.from_state("MO")
    bucket_40s = DemographicBucket("age:40-49")
    bucket_all = DemographicBucket.ALL
    la_administered_distributions = {bucket_all: [100, 200], bucket_40s: [1, 2]}

    input_data = {
        region_ca: {
            CommonFields.VACCINES_ADMINISTERED: [100, 200],
            CommonFields.VACCINATIONS_INITIATED: [50, 150],
            CommonFields.VACCINATIONS_COMPLETED: [50, 50],
        },
        region_la: {CommonFields.VACCINES_ADMINISTERED: la_administered_distributions},
        region_mo: {CommonFields.VACCINATIONS_COMPLETED_PCT: [25.0, 28.0]},
    }
    static_by_region_then_field_name = {
        region_la: {CommonFields.POPULATION: 1000},
    }
    dataset = test_helpers.build_dataset(
        input_data, static_by_region_then_field_name=static_by_region_then_field_name
    )

    dataset_out = ca_vaccination_backfill.derive_ca_county_vaccine_pct(dataset)
    expected_data_input = {
        **input_data,
        region_la: {
            CommonFields.VACCINES_ADMINISTERED: la_administered_distributions,
            CommonFields.VACCINATIONS_INITIATED_PCT: [5, 15],
            CommonFields.VACCINATIONS_COMPLETED_PCT: [5, 5],
        },
    }

    expected_dataset = test_helpers.build_dataset(
        expected_data_input, static_by_region_then_field_name=static_by_region_then_field_name
    )

    test_helpers.assert_dataset_like(dataset_out, expected_dataset, drop_na_dates=True)
