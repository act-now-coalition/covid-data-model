from libs import pipeline
from libs.datasets import AggregationLevel


def test_location_id_to_level():
    param_result_map = {
        "iso1:us#fips:99111": AggregationLevel.COUNTY,
        "iso1:us#iso2:us-wy#fips:56039": AggregationLevel.COUNTY,
        "iso1:us#iso2:us-tx": AggregationLevel.STATE,
        "iso1:us": AggregationLevel.COUNTRY,
        "iso1:us#cbsa:10100": AggregationLevel.CBSA,
    }
    assert set(param_result_map.values()) == set(AggregationLevel)

    for location_id, level in param_result_map.items():
        assert pipeline.location_id_to_level(location_id) == level
        assert pipeline.Region.from_location_id(location_id).level == level
