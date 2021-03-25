from libs.datasets import demographics
import pandas as pd


def test_distribution_bucket_str():
    bucket = demographics.DistributionBucket("distribution", "value")
    assert str(bucket) == "distribution:value"

    bucket = demographics.DistributionBucket("all", None)
    assert str(bucket) == "all"


def test_distribution_bucket_from_str():
    bucket = demographics.DistributionBucket.from_str("distribution:value")
    expected_bucket = demographics.DistributionBucket("distribution", "value")
    assert bucket == expected_bucket

    bucket = demographics.DistributionBucket.from_str("all")
    expected_bucket = demographics.DistributionBucket("all", None)
    assert bucket == expected_bucket


def test_distribution_bucket_from_row():
    row = pd.Series({"toppings": "cheese", "crust": "thin", "sauce": "tomato"})
    bucket = demographics.DistributionBucket.make_from_row(["toppings", "crust"], row)
    expected_bucket = demographics.DistributionBucket("toppings;crust", "cheese;thin")
    assert bucket == expected_bucket
