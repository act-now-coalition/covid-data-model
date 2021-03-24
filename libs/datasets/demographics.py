from typing import Optional, List, NewType, Dict
import dataclasses

import pandas as pd


# Key for ScalarDistributions are just bucket names, such as "10-19" in a distribution for age.
ScalarDistribution = NewType("Distribution", Dict[str, float])


@dataclasses.dataclass(frozen=True)
class DistributionBucket:
    """Represents a single bucket contained in a distribution."""

    distribution: str
    name: Optional[str]

    def __str__(self):
        if self.name:
            return f"{self.distribution}:{self.name}"
        else:
            # Generally in the case of "all"
            return f"{self.distribution}"

    @staticmethod
    def from_str(short_name: str) -> "DistributionBucket":
        """Creates distribution bucket from string representation."""
        demographic_buckets = short_name.split(":")
        if demographic_buckets == ["all"]:
            return DistributionBucket("all", None)

        assert len(demographic_buckets) == 2
        distribution, bucket = demographic_buckets
        return DistributionBucket(distribution, bucket)

    @staticmethod
    def make_from_row(demographic_fields: List[str], row: pd.Series) -> "DistributionBucket":
        """Creates a bucket from a row of data with demographic fields."""
        non_all = []
        for field in demographic_fields:
            if row[field] != "all":
                non_all.append((field, row[field]))

        if non_all:
            fields, values = zip(*non_all)
            distribution_name = ";".join(fields)
            bucket = ";".join(values)
            return DistributionBucket(distribution_name, bucket)
        else:
            return DistributionBucket("all", None)
