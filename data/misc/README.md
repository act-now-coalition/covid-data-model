Misc. data sets.

* hospital_beds_by_county.csv (source TBD)
* populations.csv (source TBD)
* interventions.json (manually curated)
* state.txt (State names, abbreviations and FIPS from [census.gov](https://www2.census.gov/geo/docs/reference/state.txt))

# Source data for MSAs

This directory contains a copy of data related to metropolitan statistical areas (MSAs).

The MSAs of the 1990s have led to the more modern Core-Based Statistical Areas and
consolidated metropolitan statistical areas. For more information see
https://www.census.gov/topics/housing/housing-patterns/about/core-based-statistical-areas.html

The source file `list1_2020.xls` is from the
[Census.gov Delineation Files](https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html).
It is parsed by code landing in the covid-data-model repo at `libs/datasets/statistical_areas.py`.

