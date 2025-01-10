> [!CAUTION]
> This page is no longer being updated due to a lack of reliable data. While we continue to surface this content for archival purposes, we recommend that you visit more regularly updated resources, such as from the [CDC](https://www.cdc.gov/coronavirus/2019-ncov/index.html).

# COVID-19 Data Pipeline

COVID data pipeline / API supporting https://covidactnow.org/.

It ingests data scraped via https://github.com/covid-projections/can-scrapers, combines it, calculates metrics, and generates data files for the [Covid Act Now API](https://apidocs.covidactnow.org/) and [website](https://covidactnow.org/)

# Development

## Setup

Detailed setup instructions can be found [here](./SETUP.md).

## Local development.
Normally the pipeline is run via github actions on a beefy cloud VM and still takes 2+ hours. When developing locally it is often useful to run the pipeline on a subset of locations ond/or to skip pipeline steps.

To run the pipeline end-to-end but only generate data for Connecticut state / counties, you can run:

```
# Fetches latest scraped data from can-scrapers and combines all data sources
# into a combined dataset, runs all filters, etc.  Adding --no-refresh-datasets
# will make this much faster but skips fetching / combining latest datasets.
./run.py data update --states CT

# Runs the pyseir code to generate the infection rate, i.e. r(t) metric data for locations.
python ./pyseir/cli.py build-all --states=CT

# Runs the API generation
./run.py api generate-api-v2 --state CT output -o output/api

# API files are generated to output/api.
```

### Downloading Model Run Data

If you just want to run the API generation you can skip the first two steps above by downloading the pyseir model results from a previous snapshot. You can download the pyseir model output from a recent github action run with:
```
export GITHUB_TOKEN=<YOUR PERSONAL GITHUB TOKEN>
./run.py utils download-model-artifact
```
By default it downloads the last run, but you can choose a specific run with `--run-number`

## Running PySEIR
PySEIR provides a command line interface in the activated environment. You can access the model with `pyseir --help ` and `pyseir <subcommand> --help` providing more information.

Example:
`pyseir build-all --states="NY"` will run state and county models for New York.
States can also be specified by their state code: `--states="New York"` and `--states=NY` are equivalent.


`pyseir build-all --states=NY --fips=36061` will run the New York state model and the model for the specified
FIPS code (in this case New York City).


Check the `output/` folder for results.

### Model Output

There are a variety of output artifacts to paths described in pyseir/utils.py.
The main artifact is the ensemble_result which contains the output information
for each `suppression policy -> model compartment` as well as capacity
information.


## API Documentation

We host an API documentation site available in [api/docs](api/docs).  It is a static site built using [Docusaurus 2](https://v2.docusaurus.io/).

Additionally, we define the API output using [pydantic](https://pydantic-docs.helpmanual.io)
schemas and generate Open API specs (default output [api/docs/open_api_schema.json](api/docs/open_api_schema.json) and json-schema outputs (default output [api/schemas_v2/](api/schemas_v2).

When modifying the API schema, run `./run.py api update-schemas` to update the schemas.

### Simple setup

Build environment:
```bash
$ cd api/docs
$ yarn
```

Start server locally:
```bash
$ cd api/docs
$ yarn start
```

Deploy update to [apidocs.covidactnow.org](https://apidocs.covidactnow.org):
```bash
$ tools/deploy-docs.sh
```
