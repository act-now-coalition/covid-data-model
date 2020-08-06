# COVID-19 Data Model

*Objective:* Provide a library/API that ingests COVID-19 data to provide simulated outcomes based on local isolation/quarantine policy levers as represented in published models.

This work supports https://covidactnow.org/.

Check It Out via Jupyter:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/covid-projections/covid-data-model/master)


## Resources

We track our work via Trello. Our [engineering board](https://trello.com/b/zLf79HJ8/public-engpmux) is public.

Please file any issues via GitHub.

We are always looking for more collaborators. See our [contact page](https://covidactnow.org/contact) for general information or reach out directly about helping via this [form](https://docs.google.com/forms/d/e/1FAIpQLSfQkdwXsbDbwLHhWwBD6wzNiw54_0P6A60r8hujP3qnaxxFkA/viewform).

### Models

* [R-Based Model](https://alhill.shinyapps.io/COVID19seir/)
  * [Source Code](https://github.com/alsnhll/SEIR_COVID19)
* [Penn-Chime](http://penn-chime.phl.io/)
* [Imperial College Model](https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-NPI-modelling-16-03-2020.pdf)
* [Astor Sq ICU Beds Model](https://docs.google.com/spreadsheets/d/1DlC5kh9ve-Giv96XTnhCiB6vQAkQCjl5bDSjT68Q0FY/htmlview#)
* [Epidemic Calculator built by Open AI member](https://gabgoh.github.io/COVID/index.html)
* [Hopkins IDD Model](https://github.com/HopkinsIDD/COVIDScenarioPipeline)

### Data Sources
See [covid-data-public](https://github.com/covid-projections/covid-data-public) for data sources being used or considered.

Some code in the `covid-data-model` repo depends on there being a copy of the `covid-data-public` repo at
`../covid-data-public`.


## Setup

Detailed setup instructions can be found [here](./SETUP.md)

# API Snapshots
We automatically build & publish an API snapshot (e.g.
https://data.covidactnow.org/snapshot/123/) twice a day via a [github
action](./.github/workflows/daily_build.yml).

To manually kick off a new snapshot (e.g. from a feature branch), get a
[github personal access token](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line),
and run:

```bash
GITHUB_TOKEN=<YOUR PERSONAL GITHUB TOKEN> ./tools/build-snapshot.sh <branch (or commit sha, etc.)>
```

## Labeling Snapshots
Once a snapshot has been vetted and shipped to the production website, it should also be labeled
as our "latest" snapshot, which will allow our API consumers to pick it up.

The command to label snapshot 123 as latest (i.e. pointing https://data.covidactnow.org/latest/
at https://data.covidactnow.org/snapshot/123/) is:

```bash
GITHUB_TOKEN=<YOUR PERSONAL GITHUB TOKEN> ./tools/label-api.sh latest 123
```

# Development

# Sentry
In order to have sentry run locally and report errors to the dev sentry
instance, add the following to your .env

```
export SENTRY_DSN=https://<GET_SENTRY_DSN_FOR_DEV_INSTANCE>.ingest.sentry.io/<DEV_INSTANCE>
```

The gitub action pulls the sentry_dsn for the prod instance from a secrets stored within github.

# Downloading Model Run Data

You can download recent model data output from the github action:
```
export GITHUB_TOKEN=<YOUR PERSONAL GITHUB TOKEN>
./run.py utils download-model-artifact --run-number <optional run number>
```
By default it downloads the last run, but you can choose a specific run with `--run-number`

# PySEIR Setup

## Installation

Recommend virtualenv or miniconda python 3.7 from here
[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

If using conda, you can use the following:
- `conda create python=3.7 -n covid-data-model`
- `conda activate covid-data-model`
- `pip install -r requirements.txt -r requirements_test.txt`

### Running Models
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
