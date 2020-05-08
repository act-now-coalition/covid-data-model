# COVID-19 Data Model

*Objective:* Provide a library/API that ingests COVID-19 data to provide simulated outcomes based on local isolation/quarantine policy levers as represented in published models.

Check It Out in Jupyter
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/covid-projections/covid-data-model/master)


## Resources

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


## [Setup](./SETUP.md)

# API Snapshots

We automatically build & publish an API snapshot (e.g. https://data.covidactnow.org/snapshot/123/) 
twice a day via a [github action](./.github/workflows/deploy_api.yml).  To manually kick off a new
snapshot, get a
[personal access token](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line),
and run:

```bash
export GITHUB_TOKEN=<YOUR PERSONAL GITHUB TOKEN>
./tools/push-api.sh
```

Once a snapshot has been vetted, you can "label" it with a friendly name, e.g. pointing https://data.covidactnow.org/v0/ at https://data.covidactnow.org/snapshot/123/ with:
```bash
export GITHUB_TOKEN=<YOUR PERSONAL GITHUB TOKEN>
./tools/label-api.sh v0 123
```

# Development

### Run website data deploy

This will run all models and generate data needed for the website, outputting to ``../covid-projections/public/data``.
```bash
./deploy_website.sh
```

### Run DoD dataset deploy
Run all models, generate DoD datasets, output to `./dod_results/` folder:
```bash
./deploy_dod.sh
```

Run all models, generate DoD datasets, upload the files to S3 bucket specified by `BUCKET_NAME`.
```bash
BUCKET_NAME=<bucket name> ./deploy_dod.sh
```

If you've previously ran the full model, you can just re-run the DoD dataset part via:
```
# Output artifacts locally to dod_results/:
python deploy_dod_dataset.py
# Upload artifacts to S3 bucket:
BUCKET_NAME=<bucket name> python deploy_dod_dataset.py
```

# Sentry
In order to have sentry run locally and report errors to the dev sentry
instance, add the following to your .env

```
export SENTRY_DSN=https://<GET_SENTRY_DSN_FOR_DEV_INSTANCE>.ingest.sentry.io/<DEV_INSTANCE>
```

The gitub action pulls the sentry_dsn for the prod instance from a secrets stored within github. 

# [NEW 4/7] PySEIR Setup

## Installation

Recommend virtualenv or miniconda python 3.7 from here
[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

Execute
`pip install -r requirements.txt -r requirements_test.txt`

If using conda, activate the environment here..
`conda activate pyseir`


### Running Models
Example here. You can remove the `--state` flag to run everything. To run only
states, add `--states-only`. `pyseir run-all --state="California"`

`pyseir --help ` and `pyseir <subcommand> --help` also provide specific flags. 

Check the `output/` folder for results.

The flag `--states-only` will skip counties and the flat `--state` will only run
a single state. These can be combined if you want to run things quickly.

### Model Output

There are a variety of output artifacts to paths described in pyseir/utils.py.
The main artifact is the ensemble_result which contains the output information
for each `suppression policy -> model compartment` as well as capacity
information. It is described in 
[this running document](https://docs.google.com/document/d/1U0zTP_jjwp8i-hCj3jPosfeyj74vKqTy6ERFYzYgcvE/edit?usp=sharing).