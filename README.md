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

## Running

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


# [NEW 4/7] PySEIR Setup

## Installation

Install miniconda python 3.7 from here [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

Execute
`conda env create -f environment.yaml`

Activate the environment here..
`conda activate pyseir`

### Installing pyseir
Change to into the county_covid_seir_models directory
`pip install -e .`


### Running Models
`pyseir run-all --state=California`

This will take a few minutes to download today's data, run inference and model
ensembles, and generate the output. Then check the `output/` folder for results.
