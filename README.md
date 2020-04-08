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

### Create JSON for UI
```bash
python run.py
```

### Create JSON for API
Creating the files for local inspection.
```bash
python deploy_dod_dataset.py
```

Creating and uploading the files, where the `BUCKET_NAME` is name of the s3 bucket hosting these files.
```bash
BUCKET_NAME=<bucket name> python deploy_sir_dataset.py
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
