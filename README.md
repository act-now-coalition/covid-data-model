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



### [Setup](./SETUP.md)


## Running


### Prep
If you've cloned a covid-data-public repo to a local source
```bash
export COVID_DATA_PUBLIC=../covid-data-public
```

```bash
mkdir -p results/test/
```

### Creating the Model
```bash
python run.py
```

### Creating a Local Model
```bash
python deploy_dod_dataset.py
```