# Introduction  
This directory contains files that prepare CovidActNow forecast for submission to CDC model ensemble and backtest forecast
for how well the quantiles contains the observations. 
Ideally the preparation and validation of forecast runs weekly. 

# Files
* output_mapper.py: python module that transforms MLE inference and forecast of pyseir model
                    to the output ready to submit to CDC model ensemble.
* parameters.py: python module that contains all the parameters' definitions and default values needed 
                 by forecast submission.
* 
* plots.py: plot

# Running
To run the mapper, one can either run:   
`python output_mapper.py`  
Default date of forecast is 'today', to specify the date to start forecast, one can run:  
`python output_mapper.py --forecast_date <forecast_date>`   
where forecast_date should be a date string in format "%Y-%m-%d"

# Submitting Forecast
To submit forecast, clone the repo forked from the model ensemble repo (if you haven't yet):  
`git clone https://github.com/covid-projections/covid19-forecast-hub.git`

Also set the original repo as upstream so that the forked repo syncs with the origin master.  
`git remote add upstream https://github.com/reichlab/covid19-forecast-hub.git`

Checkout the branch `CovidActNow`, the remote branch already exits. 

Change to `cdc/report` folder copy the most recent forecast and metadata to folder:   
```
cp -t 
<directory_of_forked_repo>/covid19-forecast-hub/data-processed/CovidActNow-SEIR_CAN/
<forecast date>-CovidActNow-SEIR_CAN.csv metadata-CovidActNow-SEIR_CAN.txt
```
Where `forecast date` should be in `YYYY-MM-DD` format.

# Approach
## Forecast
(TBC)

# Data Visualization
To visualize the submitted forecast (and forecast from other models), change to the folder 
`<directory_of_forked_repo>/covid19-forecast-hub/data-processed/`, 
and run in R:  
```
source("explore_processed_data.R")
shinyApp(ui = ui, server = server)
```
ref: https://github.com/reichlab/covid19-forecast-hub/blob/master/data-processed/README.md#data-visualization