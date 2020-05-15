# NYT Raw Data QA

*Objective:* Provide library that ingests CAN's cached NYT data from the current prod site and the current NYT dataset and compares.

## Running QA
get_data.sh and get_hash.py together grab the cached data from the current prod site and the current NYT data and store in the data folder. There are extra settings where you can grab data from a specific covid-data-public git commit as well.

check_nyt_raw_data.py will compare up to three NYT us-counties.csv files. Only areas that meet the abnormality requirements will be plotted. Currently, it is required that there are 5 days where the datasets differ by at least 10% for old data, and the new data must be greater than new_day_percent_thres = 1.2. All of these setting may be changed.

The outputs will be sorted by if the area was abnormal due to historical data changes, new data changes, or both and will be put in the args.output_dir folder. All areas where the historical data changes will also produce meta data plots in the args.output_dir folder. 

You can run run.sh to download the data and process it.

Happy hunting for anomalies!
