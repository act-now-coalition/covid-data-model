# NYT Raw Data QA

*Objective:* Provide library that ingests CAN's cached NYT data from the current prod site and the current NYT dataset and compares.

## Running QA
`check_nyt_raw_data.py `will compare the locally cached version of `us-counties.csv` that is locally cached in `/covid-data-public/data/nyt-cases/`, the latest from NYT, and what is currently being used on the production site. Only areas in the production and latest NYT datasets that meet the abnormality requirements will be plotted. For the case and death data stored in both the prod and latest datasets, datasets are considered abnormal if there are `args.n_days_over_thres` where the datasets differ by at least 10% (`args.percent_threshold`). For the additional days in the latest NYT dataset (not present in the prod dataset) are considered abnormal, if for those days data the percent different between that day and the previous day is greater than `args.new_day_percent_thres`, which is by default set to 1.2. All of these settings may be changed.

The outputs will be sorted by if the area was abnormal due to historical data changes, new data changes, or both and will be put in the `args.output_dir` folder. All areas where the historical data changes will also produce metadata plots in the `args.output_dir` folder. The csvs used will also be stored in that folder for further investigation if needed.  

To run:
`python check_nyt_raw_data.py`
Happy hunting for anomalies!
