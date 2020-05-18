# NYT Raw Data QA

*Objective:* Provide library that ingests CAN's cached NYT data from the current prod site and the current NYT dataset and compares. The locally cached version is also compared, but the when the next production would use the latest NYT dataset, so the metrics compare the oldest prod files to the latest NYT datasets.

## Running QA
`check_nyt_raw_data.py `will compare the locally cached version of `us-counties.csv` (in `/covid-data-public/data/nyt-cases/`), the latest NYT data, and the data used on the production site. Only areas where the production and latest NYT datasets differ enough to meet the abnormality requirements will be plotted. The difference between these two datasets are considered abnormal if there are `args.n_days_over_thres` where the case/death counts differ by at least 10% (`args.percent_threshold`). For the additional days in the latest NYT dataset (not present in the prod dataset) are considered abnormal, if for those days data the percent different between that day and the previous day is greater than `args.new_day_percent_thres`, which is by default set to 1.2. All of these settings may be changed.

The outputs will be sorted by if the area was abnormal due to historical data changes, new data changes, or both and will be put in the `args.output_dir` folder. All areas where the historical data changes will also produce metadata plots in the `args.output_dir` folder. The csvs used will also be stored in that folder for further investigation if needed.  

To run:
`python check_nyt_raw_data.py`
Happy hunting for anomalies!
