import pandas as pd
import numpy as np
import datetime
import pprint

# @TODO: Attempt today. If that fails, attempt yesterday.
latest = datetime.date.today() - datetime.timedelta(days=1)

url = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_daily_reports/{}.csv'.format(
    latest.strftime("%m-%d-%Y"))
raw_df = pd.read_csv(url)
raw_df.info()

column_mapping = {"Province_State": "Province/State",
                  "Country_Region": "Country/Region",
                  "Last_Update": "Last Update",
                  "Lat": "Latitude",
                  "Long_": "Longitude",
                  "Combined_Key": "Combined Key",
                  "Admin2": "County",
                  "FIPS": "State/County FIPS Code"
                  }

remapped_df = raw_df.rename(columns=column_mapping)
remapped_df.info()


cols = ["Province/State",
        "Country/Region",
        "Last Update",
        "Latitude",
        "Longitude",
        "Confirmed",
        "Recovered",
        "Deaths",
        "Active",
        "County",
        "State/County FIPS Code",
        "Combined Key",
        "Incident Rate",
        "People Tested",
        "Shape"
        ]

final_df = pd.DataFrame(remapped_df, columns=cols)
final_df["Shape"] = "Point"
final_df['Last Update'] = pd.to_datetime(final_df['Last Update'])
final_df['Last Update'] = final_df['Last Update'].dt.strftime(
    '%-m/%-d/%Y %H:%M')

final_df = final_df.fillna("<Null>")

us_only = final_df[(final_df["Country/Region"] == "US")]

pprint.pprint(us_only.head())

us_only.to_csv("results/counties.csv")
