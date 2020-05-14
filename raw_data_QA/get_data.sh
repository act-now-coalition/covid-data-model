#Remove old data directory if it exists and make new one
OUTPUT_DIR="data/"
rm -r $OUTPUT_DIR
mkdir $OUTPUT_DIR


#Extract git hash of current prod version on website
#Get json file from current production site
WEBSITE_PROD="https://raw.githubusercontent.com/covid-projections/covid-projections/master/src/assets/data/data_url.json"
PROD_VERSION_FILE="version.json"
PROD_FILE=$OUTPUT_DIR"prod.version"
HASH_FILE=$OUTPUT_DIR"hashinfo.txt"
curl $WEBSITE_PROD > $PROD_FILE
json_prod=$(grep -o 'http[^"]*' $PROD_FILE)$PROD_VERSION_FILE
curl $json_prod > $HASH_FILE
COVID_DATA_PUBLIC_MASTER_PROD=$(python get_hash.py $HASH_FILE)

#The current prod site does not have the counties csv counties cached so we will use old ones here
COVID_DATA_PUBLIC_MASTER_OLD="6f24dcb9a6b2203572c7509e506182be62117a38"
COVID_DATA_PUBLIC_MASTER_NOW="f2fb45e3e467fd3737865b3aef29e282b22f03d6"

#Get Data from those commit hashes that are cached in covid-data-public
BASE_URL="https://raw.githubusercontent.com/covid-projections/covid-data-public/"
COUNTY_FILE_PATH="/data/cases-nytimes/us-counties.csv"
OLD_DATA_URL=$BASE_URL$COVID_DATA_PUBLIC_MASTER_OLD$COUNTY_FILE_PATH
NEW_DATA_URL=$BASE_URL$COVID_DATA_PUBLIC_MASTER_NOW$COUNTY_FILE_PATH

#Get the latest data directly from NYT
LATEST_DATA_URL="https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
LATEST_DATA_STATE_URL="https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
echo $OLD_DATA_URL
curl -L $OLD_DATA_URL > data/us-counties-old.csv
curl -L $NEW_DATA_URL > data/us-counties-new.csv
curl -L $LATEST_DATA_URL > data/us-counties-latest.csv
curl -L $LATEST_DATA_STATE_URL > data/us-states-latest.csv

