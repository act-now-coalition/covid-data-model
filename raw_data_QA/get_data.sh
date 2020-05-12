rm -r data
mkdir data
curl -L https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv > data/us-counties-latest.csv
curl -L https://raw.githubusercontent.com/nytimes/covid-19-data/1a710fb5ece55b8d5e36724353366b72387b1978/us-counties.csv > data/us-counties-may10.csv
curl -L https://raw.githubusercontent.com/nytimes/covid-19-data/621fb652b003f049c1c3bc3985160da9db4fcf6d/us-counties.csv > data/us-counties-may8.csv
curl -L https://raw.githubusercontent.com/nytimes/covid-19-data/b14a9b8dd14114cf2ecb4a943c7c5051ac94212a/us-counties.csv > data/us-counties-may7.csv
curl -L https://raw.githubusercontent.com/nytimes/covid-19-data/509ada74d6cafdffa60199ac0a8ff55ffd7c69f7/us-counties.csv > data/us-counties-may9.csv
curl -L https://raw.githubusercontent.com/nytimes/covid-19-data/cdce95431148dc195a1079cb1841f8b9279dd3e7/us-counties.csv > data/us-counties-may6.csv
