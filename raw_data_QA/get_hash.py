import json, sys

if __name__ == "__main__":
    inputfile = sys.argv[1]
    f = open(inputfile)
    data = json.load(f)
    print(data["covid-data-public"]["hash"])
