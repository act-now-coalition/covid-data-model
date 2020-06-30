from covidactnow.datapublic.common_fields import CommonFields


class CommonIndexFields:
    # Column for FIPS code. Right now a column containing fips data may be
    # county fips (a length 5 string) or state fips (a length 2 string).
    FIPS = CommonFields.FIPS

    # 2 letter state abbreviation, i.e. MA
    STATE = CommonFields.STATE

    COUNTRY = CommonFields.COUNTRY

    AGGREGATE_LEVEL = CommonFields.AGGREGATE_LEVEL

    DATE = CommonFields.DATE
