

class CommonFields(object):
    AGGREGATE_LEVEL = "aggregate_level"
    COUNTRY = "country"
    COUNTY = "county"
    STATE = "state"
    FIPS = "fips"

    @classmethod
    def all_fields(cls):
        # TODO: make this automatic so fields don't get left out.
        return [
            cls.AGGREGATE_LEVEL, cls.COUNTRY, cls.COUNTY, cls.STATE, cls.FIPS
        ]

    @classmethod
    def verify_df_has_all_columns(cls, data: pd.DataFrame):
        missing = []
        for field in cls.all_fields():
            if not field in data.columns:
                missing.append(field)

        if missing:
            raise ValueError(
                f"Missing fields: {missing}. Please make sure all "
                "field names are standardized."
            )


def standardize_common_fields_data(data: pd.DataFrame):

    CommonFields.verify_df_has_all_columns(data)

    # Convert all country fields to 3 letter iso

    # Convert all state fields to 2 letter iso

    # Get all fips codes where possible
