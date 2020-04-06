NO_INTERVENTION = 0
FLATTEN = 1
FULL_CONTAINMENT = 2
SOCIAL_DISTANCING = 3

INTERVENTIONS = [
    {
        'intervention_name': 'no_intervention', 
        'intervention_enum': NO_INTERVENTION,
    }, 
    {
        'intervention_name': 'flatten',
        'intervention_enum': FLATTEN,
    },
    {
        'intervention_name': 'full_containment',
        'intervention_enum': FULL_CONTAINMENT,
    },
    {
        'intervention_name': 'social_distancing',
        'intervention_enum': SOCIAL_DISTANCING
    }

]


# The Columns for the ouput jsons, note they get a plus one b/c
# pandas adds an index
INDEX = 0
DATE = 1
ALL_HOSPITALIZED = 9
ALL_INFECTED = 10
DEAD = 11
BEDS = 12
