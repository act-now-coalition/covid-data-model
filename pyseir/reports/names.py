

compartment_to_name_map = {
    'S': 'Susceptible',
    'I': 'Infected',
    'E': 'Exposed',
    'A': 'Asymptomatic (Contagious)',
    'R': 'Recovered and Immune',
    'D': 'Direct Death',
    'HGen': 'Hospital Non-ICU',
    'HICU': 'Hospital ICU',
    'HVent': 'Hospital Ventilated',
    'deaths_from_hospital_bed_limits': 'Deaths: Non-ICU Capacity',
    'deaths_from_icu_bed_limits': 'Deaths: ICU Capacity',
    'deaths_from_ventilator_limits': 'Deaths: Ventilator Capacity',
    'total_deaths': 'Total Deaths (All Cause)',
    'HGen_cumulative': 'Cumulative Hospitalizations',
    'HICU_cumulative': 'Cumulative ICU',
    'HVent_cumulative': 'Cumulative Ventilators',
    'direct_deaths_per_day': 'Direct Deaths Per Day',
    'total_deaths_per_day': 'Total Deaths Per Day (All Cause)',
    'general_admissions_per_day': 'General Admissions Per Day',
    'icu_admissions_per_day': 'ICU Admissions Per Day',
    'total_new_infections': 'Total New Infections'
}


def policy_to_mitigation(s):
    """
    We have defined suppression as 1=unmitigated. For display we usually want
    mitigated = (1 - suppression). This converts that string repr.

    Parameters
    ----------
    s: str
        String to convert. Structure is e.g. suppression_policy__1.0

    Returns
    -------
    mitigation: str
        Mitigation display.
    """
    return f'{100 * (1 - float(s.split("__")[1])):.0f}% Mitigation'
