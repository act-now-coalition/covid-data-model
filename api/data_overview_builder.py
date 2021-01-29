"""
Data Overiew Builder generates a markdown file with information about data in the API.

By default this is shown on the docs site.

"""
from typing import List
import re
import dataclasses

import pydantic

from api import can_api_v2_definition

Actuals = can_api_v2_definition.Actuals
Metrics = can_api_v2_definition.Metrics
RiskLevels = can_api_v2_definition.RiskLevels


@dataclasses.dataclass
class Fields:
    actuals: List[str]
    metrics: List[str]


CASES_FIELDS = Fields(["cases", "newCases"], ["caseDensity", "infectionRate", "infectionRateCI90"],)

DEATHS_FIELDS = Fields(["deaths"], [])

HOSPITALIZATIONS_FIELDS = Fields(
    ["icuBeds", "hospitalBeds"], ["icuCapacityRatio", "icuHeadroomRatio"],
)

VACCINATION_FIELDS = Fields(
    ["vaccinesDistributed", "vaccinationsInitiated", "vaccinationsCompleted"],
    ["vaccinationsInitiatedRatio", "vaccinationsCompletedRatio"],
)


TESTING_FIELDS = Fields(
    ["positiveTests", "negativeTests"], ["testPositivityRatio", "testPositivityRatioDetails"],
)


@dataclasses.dataclass
class Section:

    name: str
    fields: List[pydantic.Field]
    description: str = ""


CASE_DESCRIPTION = """


"""
SECTIONS = [
    Section("Cases", CASES_FIELDS),
    Section("Tests", TESTING_FIELDS),
    Section("Hospitalizations", HOSPITALIZATIONS_FIELDS),
    Section("Vaccinations", VACCINATION_FIELDS),
    Section("Deaths", DEATHS_FIELDS),
]


def split_camel_case(text):
    # Camel case split function from https://stackoverflow.com/a/37697078
    return " ".join(re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", text)).split())


def indent(text, indent=2):
    lines = text.split("\n")
    lines = [" " * indent + line for line in lines]
    return "\n".join(lines)


def build_section_block(section: Section):
    fields = []

    def _format_field(cls, field_name, field_prefix):
        field = cls.__fields__[field_name]
        field_title = split_camel_case(field.name).title()
        field_name = field.name
        description = field.field_info.description or ""
        description = description.rstrip("\n").lstrip("\n")
        space = "  "
        field_fmt = f"""### {field_title}

  {description}

**Where to access**{space}
* CSV column names: ``{field_prefix}.{field_name}``
* JSON file fields: ``{field_prefix}.{field_name}``, ``{field_prefix}Timeseries.*.{field_name}``
"""
        return field_fmt

    actuals_fields = "\n".join(
        _format_field(Actuals, field_name, "actuals") for field_name in section.fields.actuals
    )
    metrics_fields = "\n".join(
        _format_field(Metrics, field_name, "metrics") for field_name in section.fields.metrics
    )

    template = f"""
## {section.name}

{actuals_fields}
{metrics_fields}
"""
    return template


def build_sections(sections: List[Section] = SECTIONS):
    blocks = "\n".join(build_section_block(section) for section in sections)

    return f"""---
id: data
title: Data Definitions
---

Read more about the data included in the Covid Act Now API.

{blocks}
"""
