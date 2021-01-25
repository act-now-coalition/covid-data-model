"""
Data Overiew Builder generates a markdown file with information about data in the API.

By default this is shown on the docs site.

"""
from typing import List
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
    risk_levels: List[str]


CASES_FIELDS = Fields(
    ["cases", "newCases"],
    ["caseDensity", "infectionRate", "infectionRateCI90"],
    ["infectionRate", "caseDensity"],
)

DEATHS_FIELDS = Fields(["deaths"], [], [])

HOSPITALIZATIONS_FIELDS = Fields(
    ["icuBeds", "hospitalBeds"], ["icuCapacityRatio", "icuHeadroomRatio"], []
)

VACCINATION_FIELDS = Fields(
    ["vaccinesDistributed", "vaccinationsInitiated", "vaccinationsCompleted"],
    ["vaccinationsInitiatedRatio", "vaccinationsCompletedRatio"],
    [],
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
    Section("Deaths", DEATHS_FIELDS),
    Section("Hospitalizations", HOSPITALIZATIONS_FIELDS),
    Section("Vaccinations", VACCINATION_FIELDS),
]


def split_camel_case(text):
    return "".join(x if x.islower() else " " + x for x in text)


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
* JSON file fields: ``{field_prefix}.{field_name}``, ``{field_prefix}Timeseries.{field_name}``

"""
        return field_fmt

    actuals_fields = "\n".join(
        _format_field(Actuals, field_name, "actuals") for field_name in section.fields.actuals
    )
    metrics_fields = "\n".join(
        _format_field(Metrics, field_name, "metrics") for field_name in section.fields.metrics
    )
    risk_levels_fields = "\n".join(
        _format_field(RiskLevels, field_name, "riskLevels")
        for field_name in section.fields.risk_levels
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
title: Data
---

Read more about the data included in the Covid Act Now API.



{blocks}
"""
