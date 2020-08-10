import tabulate
import pydantic
import api
import pathlib


README_TEMPLATE = "####api_descriptions####"


def _model_from_ref(ref: str, all_schemas):
    _, key, value = ref.split("/")
    return all_schemas[key][value]


def _generate_property_list(properties, all_schemas):
    rows = []
    for name, prop in properties.items():
        title = None
        if "$ref" in prop:
            ref_model = _model_from_ref(prop["$ref"], all_schemas)
            obj_type = ref_model["title"]
            row = {
                "name": name,
                # TODO: Make this a link
                "type": f"[{obj_type}](#{obj_type})",
                "description": ref_model.get("description", ""),
            }
        else:
            row = {
                "name": name,
                "type": prop.get("type"),
                "description": prop.get("description", ""),
            }
        rows.append(row)
    return rows


def schema_to_md(schema, all_schemas):
    title = schema["title"]
    description = schema["description"]
    rows = []

    if schema.get("type") == "array":

        ref_model = _model_from_ref(schema["items"]["$ref"], all_schemas)
        return f"""
### {title}
{description}

List of [{ref_model['title']}](#{ref_model['title']})
        """
    table_rows = _generate_property_list(schema["properties"], all_schemas)
    keys = {key: key for key in table_rows[0]}
    table_output = tabulate.tabulate(table_rows, keys, tablefmt="github")
    return f"""

### {title}
{description}

{table_output}

"""


def generate_markdown_for_schema(schema: dict):

    all_text = ""
    for sch in schema["definitions"].values():
        all_text += schema_to_md(sch, schema) or ""

    return all_text


def replace_schemas_template(
    input_path: pathlib.Path, output_path: pathlib.Path, text, template=README_TEMPLATE
):

    input_text = input_path.read_text()
    input_text = input_text.replace(template, text)
    output_path.write_text(input_text)
