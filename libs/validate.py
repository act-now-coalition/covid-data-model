from libs import build_params
import pathlib

UNSUPPORTED_REGIONS = ["AS", "GU", "MP"]


def validate_results(result_dir: str) -> None:
    """For each state, check that we have a file for each intervention,
    and that the file is non-empty

    Args:
        result_dir: Directory where model results are saved.
    """
    result_dir = pathlib.Path(result_dir)
    per_state_expected = len(build_params.INTERVENTIONS)
    missing_or_empty = []
    for state in build_params.US_STATE_ABBREV.values():
        if state in UNSUPPORTED_REGIONS:
            continue
        for i in range(per_state_expected):
            output_path = result_dir / f"{state}.{i}.json"
            # if the path doesn't exist or is empty, count as error.
            if not output_path.exists() or not output_path.read_bytes():
                missing_or_empty.append(str(output_path))

    if missing_or_empty:
        raise RuntimeError(
            f'Missing or empty expected files: {", ".join(missing_or_empty)}'
        )
