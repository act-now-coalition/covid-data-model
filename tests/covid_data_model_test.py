import pathlib
from covid_data_model import run


def test_run_simulation(tmpdir):
    test_interventions = run.INTERVENTIONS[:2]
    results = run.main(test_interventions, tmpdir, states=["MA"])
    tmpdir = pathlib.Path(str(tmpdir))
    generated_filenames = set([path.name for path in tmpdir.iterdir()])
    assert generated_filenames == set(["MA.1.json", "MA.0.json"])

    # TODO: Add actual test case data that doesn't change. Right now, this just
    # checks that intervention ran
    assert len(results["MA"]) == 2
