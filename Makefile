.PHONY: setup-dev test unittest lint fmt

setup-dev: requirements.txt requirements_test.txt
	pip install --upgrade -r requirements.txt -r requirements_test.txt
	pre-commit install

# As a work-around for a test failure in tests/libs/datasets/data_source_test.py::test_state_providers_smoke_test
# run tests on a single process. The failure seems to happen on the server (which runs unittest) but not locally.
unittest:
	pytest -n 1 tests/

unittest-not-slow:
	pytest -k 'not slow' -n 2 --durations=5 tests/

lint:
	pytest --pylint -m pylint --pylint-jobs=2 .

# Run unittests then linting
test: unittest lint

test-fast: lint unittest-not-slow

fmt:
	black .
