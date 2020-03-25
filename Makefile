
setup: requirements.txt
	pip install -r requirements.txt

setup-tests: requirements_test.txt
	pip install -r requirements_test.txt

test:
	pytest tests/

lint:
	pytest --pylint -m pylint --pylint-error-types=EF .

fmt:
	black covid_data_model/ tests/

.PHONY: setup setup-tests test lint fmt
