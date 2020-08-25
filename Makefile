.PHONY: setup-dev test unittest lint fmt

setup-dev: requirements.txt requirements_test.txt
	pip install --upgrade -r requirements.txt -r requirements_test.txt
	pre-commit install

unittest:
	pytest -n 2 test/

lint:
	pytest --pylint -m pylint --pylint-jobs=2 .

# Run unittests then linting
test: unittest lint

fmt:
	black .
