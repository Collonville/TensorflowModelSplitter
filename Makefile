install:
	poetry install

format:
	poetry run black .
	poetry run isort .

run:
	poetry run python -m tensorflowmodelsplitter
	