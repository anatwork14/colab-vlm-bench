.PHONY: install test lint plan validate check
install:
	python -m pip install -e '.[dev]'
test:
	python -m pytest -q
lint:
	python -m ruff check --select E4,E7,E9,F821 src tests
plan:
	python -m vlmbench.cli plan --config configs/a100_campaign.json
validate:
	python -m vlmbench.cli validate --config configs/a100_campaign.json
check: test lint plan
