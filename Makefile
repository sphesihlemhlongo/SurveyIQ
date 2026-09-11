.PHONY: install test eval run bundle bundle-blank clean

install:
	pip install -r requirements.txt langchain-anthropic

test:
	python -m pytest -v tests/

eval:
	python -m src.evaluation

bundle:
	python bundle.py

bundle-blank:
	python bundle.py --blank-tradeoffs

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f sphesihle-anthony-mhlongo-takehome.zip
