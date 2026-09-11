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
	python bundle.py --clean

