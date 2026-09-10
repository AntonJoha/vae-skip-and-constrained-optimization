.PHONY: main tDLGM util eval lint shampoo

_: main


main:
	python -m experiments.main --verbose

tune:
	python -m experiments.main --verbose --tune


upper:
	python -m experiments.main --verbose --upper

upper_tune:
	python -m experiments.main --verbose --tune --upper



baseline:
	python -m experiments.main --verbose --baseline  

baseline_tune:
	python -m experiments.main --verbose --baseline --tune