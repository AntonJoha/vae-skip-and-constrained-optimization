.PHONY: main tDLGM util eval lint shampoo

_: main


main:
	python -m experiments.main --verbose

tune:
	python -m experiments.main --verbose --tune
