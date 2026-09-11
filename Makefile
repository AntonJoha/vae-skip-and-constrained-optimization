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

lower:
	python -m experiments.main --verbose --lower

lower_tune:
	python -m experiments.main --verbose --tune --lower



baseline:
	python -m experiments.main --verbose --baseline  

baseline_tune:
	python -m experiments.main --verbose --baseline --tune




eval_main:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260910-110534.pt
eval_upper:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260910-110537.pt

eval_baseline:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260910-135126.pt
