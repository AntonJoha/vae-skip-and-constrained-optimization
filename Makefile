.PHONY: main tDLGM util eval lint shampoo

_: main


main:
	python -m experiments.main --verbose --horizon 20


tune:
	python -m experiments.main --verbose --tune --horizon 20



upper:
	python -m experiments.main --verbose --upper --horizon 20


upper_tune:
	python -m experiments.main --verbose --tune --upper --horizon 20


lower:
	python -m experiments.main --verbose --lower --horizon 20


lower_tune:
	python -m experiments.main --verbose --tune --lower --horizon 20




baseline:
	python -m experiments.main --verbose --baseline  --horizon 20

baseline_tune:
	python -m experiments.main --verbose --baseline --tune --horizon 20





eval_main:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260910-110534.pt
eval_upper:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260910-110537.pt

eval_baseline:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260910-135126.pt
