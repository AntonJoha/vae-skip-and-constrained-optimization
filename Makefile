.PHONY: main tDLGM util eval lint shampoo

_: main


main:
	python -m experiments.main --verbose --horizon 20 --learning_rate 0.001


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




basic:
	python -m experiments.main --verbose --basic  --horizon 20

basic_tune:
	python -m experiments.main --verbose --basic --tune --horizon 20

vrnn:
	python -m experiments.main --verbose --vrnn --horizon 20

vrnn_tune:
	python -m experiments.main --verbose --vrnn --tune --horizon 20


eval_main:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochfinal_20260914-223055.pt
eval_upper:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260914-102739.pt

eval_baseline:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260915-091405.pt
