.PHONY: main tDLGM util eval lint shampoo vae_baseline_tune
horizon = 50

main:
	python -m experiments.main --verbose --horizon 50 --learning_rate 0.0001 --hidden_dim 512 --layers 3 --beta 1 --batch_size 64 --latent_dim 16

tune:
	python -m experiments.main --verbose --tune --horizon $(horizon)


vae_baseline_tune:
	python -m experiments.main --verbose --tune --horizon $(horizon) --vae-baseline

tune_reverse:
	python -m experiments.main --verbose --tune --horizon $(horizon) --reverse

upper_tune:
	python -m experiments.main --verbose --tune --upper --horizon $(horizon)

upper_tune_reverse:
	python -m experiments.main --verbose --tune --upper --horizon $(horizon) --reverse


lower_tune:
	python -m experiments.main --verbose --tune --lower --horizon $(horizon)


vrnn:
	python -m experiments.main --verbose --vrnn --horizon $(horizon)

vrnn_tune:
	python -m experiments.main --verbose --vrnn --tune --horizon $(horizon)

baseline_tune:
	python -m experiments.main --verbose --baseline --tune --horizon $(horizon)


basic_tune:
	python -m experiments.main --verbose --basic --tune --horizon $(horizon)



eval_tune:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260921-132124.pt

eval_upper_reverse:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260921-132050.pt

eval_upper:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260921-132103.pt

eval_reverse:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260921-132113.pt

eval_baseline:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260921-132137.pt

eval_vae:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260921-132144.pt

