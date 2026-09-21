.PHONY: main tDLGM util eval lint shampoo vae_baseline_tune
horizon = 50

_: main


main:
	python -m experiments.main --verbose --horizon 20 --learning_rate 0.0005 --latent_dim 64 --hidden_dim 512 --layers 3 --beta 1 --batch_size 128 --latent_dim 16

tune:
	python -m experiments.main --verbose --tune --horizon $(horizon)


vae_baseline_tune:
	python -m experiments.main --verbose --tune --horizon $(horizon) --vae-baseline

tune_reverse:
	python -m experiments.main --verbose --tune --horizon $(horizon) --reverse



upper_one:
	python -m experiments.main --verbose --upper --horizon $(horizon) --learning_rate 0.001122 --latent_dim 64 --hidden_dim 128 --layers 1 --beta 0.6590 --batch_size 128 





upper:
	python -m experiments.main --verbose --upper --horizon $(horizon) --learning_rate 0.001122 --latent_dim 64 --hidden_dim 128 --layers 3 --beta 0.6590 --batch_size 128 --skip_connection


upper_tune:
	python -m experiments.main --verbose --tune --upper --horizon $(horizon)

upper_tune_reverse:
	python -m experiments.main --verbose --tune --upper --horizon $(horizon) --reverse





lower:
	python -m experiments.main --verbose --lower --horizon $(horizon)


lower_tune:
	python -m experiments.main --verbose --tune --lower --horizon $(horizon)


vrnn:
	python -m experiments.main --verbose --vrnn --horizon $(horizon)

vrnn_tune:
	python -m experiments.main --verbose --vrnn --tune --horizon $(horizon)

baseline:
	python -m experiments.main --verbose --baseline  --horizon $(horizon) --hidden_dim 256 --batch_size 32 --layers 2 --learning_rate 0.00077

baseline_tune:
	python -m experiments.main --verbose --baseline --tune --horizon $(horizon)




basic:
	python -m experiments.main --verbose --basic  --horizon $(horizon)

basic_tune:
	python -m experiments.main --verbose --basic --tune --horizon $(horizon)


eval_tune:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260917-095823.pt

eval_tune_reverse:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260917-095921.pt

eval_upper:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260917-095307.pt



eval_reverse:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260917-095615.pt




eval_baseline:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260918-065843.pt

eval_baseline_long:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260918-081326.pt

eval_tune_long:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260918-140740.pt
