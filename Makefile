.PHONY: main tDLGM util eval lint shampoo vae_baseline_tune
horizon = 20

main:
	python -m experiments.main --verbose --horizon 50 --learning_rate 0.00001 --hidden_dim 32 --layers 2 --beta 0.5 --batch_siz 64 --reverse 


upper:
	python -m experiments.main --verbose --horizon 5 --learning_rate 0.0001 --hidden_dim 512 --layers 3 --batch_size 64 --upper --skip_connection --reverse --beta 1



tune:
	python -m experiments.main --verbose --tune --horizon $(horizon)


vae_baseline_reverse_tune:
	python -m experiments.main --verbose --tune --horizon $(horizon) --vae-baseline --reverse



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
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260924-055637.pt

eval_upper_reverse:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260924-055618.pt

eval_upper:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260924-055620.pt

eval_reverse:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260924-055622.pt

eval_baseline:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260924-055634.pt

eval_vae:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260924-055628.pt

eval_vae_reverse:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20260924-055631.pt
