

short_horizon = 10
mid_horizon = 50
long_horizon = 100


_: tune vae vae_no_warmup range


main:
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 32 --layers 2 --epochs 50 --beta 0.4 --learning_rate 0.01 --skip_connection

vae_test:
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 32 --layers 2 --epochs 50 --beta 0.4 --learning_rate 0.01 --vae_baseline --skip_connection


ranged_test:
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 32 --layers 2 --epochs 50 --beta 0.4 --learning_rate 0.01 --range --skip_connection




eval_tune:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20261112-125659.pt


eval_vae:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20261112-125646.pt


eval_range:
	python -m experiments.eval --verbose  --checkpoint_path $(eval_range_short)


eval_vae_no_warmup:
	python -m experiments.eval --verbose  --checkpoint_path $(eval_vae_no_warmup_short)




tune: folder
	sleep 33
	python -m experiments.main --verbose --tune --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_short_.log
	#python -m experiments.main --verbose --tune --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_mid.log
	#python -m experiments.main --verbose --tune --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_long.log


range: folder
	sleep 10
	python -m experiments.main --verbose --tune --range --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_short_.log
	#python -m experiments.main --verbose --tune --range --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_mid.log
	#python -m experiments.main --verbose --tune --range --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_long.log


vae: folder
	sleep 20
	python -m experiments.main --verbose --tune --vae_baseline --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_short_.log
	#python -m experiments.main --verbose --tune --vae_baseline --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_mid.log
	#python -m experiments.main --verbose --tune --vae_baseline --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_long.log

vae_no_warmup: folder
	sleep 1
	python -m experiments.main --verbose --tune --vae_baseline --vae_no_warmup --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_short_.log
	#python -m experiments.main --verbose --tune --vae_baseline --vae_no_warmup --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_mid.log
	#python -m experiments.main --verbose --tune --vae_baseline --vae_no_warmup --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_long.log

folder:
	mkdir -p results/tune
	mkdir -p results/upper
	mkdir -p results/lower
	mkdir -p results/range
	mkdir -p results/vae_no_warmup
	mkdir -p results/vae



