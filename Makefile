
seq_len = 5
short_horizon = 10
mid_horizon = 50
long_horizon = 100


_: tune vae vae_no_warmup range


main:
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 32 --layers 2 --epochs 50 --beta 0.4 --learning_rate 0.001 --skip_connection

vae_test:
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 32 --layers 2 --epochs 50 --beta 0.4 --learning_rate 0.01 --vae_baseline --skip_connection


ranged_test:
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 32 --layers 2 --epochs 50 --beta 0.4 --learning_rate 0.01 --range --skip_connection


seed: tune_seed vae_seed range_seed vae_no_warmup_seed


tune_seed:
	python -m experiments.main --verbose --horizon 10 --batch_size 16 --hidden_dim 16 --layers 5 --epochs 100 --beta 0.13 --learning_rate 0.001 --skip_connection --seed 1 2>&1 | tee results/tune/seed1.log
	python -m experiments.main --verbose --horizon 10 --batch_size 16 --hidden_dim 16 --layers 5 --epochs 100 --beta 0.13 --learning_rate 0.001 --skip_connection --seed 2 2>&1 | tee results/tune/seed2.log
	python -m experiments.main --verbose --horizon 10 --batch_size 16 --hidden_dim 16 --layers 5 --epochs 100 --beta 0.13 --learning_rate 0.001 --skip_connection --seed 3 2>&1 | tee results/tune/seed3.log
	python -m experiments.main --verbose --horizon 10 --batch_size 16 --hidden_dim 16 --layers 5 --epochs 100 --beta 0.13 --learning_rate 0.001 --skip_connection --seed 4 2>&1 | tee results/tune/seed4.log
	python -m experiments.main --verbose --horizon 10 --batch_size 16 --hidden_dim 16 --layers 5 --epochs 100 --beta 0.13 --learning_rate 0.001 --skip_connection --seed 5 2>&1 | tee results/tune/seed5.log
	python -m experiments.main --verbose --horizon 10 --batch_size 16 --hidden_dim 16 --layers 5 --epochs 100 --beta 0.13 --learning_rate 0.001 --skip_connection --seed 6 2>&1 | tee results/tune/seed6.log
	python -m experiments.main --verbose --horizon 10 --batch_size 16 --hidden_dim 16 --layers 5 --epochs 100 --beta 0.13 --learning_rate 0.001 --skip_connection --seed 7 2>&1 | tee results/tune/seed7.log
	python -m experiments.main --verbose --horizon 10 --batch_size 16 --hidden_dim 16 --layers 5 --epochs 100 --beta 0.13 --learning_rate 0.001 --skip_connection --seed 8 2>&1 | tee results/tune/seed8.log
	python -m experiments.main --verbose --horizon 10 --batch_size 16 --hidden_dim 16 --layers 5 --epochs 100 --beta 0.13 --learning_rate 0.001 --skip_connection --seed 9 2>&1 | tee results/tune/seed9.log
	python -m experiments.main --verbose --horizon 10 --batch_size 16 --hidden_dim 16 --layers 5 --epochs 100 --beta 0.13 --learning_rate 0.001 --skip_connection --seed 10 2>&1 | tee results/tune/seed10.log

vae_seed:
	python -m experiments.main --verbose --vae_baseline --skip_connection --horizon 10 --seq_len 5 --batch_size 16 --hidden_dim 32 --learning_rate 0.0022 --beta 0.075 --seed 1 2>&1 | tee results/vae/seed1.log
	python -m experiments.main --verbose --vae_baseline --skip_connection --horizon 10 --seq_len 5 --batch_size 16 --hidden_dim 32 --learning_rate 0.0022 --beta 0.075 --seed 2 2>&1 | tee results/vae/seed2.log
	python -m experiments.main --verbose --vae_baseline --skip_connection --horizon 10 --seq_len 5 --batch_size 16 --hidden_dim 32 --learning_rate 0.0022 --beta 0.075 --seed 3 2>&1 | tee results/vae/seed3.log
	python -m experiments.main --verbose --vae_baseline --skip_connection --horizon 10 --seq_len 5 --batch_size 16 --hidden_dim 32 --learning_rate 0.0022 --beta 0.075 --seed 4 2>&1 | tee results/vae/seed4.log
	python -m experiments.main --verbose --vae_baseline --skip_connection --horizon 10 --seq_len 5 --batch_size 16 --hidden_dim 32 --learning_rate 0.0022 --beta 0.075 --seed 5 2>&1 | tee results/vae/seed5.log
	python -m experiments.main --verbose --vae_baseline --skip_connection --horizon 10 --seq_len 5 --batch_size 16 --hidden_dim 32 --learning_rate 0.0022 --beta 0.075 --seed 6 2>&1 | tee results/vae/seed6.log
	python -m experiments.main --verbose --vae_baseline --skip_connection --horizon 10 --seq_len 5 --batch_size 16 --hidden_dim 32 --learning_rate 0.0022 --beta 0.075 --seed 7 2>&1 | tee results/vae/seed7.log
	python -m experiments.main --verbose --vae_baseline --skip_connection --horizon 10 --seq_len 5 --batch_size 16 --hidden_dim 32 --learning_rate 0.0022 --beta 0.075 --seed 8 2>&1 | tee results/vae/seed8.log
	python -m experiments.main --verbose --vae_baseline --skip_connection --horizon 10 --seq_len 5 --batch_size 16 --hidden_dim 32 --learning_rate 0.0022 --beta 0.075 --seed 9 2>&1 | tee results/vae/seed9.log
	python -m experiments.main --verbose --vae_baseline --skip_connection --horizon 10 --seq_len 5 --batch_size 16 --hidden_dim 32 --learning_rate 0.0022 --beta 0.075 --seed 10 2>&1 | tee results/vae/seed10.log



range_seed:
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 64 --layers 3 --beta 0.03 --learning_rate 0.00007 --range --skip_connection --seq_len 5 --seed 1 2>&1 | tee results/range/seed1.log
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 64 --layers 3 --beta 0.03 --learning_rate 0.00007 --range --skip_connection --seq_len 5 --seed 2 2>&1 | tee results/range/seed2.log
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 64 --layers 3 --beta 0.03 --learning_rate 0.00007 --range --skip_connection --seq_len 5 --seed 3 2>&1 | tee results/range/seed3.log
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 64 --layers 3 --beta 0.03 --learning_rate 0.00007 --range --skip_connection --seq_len 5 --seed 4 2>&1 | tee results/range/seed4.log
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 64 --layers 3 --beta 0.03 --learning_rate 0.00007 --range --skip_connection --seq_len 5 --seed 5 2>&1 | tee results/range/seed5.log
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 64 --layers 3 --beta 0.03 --learning_rate 0.00007 --range --skip_connection --seq_len 5 --seed 6 2>&1 | tee results/range/seed6.log
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 64 --layers 3 --beta 0.03 --learning_rate 0.00007 --range --skip_connection --seq_len 5 --seed 7 2>&1 | tee results/range/seed7.log
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 64 --layers 3 --beta 0.03 --learning_rate 0.00007 --range --skip_connection --seq_len 5 --seed 8 2>&1 | tee results/range/seed8.log
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 64 --layers 3 --beta 0.03 --learning_rate 0.00007 --range --skip_connection --seq_len 5 --seed 9 2>&1 | tee results/range/seed9.log
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 64 --layers 3 --beta 0.03 --learning_rate 0.00007 --range --skip_connection --seq_len 5 --seed 10 2>&1 | tee results/range/seed10.log



vae_no_warmup_seed:
	python -m experiments.main --verbose --vae_baseline --vae_no_warmup --horizon 10 --seq_len 5 --learning_rate 0.0001 --hidden_dim 16 --layers 5 --beta 0.09 --batch_size 16  --seed 1 2>&1 | tee results/vae_no_warmup/seed1.log
	python -m experiments.main --verbose --vae_baseline --vae_no_warmup --horizon 10 --seq_len 5 --learning_rate 0.001 --hidden_dim 16 --layers 5 --beta 0.09 --batch_size 16  --seed 2 2>&1 | tee results/vae_no_warmup/seed2.log
	python -m experiments.main --verbose --vae_baseline --vae_no_warmup --horizon 10 --seq_len 5 --learning_rate 0.001 --hidden_dim 16 --layers 5 --beta 0.09 --batch_size 16  --seed 3 1>&1 | tee results/vae_no_warmup/seed3.log
	python -m experiments.main --verbose --vae_baseline --vae_no_warmup --horizon 10 --seq_len 5 --learning_rate 0.001 --hidden_dim 16 --layers 5 --beta 0.09 --batch_size 16  --seed 4 2>&1 | tee results/vae_no_warmup/seed4.log
	python -m experiments.main --verbose --vae_baseline --vae_no_warmup --horizon 10 --seq_len 5 --learning_rate 0.001 --hidden_dim 16 --layers 5 --beta 0.09 --batch_size 16  --seed 5 2>&1 | tee results/vae_no_warmup/seed5.log
	python -m experiments.main --verbose --vae_baseline --vae_no_warmup --horizon 10 --seq_len 5 --learning_rate 0.001 --hidden_dim 16 --layers 5 --beta 0.09 --batch_size 16  --seed 6 2>&1 | tee results/vae_no_warmup/seed6.log
	python -m experiments.main --verbose --vae_baseline --vae_no_warmup --horizon 10 --seq_len 5 --learning_rate 0.001 --hidden_dim 16 --layers 5 --beta 0.09 --batch_size 16  --seed 7 2>&1 | tee results/vae_no_warmup/seed7.log
	python -m experiments.main --verbose --vae_baseline --vae_no_warmup --horizon 10 --seq_len 5 --learning_rate 0.001 --hidden_dim 16 --layers 5 --beta 0.09 --batch_size 16  --seed 8 2>&1 | tee results/vae_no_warmup/seed8.log
	python -m experiments.main --verbose --vae_baseline --vae_no_warmup --horizon 10 --seq_len 5 --learning_rate 0.001 --hidden_dim 16 --layers 5 --beta 0.09 --batch_size 16  --seed 9 2>&1 | tee results/vae_no_warmup/seed9.log
	python -m experiments.main --verbose --vae_baseline --vae_no_warmup --horizon 10 --seq_len 5 --learning_rate 0.001 --hidden_dim 16 --layers 5 --beta 0.09 --batch_size 16  --seed 10 2>&1 | tee results/vae_no_warmup/seed10.log



eval_tune:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20261124-141420.pt


eval_vae:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20261124-141407.pt


eval_range:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20261124-141357.pt


eval_vae_no_warmup:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20261124-141348.pt


cyclic_vae:
	python -m experiments.main --verbose --tune  --cyclical_kl --vae_baseline --horizon $(short_horizon) --seq_len $(seq_len) --scrabbled_y 2>&1 | tee results/cyclic/cyclic_short_.log

tune: folder
	sleep 33
	python -m experiments.main --verbose --tune --horizon $(short_horizon) --seq_len $(seq_len) --scrabbled_y 2>&1 | tee results/tune/tune_short_.log
	#python -m experiments.main --verbose --tune --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_mid.log
	#python -m experiments.main --verbose --tune --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_long.log


range: folder
	sleep 10
	python -m experiments.main --verbose --tune --range --horizon $(short_horizon) --seq_len $(seq_len) --scrabbled_y 2>&1 | tee results/range/range_short_.log
	#python -m experiments.main --verbose --tune --range --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_mid.log
	#python -m experiments.main --verbose --tune --range --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_long.log


vae: folder
	sleep 20
	python -m experiments.main --verbose --tune --vae_baseline --horizon $(short_horizon)  --seq_len $(seq_len) --scrabbled_y 2>&1 | tee results/vae/vae_short_.log
	#python -m experiments.main --verbose --tune --vae_baseline --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_mid.log
	#python -m experiments.main --verbose --tune --vae_baseline --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_long.log

vae_no_warmup: folder
	sleep 1
	python -m experiments.main --verbose --tune --vae_baseline --vae_no_warmup --horizon $(short_horizon) --seq_len $(seq_len)  --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_short_.log
	#python -m experiments.main --verbose --tune --vae_baseline --vae_no_warmup --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_mid.log
	#python -m experiments.main --verbose --tune --vae_baseline --vae_no_warmup --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_long.log

folder:
	mkdir -p results/tune
	mkdir -p results/upper
	mkdir -p results/lower
	mkdir -p results/range
	mkdir -p results/vae_no_warmup
	mkdir -p results/vae
	mkdir -p results/cyclic



