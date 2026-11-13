

short_horizon = 10
mid_horizon = 50
long_horizon = 100


_: tune  upper vae lower vae_no_warmup


main:
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 32 --layers 2 --epochs 50 --beta 0.4 --learning_rate 0.01 --skip_connection

vae_test:
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 32 --layers 2 --epochs 50 --beta 0.4 --learning_rate 0.01 --vae_baseline --skip_connection


ranged_test:
	python -m experiments.main --verbose --horizon 10 --batch_size 64 --hidden_dim 32 --layers 2 --epochs 50 --beta 0.4 --learning_rate 0.01 --range --skip_connection






test:
	python -m experiments.main --verbose --tune --horizon 10

eval_tune_short="artifacts_dev/tdlgm/checkpoint_epochbest_20261027-092907.pt"
eval_tune_mid="artifacts_dev/tdlgm/checkpoint_epochbest_20261028-223040.pt"
eval_tune_long="artifacts_dev/tdlgm/checkpoint_epochbest_20261029-045814.pt"


eval_upper_short="artifacts_dev/tdlgm/checkpoint_epochbest_20261027-092907.pt"
eval_upper_mid="artifacts_dev/tdlgm/checkpoint_epochbest_20261028-223040.pt"
eval_upper_long="artifacts_dev/tdlgm/checkpoint_epochbest_20261028-004821.pt"


eval_lower_short="artifacts_dev/tdlgm/checkpoint_epochbest_20261027-092907.pt"
eval_lower_mid="artifacts_dev/tdlgm/checkpoint_epochbest_20261028-223041.pt"
eval_lower_long="artifacts_dev/tdlgm/checkpoint_epochbest_20261029-030930.pt"




eval_vae_short="artifacts_dev/tdlgm/checkpoint_epochbest_20261027-092907.pt"
eval_vae_mid="artifacts_dev/tdlgm/checkpoint_epochbest_20261030-121311.pt"
eval_vae_long="artifacts_dev/tdlgm/checkpoint_epochbest_20261027-215153.pt"





eval_range_short="artifacts_dev/tdlgm/checkpoint_epochbest_20261027-092907.pt"
eval_range_mid="artifacts_dev/tdlgm/checkpoint_epochbest_20261027-174304.pt"
eval_range_long="artifacts_dev/tdlgm/checkpoint_epochbest_20261028-034154.pt"




eval_vae_no_warmup_short="artifacts_dev/tdlgm/checkpoint_epochbest_20261027-092907.pt"
eval_vae_no_warmup_mid="artifacts_dev/tdlgm/checkpoint_epochbest_20261028-223040.pt"
eval_vae_no_warmup_long="artifacts_dev/tdlgm/checkpoint_epochbest_20261027-170122.pt"






eval_tune:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20261112-125659.pt


eval_upper:
	python -m experiments.eval --verbose  --checkpoint_path $(eval_upper_short)


eval_lower:
	python -m experiments.eval --verbose  --checkpoint_path $(eval_lower_short)


eval_vae:
	python -m experiments.eval --verbose  --checkpoint_path artifacts_dev/tdlgm/checkpoint_epochbest_20261112-125646.pt


eval_range:
	python -m experiments.eval --verbose  --checkpoint_path $(eval_range_short)


eval_vae_no_warmup:
	python -m experiments.eval --verbose  --checkpoint_path $(eval_vae_no_warmup_short)




tune: folder
	sleep 33
	python -m experiments.main --verbose --tune --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_short_new.log
	#python -m experiments.main --verbose --tune --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_mid.log
	#python -m experiments.main --verbose --tune --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_long.log


upper: folder
	sleep 123
	python -m experiments.main --verbose --tune --upper --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/upper/upper_short_new.log
	#python -m experiments.main --verbose --tune --upper --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/upper/upper_mid.log
	#python -m experiments.main --verbose --tune --upper --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/upper/upper_long.log

lower: folder
	sleep 40
	python -m experiments.main --verbose --tune --lower --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/lower/lower_short_new.log
	#python -m experiments.main --verbose --tune --lower --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/lower/lower_mid.log
	#python -m experiments.main --verbose --tune --lower --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/lower/lower_long.log


range: folder
	sleep 10
	python -m experiments.main --verbose --tune --range --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_short_new.log
	#python -m experiments.main --verbose --tune --range --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_mid.log
	#python -m experiments.main --verbose --tune --range --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_long.log


vae: folder
	sleep 20
	python -m experiments.main --verbose --tune --vae_baseline --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_short_new.log
	#python -m experiments.main --verbose --tune --vae_baseline --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_mid.log
	#python -m experiments.main --verbose --tune --vae_baseline --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_long.log

vae_no_warmup: folder
	sleep 1
	python -m experiments.main --verbose --tune --vae_baseline --vae_no_warmup --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_short_new.log
	#python -m experiments.main --verbose --tune --vae_baseline --vae_no_warmup --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_mid.log
	#python -m experiments.main --verbose --tune --vae_baseline --vae_no_warmup --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_long.log

folder:
	mkdir -p results/tune
	mkdir -p results/upper
	mkdir -p results/lower
	mkdir -p results/range
	mkdir -p results/vae_no_warmup
	mkdir -p results/vae



