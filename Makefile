

short_horizon = 10
mid_horizon = 50
long_horizon = 100


tune: folder
	python -m experiments.main --verbose --tune --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_short.log
	python -m experiments.main --verbose --tune --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_mid.log
	python -m experiments.main --verbose --tune --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/tune/tune_long.log


upper: folder
	python -m experiments.main --verbose --upper --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/upper/upper_short.log
	python -m experiments.main --verbose --upper --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/upper/upper_mid.log
	python -m experiments.main --verbose --upper --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/upper/upper_long.log

lower: folder
	python -m experiments.main --verbose --lower --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/lower/lower_short.log
	python -m experiments.main --verbose --lower --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/lower/lower_mid.log
	python -m experiments.main --verbose --lower --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/lower/lower_long.log


range: folder
	python -m experiments.main --verbose --range --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_short.log
	python -m experiments.main --verbose --range --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_mid.log
	python -m experiments.main --verbose --range --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/range/range_long.log


vae: folder
	python -m experiments.main --verbose --vae --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_short.log
	python -m experiments.main --verbose --vae --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_mid.log
	python -m experiments.main --verbose --vae --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae/vae_long.log

vae_no_warmup: folder
	python -m experiments.main --verbose --vae_no_warmup --horizon $(short_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_short.log
	python -m experiments.main --verbose --vae_no_warmup --horizon $(mid_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_mid.log
	python -m experiments.main --verbose --vae_no_warmup --horizon $(long_horizon) --grad_diagnostics --scrabbled_y 2>&1 | tee results/vae_no_warmup/vae_no_warmup_long.log

folder:
	mkdir -p results/tune
	mkdir -p results/upper
	mkdir -p results/lower
	mkdir -p results/range



