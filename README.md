# VAE skip connections and constrained optimization

## Cyclical KL annealing

The VAE baseline supports an optional cyclical KL schedule inspired by Fu et al.,
*Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing* (2019).
For example:

```sh
PYTHONPATH=src python -m experiments.main --vae_baseline --cyclical_kl \
  --epochs 100 --beta 1 --kl_cycle_epochs 20 --kl_cycle_ratio 0.5
```

Each cycle resets the KL coefficient to zero, increases it linearly to `beta`,
and holds it there for the remaining epochs. The ramp lasts
`ceil(kl_cycle_epochs * kl_cycle_ratio)` epochs, clamped to at least one and at
most one less than the cycle length so every cycle reaches `beta`.
The cycle length must be at least 2 and the ratio must be in `(0, 1]`.
The schedule operates per epoch, including during tuning; it cannot be combined
with `--vae_no_warmup`. Without `--cyclical_kl`, the existing VAE warmup and
other models' objectives are unchanged.
