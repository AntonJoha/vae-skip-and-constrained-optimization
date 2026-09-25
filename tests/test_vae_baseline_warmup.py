import unittest

from experiments.util import SeriesConfig
from model.vae_baseline import Model


class TestVAEBaselineWarmup(unittest.TestCase):
    def _make_config(self) -> SeriesConfig:
        return SeriesConfig(
            input_dim=1,
            output_dim=1,
            horizon=1,
            hidden_dim=16,
            layers=1,
            epochs=10,
            beta=2.0,
            vae_baseline=True,
            vae_kl_warmup_fraction=0.5,
        )

    def test_warmup_starts_at_zero(self):
        model = Model(self._make_config())
        model.set_epoch(0)
        self.assertEqual(model.kl_weight, 0.0)

    def test_warmup_ramps_up(self):
        model = Model(self._make_config())
        model.set_epoch(2)
        self.assertAlmostEqual(model.kl_weight, 0.8)

    def test_warmup_saturates_at_beta(self):
        model = Model(self._make_config())
        model.set_epoch(100)
        self.assertEqual(model.kl_weight, 2.0)


if __name__ == "__main__":
    unittest.main()
