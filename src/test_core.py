"""Unit tests T01-T08. Run from src/: python3 -m unittest test_core -v"""
import inspect
import os
import unittest

import numpy as np
import pandas as pd

import core
import weather


class T01Certifiable(unittest.TestCase):
    def test_hand_computed_day(self):
        w = np.array([0.1] * 10 + [0.9] * 5 + [0.3] * 9)  # 24 h; the cap E=0.5 binds on the five 0.9 hours
        exc = np.zeros(24, bool)
        exc[[0, 1, 20]] = True  # three exception hours run at E regardless of wind
        expected = 3 * 0.5 + 8 * 0.1 + 5 * 0.5 + 8 * 0.3
        self.assertAlmostEqual(core.certifiable(w, exc, 0.5), expected, places=12)


class T02DaylightSaving(unittest.TestCase):
    def test_23_and_25_hour_days(self):
        spring = pd.date_range('2024-03-30T23:00', '2024-03-31T22:00', freq='h', tz='UTC')
        autumn = pd.date_range('2024-10-26T22:00', '2024-10-27T22:00', freq='h', tz='UTC')
        self.assertEqual(int((core.local_day(spring) == pd.Timestamp('2024-03-31')).sum()), 23)
        self.assertEqual(int((core.local_day(autumn) == pd.Timestamp('2024-10-27')).sum()), 25)
        frame = pd.DataFrame({'Qhat': [23 * 0.5, 25 * 0.5], 'n_hours': [23, 25]})
        np.testing.assert_allclose(core._bin_value(frame, 0.5), [1.0, 1.0])


class T03Conformal(unittest.TestCase):
    def test_rank(self):
        self.assertEqual(core.conformal_shift(np.arange(19.0)), 18.0)  # ceil(20*0.95)=19 -> maximum
        self.assertEqual(core.conformal_shift(np.arange(39.0)), 37.0)  # ceil(40*0.95)=38 -> 38th smallest
        self.assertEqual(core.conformal_shift(np.arange(10.0)), np.inf)  # too few points: never commit


class T04Leakage(unittest.TestCase):
    def test_feature_sources(self):
        source = inspect.getsource(core.load_hourly) + inspect.getsource(core.day_features)
        self.assertIn("values='ForecastDayAhead'", source)
        for banned in ('ForecastIntraday', 'ForecastCurrent'):
            self.assertNotIn(banned, source)
        for name in core.FEATURES:
            self.assertFalse(name.startswith('real') or name == 'Q', name)
        self.assertTrue(all(v.endswith('_previous_day2') for v in weather.VARS))

    def test_capacity_window_ends_D_minus_16(self):
        hours = pd.date_range('2023-01-01', '2024-03-01', freq='h', tz='UTC')
        series = pd.Series(1.0, index=hours)
        day = pd.Timestamp('2024-02-01')
        late = hours[core.local_day(hours) > day - pd.Timedelta(days=16)]
        series[late] = 1000.0  # any leak from D-15..D would raise K_D
        self.assertEqual(core.capacity_proxy(series, [day])[day], 1.0)


class T05Prices(unittest.TestCase):
    def test_quarter_hours_are_averaged(self):
        if not any(f.startswith('DayAheadPrices') for f in os.listdir(core.RAW)):
            self.skipTest('DayAheadPrices not downloaded yet')
        price = core.hourly_prices()
        raw = core._read('DayAheadPrices_*.csv')
        raw = raw[raw['PriceArea'] == 'DK1']
        hour = pd.Timestamp('2025-10-15T10:00', tz='UTC')
        quarters = raw[pd.to_datetime(raw['TimeUTC'], utc=True).dt.floor('h') == hour]['DayAheadPriceEUR']
        self.assertEqual(len(quarters), 4)
        self.assertAlmostEqual(price[hour], quarters.mean(), places=9)


class T06Splits(unittest.TestCase):
    def test_no_overlap_with_test(self):
        test_start = pd.Timestamp(core.SPLITS['test'][0])
        for split in ('train', 'calib', 'pretest'):
            self.assertLess(pd.Timestamp(core.SPLITS[split][1]), test_start)
        c2_test_start = pd.Timestamp(core.SPLITS['c2_test'][0])
        for split in ('c2_train', 'c2_calib', 'c2_pretest_b'):
            self.assertLess(pd.Timestamp(core.SPLITS[split][1]), c2_test_start)


@unittest.skipUnless(os.environ.get('RUN_EVM_TESTS'), 'set RUN_EVM_TESTS=1 (needs anvil and a loopback port)')
class T07Registry(unittest.TestCase):
    def test_roundtrip_and_refusals(self):
        import tempfile
        import registry
        test = pd.DataFrame({'Q': [1.0, 2.0, 0.5]}, index=pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']))
        with tempfile.TemporaryDirectory() as tmp:
            result = registry.run(test, {'C1': np.array([0.8, 2.5, 0.0])}, tmp)
        self.assertTrue(result['pass'], result)
        self.assertEqual(result['mismatches'], 0)


class T08Bootstrap(unittest.TestCase):
    def test_deterministic_and_paired(self):
        rng = np.random.default_rng(0)
        a, b = rng.random(100), rng.random(100) + 0.1
        self.assertEqual(core.bootstrap_gain(a, b, n_boot=200), core.bootstrap_gain(a, b, n_boot=200))
        same = core.bootstrap_gain(b, b, n_boot=200)
        self.assertEqual(same['ci95'], [0.0, 0.0])  # paired blocks: identical losses give exactly zero


if __name__ == '__main__':
    unittest.main()
