import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from preprocess_metropt3 import BASE, HISTORY, build_cases, split_development


def frame(times):
    n = len(times)
    return pd.DataFrame({'timestamp': pd.to_datetime(times), 'Reservoirs': np.arange(n, dtype=float),
                         'Motor_current': np.ones(n), 'Oil_temperature': np.ones(n) * 20,
                         'COMP': np.ones(n)})


class PreprocessingTests(unittest.TestCase):
    def test_nearest_and_tie_earlier(self):
        # One anchor at index 31; 295 and 305 seconds from it are tied.
        times = pd.Timestamp('2020-02-01') + pd.to_timedelta(np.r_[np.arange(32) * 10,
                    315 + np.arange(1, 31) * 10, 620], unit='s')
        cases = build_cases(frame(times))
        match = np.flatnonzero(cases.anchors == 31)
        self.assertEqual(len(match), 1)
        self.assertEqual(cases.timestamps[cases.targets[match[0]]] - cases.timestamps[31], np.timedelta64(295, 's'))
        # An exact 300-second reading must beat adjacent candidates.
        regular = build_cases(frame(pd.date_range('2020-02-01', periods=100, freq='10s')))
        self.assertEqual(regular.timestamps[regular.targets[0]] - regular.timestamps[regular.anchors[0]], np.timedelta64(300, 's'))

    def test_gap_and_partition_boundaries(self):
        times = pd.date_range('2020-03-31 23:50:00', periods=180, freq='10s')
        df = frame(times)
        both = build_cases(df)
        for anchor, target in zip(both.anchors, both.targets):
            self.assertEqual(df.timestamp.iloc[anchor - HISTORY + 1] < pd.Timestamp('2020-04-01'),
                             df.timestamp.iloc[target] < pd.Timestamp('2020-04-01'))
        train, valid = split_development(df)
        self.assertEqual(len(both.anchors), len(build_cases(train).anchors) + len(build_cases(valid).anchors))
        df.loc[90:, 'timestamp'] += pd.Timedelta(seconds=30)
        gap = build_cases(df)
        for anchor, target in zip(gap.anchors, gap.targets):
            self.assertFalse(anchor - HISTORY + 1 < 90 <= target)

    def test_causality_shared_cases_and_reset(self):
        df = frame(pd.date_range('2020-02-01', periods=120, freq='10s'))
        cases = build_cases(df)
        self.assertEqual(cases.anchors[0], 31)  # first window needs two preceding readings
        i = 3
        anchor = cases.anchors[i]
        original = cases.window(i, True)
        changed = df.copy()
        changed.loc[anchor + 1:, list(BASE)] = 99999
        modified = build_cases(changed)
        j = np.flatnonzero(modified.anchors == anchor)[0]
        np.testing.assert_array_equal(original, modified.window(j, True))
        np.testing.assert_array_equal(cases.window(i), original[:, :4])
        self.assertEqual(cases.window(i).shape, (30, 4))
        self.assertEqual(original.shape, (30, 6))
        self.assertEqual(original[-1, 4], anchor - 1)
        self.assertAlmostEqual(original[-1, 5], 0.1)
        self.assertEqual(cases.label(i), df.Reservoirs.iloc[cases.targets[i]])

    def test_invalid_or_missing_input(self):
        df = frame(pd.date_range('2020-02-01', periods=100, freq='10s'))
        df.loc[5, 'COMP'] = np.nan
        cases = build_cases(df)
        self.assertTrue(all(not (a - 29 <= 5 <= a) for a in cases.anchors))
        df.loc[1, 'timestamp'] = df.loc[0, 'timestamp']
        with self.assertRaises(ValueError):
            build_cases(df)


if __name__ == '__main__':
    unittest.main()
