import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from inference_metropt3 import predict_readings
    from example_request import sample
    from preprocess_metropt3 import build_cases


@unittest.skipIf(torch is None, 'PyTorch is required for inference tests')
class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.readings = sample()
        self.scaler = {'mean': [0] * 6, 'std': [1] * 6}
        class Capture(torch.nn.Module):
            def forward(self, x):
                self.x = x
                return x[:, -1, 0]
        self.model = Capture()

    def test_32_readings_match_existing_preprocessing(self):
        result = predict_readings(self.model, self.scaler, self.readings)
        frame = pd.DataFrame(self.readings)
        frame['timestamp'] = pd.to_datetime(frame['timestamp'])
        expected = build_cases(frame).values[-30:].astype('float32')
        np.testing.assert_array_equal(self.model.x.numpy()[0], expected)
        self.assertAlmostEqual(result['prediction_bar'], self.readings[-1]['Reservoirs'], places=5)
        self.assertEqual(result['forecast_timestamp'], '2020-07-01T00:10:10')
        with self.assertRaisesRegex(ValueError, '32'):
            predict_readings(self.model, self.scaler, self.readings[1:])

    def test_invalid_intervals_values_and_cut(self):
        for key, value in [('timestamp', '2020-07-01T00:00:30Z'), ('Reservoirs', float('nan'))]:
            rows = [r.copy() for r in self.readings]
            rows[2][key] = value
            with self.assertRaises(ValueError):
                predict_readings(self.model, self.scaler, rows)
        rows = [r.copy() for r in self.readings]
        rows[2]['timestamp'] = '2020-07-01T00:00:26'
        with self.assertRaisesRegex(ValueError, '8 a 15'):
            predict_readings(self.model, self.scaler, rows)
        rows = [r.copy() for r in self.readings]
        for i, row in enumerate(rows):
            row['timestamp'] = (pd.Timestamp('2020-05-31 23:58:00') + pd.Timedelta(seconds=10*i)).isoformat()
        with self.assertRaisesRegex(ValueError, 'cortes'):
            predict_readings(self.model, self.scaler, rows)


if __name__ == '__main__':
    unittest.main()
