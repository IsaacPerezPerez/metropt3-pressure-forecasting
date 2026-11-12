import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from select_metropt3 import select


class SelectionTests(unittest.TestCase):
    def test_unrounded_mae_and_ties(self):
        rows = [dict(architecture=a, variant=v, mae_bar=0.5, smoke=False, reload_batch_identical=True)
                for a in ('RNN', 'LSTM') for v in ('original', 'ampliada')]
        self.assertIs(select(rows), rows[0])
        rows[3]['mae_bar'] = 0.4999999999
        self.assertIs(select(rows), rows[3])
        rows[3]['smoke'] = True
        with self.assertRaises(ValueError):
            select(rows)


if __name__ == '__main__':
    unittest.main()
