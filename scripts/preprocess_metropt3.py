"""Index-based causal MetroPT-3 preprocessing; no model fitting or scaling."""
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/metropt3-dataset.zip'
SHA256 = 'aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a'
CSV = 'MetroPT3(AirCompressor).csv'
TRAIN_END = pd.Timestamp('2020-04-01')
VALID_END = pd.Timestamp('2020-06-01')
BASE = ('Reservoirs', 'Motor_current', 'Oil_temperature', 'COMP')
DERIVED = ('Reservoirs_mean_3', 'Reservoirs_rate_bar_s')
HISTORY = 30


def load_development(path=SOURCE):
    """Verify the bundle and read only the chronological development period."""
    with path.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != SHA256:
            raise ValueError('SHA-256 inesperado')
    chunks = []
    with zipfile.ZipFile(path) as archive, archive.open(CSV) as csv:
        for chunk in pd.read_csv(csv, usecols=['timestamp', *BASE], parse_dates=['timestamp'], chunksize=100_000):
            chunks.append(chunk.loc[chunk.timestamp < VALID_END])
            if (chunk.timestamp >= VALID_END).any():
                break
    return pd.concat(chunks, ignore_index=True)


@dataclass
class Cases:
    """One shared set of anchor/label indices; materialize a single window on demand."""
    timestamps: np.ndarray
    values: np.ndarray
    anchors: np.ndarray
    targets: np.ndarray
    segments: int

    def window(self, case, augmented=False):
        end = self.anchors[case] + 1
        cols = 6 if augmented else 4
        return self.values[end - HISTORY:end, :cols].copy()

    def label(self, case):
        return self.values[self.targets[case], 0]

    def report(self):
        horizons = (self.timestamps[self.targets] - self.timestamps[self.anchors]) / np.timedelta64(1, 's')
        values, counts = np.unique(horizons, return_counts=True)
        mae = np.abs(self.values[self.targets, 0] - self.values[self.anchors, 0]).mean() if len(self.anchors) else None
        return {'segments': self.segments, 'windows': len(self.anchors),
                'base_columns': list(BASE), 'augmented_columns': list(BASE + DERIVED),
                'base_shape': [len(self.anchors), HISTORY, 4],
                'augmented_shape': [len(self.anchors), HISTORY, 6],
                'horizon_seconds_counts': {str(int(v)) if v.is_integer() else str(v): int(c) for v, c in zip(values, counts)},
                'persistence_mae_bar': float(mae) if mae is not None else None}


def build_cases(frame, lower=8, upper=15):
    """Build cases for ONE partition. Timestamps must be strictly increasing and naive."""
    if lower <= 0 or upper < lower:
        raise ValueError('Intervalo de continuidad inválido')
    ts = pd.DatetimeIndex(frame['timestamp'])
    if ts.tz is not None or ts.hasnans or not ts.is_monotonic_increasing or ts.has_duplicates:
        raise ValueError('Timestamps deben ser crecientes, únicos, válidos y sin zona horaria')
    n = len(ts)
    times = ts.to_numpy(dtype='datetime64[ns]')
    seconds = times.astype('int64') / 1e9
    raw = frame.loc[:, BASE].to_numpy(dtype=float)
    values = np.full((n, 6), np.nan)
    values[:, :4] = raw
    if not n:
        return Cases(times, values, np.array([], dtype=int), np.array([], dtype=int), 0)
    delta = np.diff(seconds)
    cuts = ((times[:-1] < TRAIN_END.to_datetime64()) & (times[1:] >= TRAIN_END.to_datetime64())) | (
        (times[:-1] < VALID_END.to_datetime64()) & (times[1:] >= VALID_END.to_datetime64()))
    starts = np.r_[True, (delta < lower) | (delta > upper) | cuts]
    segment = np.cumsum(starts)
    same_prev = ~starts[1:]
    values[1:, 5] = np.where(same_prev, np.diff(raw[:, 0]) / delta, np.nan)
    if n >= 3:
        valid3 = (segment[2:] == segment[:-2])
        values[2:, 4] = np.where(valid3, (raw[2:, 0] + raw[1:-1, 0] + raw[:-2, 0]) / 3, np.nan)
    # Consecutive segment end indices constrain nearest-label search to the anchor segment.
    boundaries = np.r_[np.flatnonzero(starts), n]
    ends = np.repeat(boundaries[1:] - 1, np.diff(boundaries))
    anchors = np.arange(HISTORY - 1, n)
    desired = times[anchors] + np.timedelta64(300, 's')
    right = np.minimum(np.searchsorted(times, desired), ends[anchors])
    left = np.maximum(right - 1, anchors)
    right = np.maximum(right, anchors)
    dr = np.abs(times[right] - desired)
    dl = np.abs(times[left] - desired)
    targets = np.where(dl <= dr, left, right)
    finite = np.isfinite(values).all(axis=1)
    bad_prefix = np.r_[0, np.cumsum(~finite)]
    start = anchors - HISTORY + 1
    valid = ((targets > anchors) & (np.abs(times[targets] - desired) <= np.timedelta64(5, 's')) &
             (segment[start] == segment[targets]) & (bad_prefix[anchors + 1] == bad_prefix[start]) &
             np.isfinite(raw[targets, 0]))
    return Cases(times, values, anchors[valid], targets[valid], len(boundaries) - 1)


def split_development(frame):
    return (frame.loc[frame.timestamp < TRAIN_END].reset_index(drop=True),
            frame.loc[(frame.timestamp >= TRAIN_END) & (frame.timestamp < VALID_END)].reset_index(drop=True))


def main():
    train, validation = split_development(load_development())
    principal = build_cases(train)
    result = {'config': {'history': HISTORY, 'horizon_seconds': 300, 'tolerance_seconds': 5,
                         'continuity_seconds': [8, 15], 'scaling': None},
              'train': principal.report(), 'validation': build_cases(validation).report(),
              'sensitivity_train': {}}
    for lo, hi in [(7, 16), (8, 15), (9, 14)]:
        cases = principal if (lo, hi) == (8, 15) else build_cases(train, lo, hi)
        report = cases.report()
        result['sensitivity_train'][f'{lo}-{hi}'] = {k: report[k] for k in ('segments', 'windows', 'persistence_mae_bar')}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
