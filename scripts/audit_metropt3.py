"""Reproducible, read-only audit of the UCI MetroPT-3 bundle."""

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/metropt3-dataset.zip'
SHA256 = 'aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a'
CSV = 'MetroPT3(AirCompressor).csv'
TRAIN_END = pd.Timestamp('2020-04-01')
VALID_END = pd.Timestamp('2020-06-01')
MIN_STEP, MAX_STEP = 8, 15  # jitter accepted; larger intervals break a segment
HISTORY = 6  # six observations including the anchor


def main():
    digest = hashlib.file_digest(SOURCE.open('rb'), 'sha256').hexdigest()
    if digest != SHA256:
        raise ValueError(f'SHA-256 inesperado: {digest}')
    with zipfile.ZipFile(SOURCE) as archive, archive.open(CSV) as csv:
        df = pd.read_csv(csv, parse_dates=['timestamp'])
    print('rows, columns:', df.shape)
    print('date range:', df.timestamp.min(), df.timestamp.max())
    print('duplicates, nonmonotonic, missing timestamps:', df.timestamp.duplicated().sum(),
          (df.timestamp.diff().dropna() < pd.Timedelta(0)).sum(), df.timestamp.isna().sum())
    print('missing per column:\n', df.isna().sum().to_string())
    print('intervals seconds:\n', df.timestamp.diff().dt.total_seconds().value_counts().sort_index().to_string())
    print('continuous values:\n', df[['Reservoirs', 'Motor_current', 'Oil_temperature']].describe(
        percentiles=[.01, .25, .5, .75, .99]).to_string())
    digital = ['COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']
    for col in digital:
        print(col, df[col].value_counts(dropna=False).sort_index().to_dict())
    seconds = df.timestamp.diff().dt.total_seconds()
    breaks = ~seconds.between(MIN_STEP, MAX_STEP)
    segments = df.groupby(breaks.cumsum()).agg(start=('timestamp', 'first'), end=('timestamp', 'last'), n=('timestamp', 'size'))
    print('segments:', len(segments), 'largest:\n', segments.nlargest(10, 'n').to_string(index=False))
    gaps = seconds[seconds > MAX_STEP]
    print('gaps > 15s:', len(gaps), 'gap seconds total, max:', gaps.sum(), gaps.max(),
          'excess over 10s:', (gaps - 10).sum())
    print('gap quantiles seconds:', gaps.quantile([0, .5, .9, .99, 1]).to_dict())
    for n in [60, 300, 3600, 86400]:
        print('gaps >= seconds', n, ':', int((gaps >= n).sum()))
    print('split rows:', {'train': int((df.timestamp < TRAIN_END).sum()),
                         'validation': int(((df.timestamp >= TRAIN_END) & (df.timestamp < VALID_END)).sum()),
                         'test': int((df.timestamp >= VALID_END).sum())})
    # Nearest measured target within +/- 5 seconds, same segment and train split.
    train = df.loc[df.timestamp < TRAIN_END].reset_index(drop=True)
    segment = (~train.timestamp.diff().dt.total_seconds().between(MIN_STEP, MAX_STEP)).cumsum().to_numpy()
    times = train.timestamp.to_numpy().astype('datetime64[s]').astype('int64')
    pressure = train.Reservoirs.to_numpy()
    for horizon in [60, 300]:
        anchor = np.arange(HISTORY - 1, len(train))
        desired = times[anchor] + horizon
        right = np.searchsorted(times, desired).clip(0, len(times) - 1)
        left = (right - 1).clip(0, len(times) - 1)
        future = np.where(np.abs(times[left] - desired) <= np.abs(times[right] - desired), left, right)
        valid = ((future > anchor) & (np.abs(times[future] - desired) <= 5) &
                 (segment[anchor - HISTORY + 1] == segment[future]) &
                 np.isfinite(pressure[anchor]) & np.isfinite(pressure[future]))
        delta = pressure[future[valid]] - pressure[anchor[valid]]
        print('horizon seconds, windows, mean signed change, mean abs change/MAE, median abs, p95 abs:',
              horizon, len(delta), delta.mean(), np.abs(delta).mean(),
              np.median(np.abs(delta)), np.quantile(np.abs(delta), .95))


if __name__ == '__main__':
    main()
