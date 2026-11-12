import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/raw/PRSA2017_Data_20130301-20170228.zip'
SHA256 = 'd1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8'
INNER = 'PRSA2017_Data_20130301-20170228.zip'


def main():
    blob = ARCHIVE.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        raise ValueError(f'SHA-256 incorrecto: {digest}')
    with zipfile.ZipFile(io.BytesIO(blob)) as inner:
        bad = inner.testzip()
        if bad:
            raise ValueError(f'ZIP interior corrupto: {bad}')
        names = inner.namelist()
        matches = [n for n in names if n.endswith('PRSA_Data_Dongsi_20130301-20170228.csv')]
        if len(matches) != 1:
            raise ValueError(f'CSV Dongsi ambiguo: {matches}')
        with inner.open(matches[0]) as f:
            rows = list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig', newline='')))
    series = []
    for row in rows:
        if row['station'] != 'Dongsi':
            raise ValueError('Estación inesperada')
        t = datetime(*(int(row[k]) for k in ('year', 'month', 'day', 'hour')))
        value = row['PM2.5'].strip()
        if value not in ('', 'NA'):
            float(value)
        series.append((t, value not in ('', 'NA')))
    series.sort()
    counts = Counter(t for t, _ in series)
    duplicate_rows = sum(n - 1 for n in counts.values())
    if not series:
        raise ValueError('Serie vacía')
    first, last = series[0][0], series[-1][0]
    expected = int((last - first).total_seconds() // 3600) + 1
    # Count only unambiguous timestamps, with a full 24-hour history and target.
    valid = {t: ok for t, ok in series if counts[t] == 1}
    target_only = complete = 0
    for t, ok in valid.items():
        hours = [t - timedelta(hours=i) for i in range(1, 25)]
        if ok and all(h in valid for h in hours):
            target_only += 1
            complete += all(valid[h] for h in hours)
    result = {
        'sha256_archivo_original': digest, 'archivo_original': INNER,
        'miembros_zip_interior': len(names), 'csv_dongsi': matches[0],
        'filas': len(series), 'primera_hora': first.isoformat(' '),
        'ultima_hora': last.isoformat(' '), 'marcas_duplicadas': sum(n > 1 for n in counts.values()),
        'filas_duplicadas_extra': duplicate_rows, 'horas_sin_registro': expected - len(counts),
        'filas_pm25_faltante': sum(not ok for _, ok in series),
        'ventanas_24h_objetivo_valido': target_only,
        'ventanas_24h_pm25_completo_y_objetivo_valido': complete,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
