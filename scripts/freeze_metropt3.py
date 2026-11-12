"""Freeze the selected phase 03 artifacts before opening the test partition."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from preprocess_metropt3 import BASE, DERIVED, HISTORY
from train_metropt3 import Batches, Model, ROOT, prepare

RUN = '13e41b225750413a976d09f36a91ab69'
SOURCE = ROOT / 'artifacts' / RUN
TRACKED = ROOT / 'mlruns/537000561928021549' / RUN / 'artifacts'
OUT = ROOT / 'outputs/fase04_frozen'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    names = ['best.pt', 'scaler.json', 'schema.json', 'result.json', 'requirements-freeze.txt']
    hashes = {name: digest(SOURCE / name) for name in names}
    if any(hashes[name] != digest(TRACKED / name) for name in names):
        raise ValueError('Los artefactos locales y MLflow difieren')
    checkpoint = torch.load(SOURCE / 'best.pt', map_location='cpu', weights_only=True)
    scaler = json.loads((SOURCE / 'scaler.json').read_text())
    schema = json.loads((SOURCE / 'schema.json').read_text())
    assert checkpoint['architecture'] == 'LSTM' and checkpoint['features'] == 6 and checkpoint['epoch'] == 5
    assert checkpoint['scaler'] == scaler and schema['columns'] == list(BASE + DERIVED) and schema['history'] == HISTORY
    _, valid, mean, std, _ = prepare()
    np.testing.assert_allclose(mean, scaler['mean'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(std, scaler['std'], rtol=0, atol=1e-12)
    model = Model('LSTM', 6).eval()
    model.load_state_dict(checkpoint['model_state'])
    data = Batches(valid, mean, std, 6, torch.device('cpu'))
    x, _ = data.batch(torch.arange(512))
    with torch.no_grad():
        first = model(x)
        reloaded = Model('LSTM', 6).eval()
        reloaded.load_state_dict(torch.load(TRACKED / 'best.pt', map_location='cpu', weights_only=True)['model_state'])
        second = reloaded(x)
    assert torch.equal(first, second)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(SOURCE / name, OUT / name)
    manifest = {'winner_run_id': RUN, 'source_commit': '8db576e7cc03f20ed629c0d79679b684922ad9d0',
                'artifact_sha256': hashes, 'scaler': scaler, 'schema': schema,
                'validation_batch_size': 512, 'validation_batch_local_mlflow_identical': True,
                'validation_batch_prediction_sha256_float32': hashlib.sha256(first.numpy().tobytes()).hexdigest(),
                'rules': {'raw_readings': 32, 'interval_seconds_inclusive': [8, 15], 'naive_timestamps': True,
                          'features': '30 rows; mean of current and two preceding Reservoirs; rate over actual preceding interval',
                          'target': 'nearest to t+300 s within 5 s, earlier on tie; same segment and partition',
                          'finite': 'all six features in window and target Reservoirs'}}
    (ROOT / 'docs/FASE04_CONGELACION.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
