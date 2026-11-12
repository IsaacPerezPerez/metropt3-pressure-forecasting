"""Evaluate only the frozen winner and persistence on the reserved partition."""
import hashlib
import json
import zipfile

import mlflow
import numpy as np
import pandas as pd
import torch
from preprocess_metropt3 import BASE, CSV, SHA256, SOURCE, VALID_END, build_cases
from train_metropt3 import Batches, Model, ROOT, evaluate


def main():
    torch.set_num_threads(4)
    frozen = ROOT / 'outputs/fase04_frozen'
    manifest = json.loads((ROOT / 'docs/FASE04_CONGELACION.json').read_text(encoding='utf-8'))
    for name, expected in manifest['artifact_sha256'].items():
        if hashlib.sha256((frozen / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Artefacto congelado alterado: ' + name)
    with SOURCE.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != SHA256:
            raise ValueError('SHA-256 del dataset inesperado')
    with zipfile.ZipFile(SOURCE) as archive, archive.open(CSV) as csv:
        frame = pd.read_csv(csv, usecols=['timestamp', *BASE], parse_dates=['timestamp'])
    frame = frame.loc[frame.timestamp >= VALID_END].reset_index(drop=True)
    cases = build_cases(frame)
    if not len(cases.anchors):
        raise ValueError('Sin casos de prueba')
    ckpt = torch.load(frozen / 'best.pt', map_location='cpu', weights_only=True)
    scaler = ckpt['scaler']
    model = Model('LSTM', 6).eval()
    model.load_state_dict(ckpt['model_state'])
    data = Batches(cases, np.array(scaler['mean']), np.array(scaler['std']), 6, torch.device('cpu'))
    mae, rmse = evaluate(model, data, 512)
    err = cases.values[cases.anchors, 0] - cases.values[cases.targets, 0]
    p_mae, p_rmse = float(np.abs(err).mean()), float(np.sqrt(np.square(err).mean()))
    result = {'winner_run_id': manifest['winner_run_id'], 'checkpoint_sha256': manifest['artifact_sha256']['best.pt'],
              'cases': len(cases.anchors), 'first_anchor': str(cases.timestamps[cases.anchors[0]]),
              'last_anchor': str(cases.timestamps[cases.anchors[-1]]),
              'first_target': str(cases.timestamps[cases.targets[0]]), 'last_target': str(cases.timestamps[cases.targets[-1]]),
              'lstm_mae_bar': mae, 'lstm_rmse_bar': rmse, 'persistence_mae_bar': p_mae,
              'persistence_rmse_bar': p_rmse, 'mae_improvement_bar': p_mae - mae,
              'mae_improvement_percent': 100 * (p_mae - mae) / p_mae}
    mlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())
    mlflow.set_experiment('metropt3-fase04')
    with mlflow.start_run(run_name='evaluacion-final-ganador') as run:
        result['evaluation_run_id'] = run.info.run_id
        mlflow.set_tag('winner_run_id', manifest['winner_run_id'])
        mlflow.log_params({'winner_run_id': manifest['winner_run_id'], 'checkpoint_sha256': result['checkpoint_sha256'],
                           'test_start': '2020-06-01', 'cases': result['cases']})
        mlflow.log_metrics({k: v for k, v in result.items() if k.endswith('_bar') or k.endswith('_percent')})
        path = ROOT / 'docs/RESULTADOS_FASE04.json'
        path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        mlflow.log_artifact(str(path))
        mlflow.log_artifact(str(ROOT / 'docs/FASE04_CONGELACION.json'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
