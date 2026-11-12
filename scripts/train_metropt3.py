"""Train and select recurrent MetroPT-3 models without reading the test period."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

from preprocess_metropt3 import BASE, DERIVED, HISTORY, SHA256, build_cases, load_development, split_development

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'config_fase03.json').read_text(encoding='utf-8-sig'))


def prepare():
    train_frame, valid_frame = split_development(load_development())
    train, valid = build_cases(train_frame), build_cases(valid_frame)
    assert len(train.anchors) == 440342 and len(valid.anchors) == 405173
    persistence = valid.report()['persistence_mae_bar']
    assert round(persistence, 6) == 0.545120
    # Each training row contributes once, regardless of the number of windows containing it.
    counts = np.zeros(len(train.values) + 1, dtype=np.int64)
    np.add.at(counts, train.anchors - HISTORY + 1, 1)
    np.add.at(counts, train.anchors + 1, -1)
    rows = train.values[np.cumsum(counts[:-1]) > 0]
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    mean[3], std[3] = 0.0, 1.0  # binary COMP
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Invalid training scaler')
    return train, valid, mean, std, persistence


class Model(nn.Module):
    def __init__(self, architecture, n_features):
        super().__init__()
        cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM}[architecture]
        self.recurrent = cls(n_features, CONFIG['hidden_size'], CONFIG['layers'], batch_first=True)
        self.head = nn.Linear(CONFIG['hidden_size'], 1)

    def forward(self, x):
        return self.head(self.recurrent(x)[0][:, -1, :]).squeeze(-1)


class Batches:
    """Keep source rows and indices in device memory; construct only the current batch."""
    def __init__(self, cases, mean, std, features, device):
        values = cases.values[:, :features].copy()
        values = (values - mean[:features]) / std[:features]
        self.values = torch.as_tensor(values, dtype=torch.float32, device=device)
        self.labels = torch.as_tensor(cases.values[cases.targets, 0], dtype=torch.float64, device=device)
        self.anchors = torch.as_tensor(cases.anchors, dtype=torch.long, device=device)
        self.offset = torch.arange(-HISTORY + 1, 1, device=device)
        self.device = device

    def batch(self, indices):
        return self.values[self.anchors[indices, None] + self.offset], self.labels[indices]


def evaluate(model, data, batch_size, limit=None):
    model.eval()
    n = min(len(data.anchors), limit) if limit else len(data.anchors)
    absolute, squared = 0.0, 0.0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = torch.arange(start, min(start + batch_size, n), device=data.device)
            x, y = data.batch(idx)
            errors = (model(x).double() - y.double())
            absolute += errors.abs().sum().item()
            squared += errors.square().sum().item()
    return absolute / n, (squared / n) ** 0.5


def run(architecture, augmented, device, smoke=False):
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.use_deterministic_algorithms(True)
    train, valid, mean, std, persistence = prepare()
    features = 6 if augmented else 4
    columns = list((BASE + DERIVED)[:features])
    train_data = Batches(train, mean, std, features, device)
    valid_data = Batches(valid, mean, std, features, device)
    model = Model(architecture, features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    kind = 'ampliada' if augmented else 'original'
    name = f'{architecture.lower()}-{kind}-' + ('smoke' if smoke else 'completa')
    mlflow.set_tracking_uri((ROOT / 'mlruns').as_uri())
    mlflow.set_experiment('metropt3-fase03')
    with mlflow.start_run(run_name=name) as active:
        run_id = active.info.run_id
        out = ROOT / 'artifacts' / run_id
        out.mkdir(parents=True, exist_ok=False)
        scaler = {'columns': columns, 'mean': mean[:features].tolist(), 'std': std[:features].tolist(),
                  'fit': CONFIG['scaling']}
        (out / 'scaler.json').write_text(json.dumps(scaler, indent=2), encoding='utf-8')
        (out / 'schema.json').write_text(json.dumps({'columns': columns, 'history': HISTORY, 'target': 'Reservoirs',
            'target_unit': 'bar', 'horizon_seconds': 300}, indent=2), encoding='utf-8')
        (out / 'requirements-freeze.txt').write_text(subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
        script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        source_commit = subprocess.check_output(['git', '-c', 'safe.directory=*', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        source_dirty = bool(subprocess.check_output(['git', '-c', 'safe.directory=*', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip())
        params = dict(CONFIG, architecture=architecture, variant=kind, columns=json.dumps(columns),
                      dataset_sha256=SHA256, train_end='2020-04-01', validation_end='2020-06-01',
                      test_start='2020-06-01', device=str(device), device_name=(torch.cuda.get_device_name(0)
                      if device.type == 'cuda' else 'CPU'), torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
                      source_commit=source_commit, source_dirty=source_dirty, script_sha256=script_hash, run_type='smoke' if smoke else 'full')
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metric('persistence_mae_bar', persistence)
        best = float('inf')
        start_time = time.perf_counter()
        epochs = 1 if smoke else CONFIG['epochs']
        n_train = min(4096, len(train.anchors)) if smoke else len(train.anchors)
        val_limit = 4096 if smoke else None
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(n_train, device=device)
            loss_total = 0.0
            for start in range(0, n_train, CONFIG['batch_size']):
                x, y = train_data.batch(order[start:start + CONFIG['batch_size']])
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                loss = nn.functional.mse_loss(prediction, y.float())
                loss.backward()
                optimizer.step()
                loss_total += loss.item() * len(y)
            mae, rmse = evaluate(model, valid_data, CONFIG['batch_size'], val_limit)
            elapsed = time.perf_counter() - epoch_start
            mlflow.log_metrics({'train_mse': loss_total / n_train, 'val_mae_bar': mae,
                                'val_rmse_bar': rmse, 'epoch_seconds': elapsed}, step=epoch + 1)
            print(f'{name} epoch={epoch + 1} mae={mae:.9f} rmse={rmse:.9f} seconds={elapsed:.2f}', flush=True)
            if mae < best:
                best, best_rmse, best_epoch = mae, rmse, epoch + 1
                model.eval()
                check_x, _ = valid_data.batch(torch.arange(min(CONFIG['batch_size'], len(valid.anchors)), device=device))
                with torch.no_grad():
                    best_predictions = model(check_x).detach().clone()
                torch.save({'model_state': model.state_dict(), 'architecture': architecture,
                            'features': features, 'scaler': scaler, 'epoch': best_epoch}, out / 'best.pt')
        checkpoint = torch.load(out / 'best.pt', map_location=device, weights_only=True)
        reloaded = Model(architecture, features).to(device)
        reloaded.load_state_dict(checkpoint['model_state'])
        reloaded.eval()
        x, _ = valid_data.batch(torch.arange(min(CONFIG['batch_size'], len(valid.anchors)), device=device))
        with torch.no_grad():
            identical = torch.equal(best_predictions, reloaded(x))
        if not identical:
            raise RuntimeError('Checkpoint reload predictions differ')
        duration = time.perf_counter() - start_time
        result = {'run_id': run_id, 'name': name, 'architecture': architecture, 'variant': kind,
                  'best_epoch': best_epoch, 'mae_bar': best, 'rmse_bar': best_rmse,
                  'persistence_mae_bar': persistence, 'elapsed_seconds': duration,
                  'reload_batch_identical': identical, 'device': str(device), 'smoke': smoke}
        (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        mlflow.log_metrics({'best_mae_bar': best, 'best_rmse_bar': best_rmse,
                            'best_epoch': best_epoch, 'total_seconds': duration})
        mlflow.log_artifacts(str(out))
        print(json.dumps(result), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', choices=['RNN', 'LSTM'], default='RNN')
    parser.add_argument('--augmented', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu only after diagnosis')
    print(f'torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} device={args.device}', flush=True)
    run(args.architecture, args.augmented, torch.device(args.device), args.smoke)


if __name__ == '__main__':
    main()
