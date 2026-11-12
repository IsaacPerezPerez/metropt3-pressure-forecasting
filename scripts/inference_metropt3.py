"""Raw-reading contract shared by the exported MLflow model and API."""
import numpy as np
import pandas as pd
import torch
from preprocess_metropt3 import BASE, TRAIN_END, VALID_END, build_cases


def predict_readings(model, scaler, readings):
    if not isinstance(readings, list) or len(readings) != 32:
        raise ValueError('Se requieren exactamente 32 lecturas consecutivas')
    required = {'timestamp', *BASE}
    if any(not isinstance(row, dict) or set(row) != required for row in readings):
        raise ValueError('Cada lectura requiere timestamp, Reservoirs, Motor_current, Oil_temperature y COMP')
    for row in readings:
        if not isinstance(row['timestamp'], str):
            raise ValueError('timestamp debe ser texto ISO 8601 sin zona horaria')
        for name in BASE:
            if isinstance(row[name], bool) or not isinstance(row[name], (int, float)) or not np.isfinite(row[name]):
                raise ValueError(f'{name} debe ser numérico y finito')
    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime([row['timestamp'] for row in readings], format='ISO8601'))
    except (ValueError, TypeError) as exc:
        raise ValueError('timestamp debe ser ISO 8601 válido y sin zona horaria') from exc
    if timestamps.tz is not None or timestamps.hasnans:
        raise ValueError('timestamp debe ser válido y sin zona horaria')
    delta = np.diff(timestamps.asi8) / 1e9
    if ((delta < 8) | (delta > 15)).any():
        raise ValueError('Las lecturas deben ser crecientes con intervalos de 8 a 15 segundos')
    if any(timestamps[0] < cut <= timestamps[-1] for cut in (TRAIN_END, VALID_END)):
        raise ValueError('Las lecturas no pueden cruzar los cortes de partición')
    frame = pd.DataFrame(readings)
    frame['timestamp'] = timestamps
    values = build_cases(frame).values[-30:]
    if not np.isfinite(values).all():
        raise ValueError('Las características calculadas deben ser finitas')
    x = (values - np.asarray(scaler['mean'])) / np.asarray(scaler['std'])
    with torch.no_grad():
        prediction = model(torch.as_tensor(x[None], dtype=torch.float32)).item()
    return {'prediction_bar': prediction, 'target': 'Reservoirs', 'horizon_seconds': 300,
            'anchor_timestamp': timestamps[-1].isoformat(),
            'forecast_timestamp': (timestamps[-1] + pd.Timedelta(seconds=300)).isoformat()}
