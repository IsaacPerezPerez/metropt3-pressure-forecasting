"""Document the existing local MLflow runs and add the final MAE figure."""
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / 'docs/RESULTADOS_FASE04.json').read_text(encoding='utf-8'))
client = MlflowClient(tracking_uri=(ROOT / 'mlruns').as_uri())
phase03 = client.get_experiment_by_name('metropt3-fase03')
phase04 = client.get_experiment_by_name('metropt3-fase04')
assert phase03.experiment_id == '537000561928021549'
assert phase04.experiment_id == '545137120009382204'
run = client.get_run(RESULTS['evaluation_run_id'])
assert run.info.experiment_id == phase04.experiment_id
assert run.info.run_name == 'evaluacion-final-ganador'
assert run.data.params['winner_run_id'] == RESULTS['winner_run_id']
assert int(run.data.params['cases']) == RESULTS['cases']
winner = client.get_run(RESULTS['winner_run_id'])
assert winner.info.experiment_id == phase03.experiment_id
assert winner.info.run_name == 'lstm-ampliada-completa'
for key in ('lstm_mae_bar', 'lstm_rmse_bar', 'persistence_mae_bar',
            'persistence_rmse_bar', 'mae_improvement_bar', 'mae_improvement_percent'):
    assert run.data.metrics[key] == RESULTS[key], key

desc03 = ("Comparación de RNN y LSTM en validación. Original usa Reservoirs, "
          "Motor_current, Oil_temperature y COMP; ampliada añade la media causal "
          "de tres lecturas y la tasa de cambio de presión. Smoke es una prueba "
          "corta, excluida de la selección. Se eligió el menor MAE de validación "
          "sin redondear: LSTM ampliada. En Runs, filtre params.run_type = 'full' "
          "y en Columns active best_mae_bar y best_rmse_bar; consulte el README "
          "para el recorrido completo.")
desc04 = ("Evaluación en prueba del ganador elegido en fase 03, LSTM ampliada, "
          "frente a persistencia. Consulte el run evaluacion-final-ganador y su "
          "artefacto de comparación de MAE; la selección se hizo antes de prueba.")
desc_run = (f"Pronóstico de Reservoirs a 300 s (objetivo observado más cercano, ±5 s). "
            f"Persistencia usa como pronóstico la presión observada en el ancla. "
            f"Ambos métodos se evaluaron sobre {RESULTS['cases']:,} ventanas de prueba. "
            f"MAE: LSTM ampliada {RESULTS['lstm_mae_bar']:.6f} bar; persistencia "
            f"{RESULTS['persistence_mae_bar']:.6f} bar. Reducción: "
            f"{RESULTS['mae_improvement_percent']:.2f} %. ".replace(',', ' ')
            + "[Abrir corrida ganadora](http://localhost:5000/#/experiments/"
            + f"{phase03.experiment_id}/runs/{RESULTS['winner_run_id']})")
desc_winner = ("Ganadora entre cuatro corridas completas por el menor MAE de "
               "validación: 0.280736 bar (época 5). Las señales ampliadas agregan "
               "media causal y tasa de presión. La prueba se evaluó después de la selección.")

with tempfile.TemporaryDirectory() as temp:
    path = Path(temp) / 'comparacion_mae_prueba.png'
    fig, ax = plt.subplots(figsize=(8, 3.6), layout='constrained')
    values = [RESULTS['lstm_mae_bar'], RESULTS['persistence_mae_bar']]
    bars = ax.barh(['LSTM ampliada', 'Persistencia'], values, color=['#176b87', '#ac6935'], height=0.55)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.30)
    ax.set_xlabel('MAE en prueba (bar)\nPersistencia: presión observada en el ancla · 650 545 ventanas', fontsize=9)
    ax.set_title('Error de pronóstico a 300 s', loc='left', weight='bold')
    ax.bar_label(bars, labels=[f'{value:.6f} bar' for value in values], padding=6)
    ax.xaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    fig.savefig(path, dpi=90)
    plt.close(fig)
    client.set_experiment_tag(phase03.experiment_id, 'mlflow.note.content', desc03)
    client.set_experiment_tag(phase04.experiment_id, 'mlflow.note.content', desc04)
    client.set_tag(run.info.run_id, 'mlflow.note.content', desc_run)
    client.set_tag(winner.info.run_id, 'mlflow.note.content', desc_winner)
    client.log_artifact(run.info.run_id, str(path))
print('Descripciones y gráfica registradas en los experimentos y run existentes.')
