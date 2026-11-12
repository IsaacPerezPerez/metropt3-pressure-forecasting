"""Export a self-contained MLflow model from verified frozen artifacts."""
import hashlib
import json
import shutil
import mlflow.pyfunc
from mlflow_metropt3 import MetroModel
from train_metropt3 import ROOT


def main():
    frozen = ROOT / 'outputs/fase04_frozen'
    manifest = json.loads((ROOT / 'docs/FASE04_CONGELACION.json').read_text(encoding='utf-8'))
    for name, expected in manifest['artifact_sha256'].items():
        if hashlib.sha256((frozen / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Artefacto congelado alterado: ' + name)
    out = ROOT / 'outputs/metropt3_model'
    if out.exists():
        shutil.rmtree(out)
    mlflow.pyfunc.save_model(path=str(out), python_model=MetroModel(),
        artifacts={'checkpoint': str(frozen / 'best.pt'), 'scaler': str(frozen / 'scaler.json'),
                   'schema': str(frozen / 'schema.json')},
        code_paths=[str(ROOT / 'scripts' / name) for name in ('mlflow_metropt3.py', 'inference_metropt3.py', 'train_metropt3.py', 'preprocess_metropt3.py')],
        pip_requirements=['mlflow==2.22.0', 'torch==2.7.1', 'pandas==2.2.3', 'numpy==1.26.4'])
    shutil.copy2(ROOT / 'config_fase03.json', out / 'config_fase03.json')
    print(out)


if __name__ == '__main__':
    main()
