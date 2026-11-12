"""Build or install the versioned offline MLflow bundle with SHA-256 checks."""
import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
WINNER = '13e41b225750413a976d09f36a91ab69'
SOURCE = '84bf8bf8c41b242cfc66c66e05b79a12c9b5169f'
TRAINING = '8db576e7cc03f20ed629c0d79679b684922ad9d0'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(model, destination):
    frozen = json.loads((ROOT / 'docs/FASE04_CONGELACION.json').read_text(encoding='utf-8'))
    if frozen['winner_run_id'] != WINNER or frozen['source_commit'] != TRAINING:
        raise ValueError('La procedencia congelada no coincide')
    files = {p.relative_to(model).as_posix(): p.read_bytes()
             for p in model.rglob('*') if p.is_file()}
    for name in ('best.pt', 'scaler.json', 'schema.json'):
        if digest(files['artifacts/' + name]) != frozen['artifact_sha256'][name]:
            raise ValueError('Artefacto congelado alterado: ' + name)
    for name in ('inference_metropt3.py', 'mlflow_metropt3.py',
                 'preprocess_metropt3.py', 'train_metropt3.py'):
        if files['code/' + name] != (ROOT / 'scripts' / name).read_bytes():
            raise ValueError('Código exportado distinto: ' + name)
    manifest = {'format_version': 1, 'winner_run_id': WINNER,
                'phase04_source_commit': SOURCE, 'training_source_commit': TRAINING,
                'checkpoint_sha256': frozen['artifact_sha256']['best.pt'],
                'files_sha256': {name: digest(data) for name, data in sorted(files.items())}}
    entries = {'manifest.json': (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode(),
               **{'metropt3_model/' + name: data for name, data in files.items()}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    print(f'SHA-256 {digest(destination.read_bytes())}  {destination}')


def install(archive_path, expected, destination):
    data = archive_path.read_bytes()
    if digest(data) != expected.lower():
        raise ValueError('SHA-256 del ZIP no coincide')
    if destination.exists():
        raise ValueError('El destino ya existe; retírelo o elija otro')
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Entradas duplicadas en ZIP')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['format_version'] != 1 or manifest['winner_run_id'] != WINNER or manifest['phase04_source_commit'] != SOURCE or manifest['training_source_commit'] != TRAINING:
            raise ValueError('Procedencia inesperada')
        files = manifest['files_sha256']
        if set(names) != {'manifest.json'} | {'metropt3_model/' + n for n in files}:
            raise ValueError('Contenido inesperado en ZIP')
        if files.get('artifacts/best.pt') != manifest['checkpoint_sha256']:
            raise ValueError('Hash del checkpoint inconsistente')
        for name, expected_file in files.items():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
                raise ValueError('Ruta inválida en ZIP')
            if digest(archive.read('metropt3_model/' + name)) != expected_file:
                raise ValueError('Archivo alterado: ' + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
            staged = Path(tmp) / 'model'
            for name in files:
                target = staged.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read('metropt3_model/' + name))
            shutil.move(str(staged), str(destination))
    print(f'Paquete verificado e instalado en {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    make = sub.add_parser('build')
    make.add_argument('zip', type=Path)
    make.add_argument('--model', type=Path, default=ROOT / 'outputs/metropt3_model')
    put = sub.add_parser('install')
    put.add_argument('zip', type=Path)
    put.add_argument('--sha256', required=True)
    put.add_argument('--destination', type=Path, default=ROOT / 'outputs/metropt3_model')
    args = parser.parse_args()
    if args.action == 'build':
        build(args.model, args.zip)
    else:
        install(args.zip, args.sha256, args.destination)


if __name__ == '__main__':
    main()
