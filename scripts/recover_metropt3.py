"""Download the official bundle safely and verify its pinned SHA-256."""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

URL = 'https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip'
SHA256 = 'aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a'
DEST = Path(__file__).resolve().parents[1] / 'data/raw/metropt3-dataset.zip'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        if sha256(DEST) != SHA256:
            raise FileExistsError(f'El archivo existente difiere: {DEST}')
        print(f'Archivo ya verificado: {DEST}')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=DEST.parent, delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(URL, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        actual = sha256(temporary)
        if actual != SHA256:
            raise ValueError(f'SHA-256 inesperado: {actual}')
        os.replace(temporary, DEST)
        print(f'Archivo verificado: {DEST}\nSHA-256: {actual}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
