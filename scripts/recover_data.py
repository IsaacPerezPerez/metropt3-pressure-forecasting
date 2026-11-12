import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip'
BUNDLE_HASH = 'b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4'
ORIGINAL_HASH = 'd1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8'
NAME = 'PRSA2017_Data_20130301-20170228.zip'
DEST = Path(__file__).resolve().parents[1] / 'data' / 'raw' / NAME


def main():
    with urllib.request.urlopen(URL, timeout=120) as response:
        bundle = response.read()
    if hashlib.sha256(bundle).hexdigest() != BUNDLE_HASH:
        raise ValueError('SHA-256 del paquete UCI no coincide; no se escribió el archivo')
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        original = archive.read(NAME)
    if hashlib.sha256(original).hexdigest() != ORIGINAL_HASH:
        raise ValueError('SHA-256 del archivo original no coincide; no se escribió el archivo')
    if DEST.exists() and DEST.read_bytes() != original:
        raise FileExistsError(f'El archivo existente difiere; no se sobrescribe: {DEST}')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_bytes(original)
    print(f'Archivo verificado: {DEST}\nSHA-256: {ORIGINAL_HASH}')


if __name__ == '__main__':
    main()
