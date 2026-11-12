import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from release_metropt3 import SOURCE, TRAINING, WINNER, install


class ReleaseTests(unittest.TestCase):
    def test_verification_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'package.zip'
            checkpoint = b'synthetic checkpoint'
            checkpoint_hash = hashlib.sha256(checkpoint).hexdigest()
            manifest = {'format_version': 1, 'winner_run_id': WINNER,
                        'phase04_source_commit': SOURCE, 'training_source_commit': TRAINING,
                        'checkpoint_sha256': checkpoint_hash,
                        'files_sha256': {'artifacts/best.pt': checkpoint_hash}}
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('manifest.json', json.dumps(manifest))
                z.writestr('metropt3_model/artifacts/best.pt', checkpoint)
            expected = hashlib.sha256(archive.read_bytes()).hexdigest()
            dest = root / 'model'
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                install(archive, '0' * 64, dest)
            self.assertFalse(dest.exists())
            install(archive, expected, dest)
            self.assertEqual((dest / 'artifacts/best.pt').read_bytes(), checkpoint)
            with self.assertRaisesRegex(ValueError, 'ya existe'):
                install(archive, expected, dest)
            with zipfile.ZipFile(archive, 'a') as z:
                z.writestr('unexpected', b'extra')
            with self.assertRaisesRegex(ValueError, 'inesperado'):
                install(archive, hashlib.sha256(archive.read_bytes()).hexdigest(), root / 'other')


if __name__ == '__main__':
    unittest.main()
