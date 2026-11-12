"""Select completed full runs by unrounded MAE and deterministic tie-breaks."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def select(results):
    expected = {(a, v) for a in ('RNN', 'LSTM') for v in ('original', 'ampliada')}
    if len(results) != 4 or {(r['architecture'], r['variant']) for r in results} != expected:
        raise ValueError('Exactly one full result for each architecture and variant is required')
    if any(r['smoke'] or not r['reload_batch_identical'] for r in results):
        raise ValueError('Smoke or unverified result')
    return min(results, key=lambda r: (r['mae_bar'], r['variant'] != 'original', r['architecture'] != 'RNN'))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_ids', nargs=4)
    args = parser.parse_args()
    results = [json.loads((ROOT / 'artifacts' / run_id / 'result.json').read_text()) for run_id in args.run_ids]
    winner = select(results)
    baseline = winner['persistence_mae_bar']
    for r in results:
        r['difference_mae_bar'] = r['mae_bar'] - baseline
        r['difference_mae_percent'] = 100 * r['difference_mae_bar'] / baseline
    output = {'selected_run_id': winner['run_id'], 'persistence_mae_bar': baseline, 'results': results}
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
