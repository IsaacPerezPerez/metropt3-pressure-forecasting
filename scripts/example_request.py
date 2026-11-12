"""Send an executable synthetic request; no dataset is required."""
import datetime
import json
import urllib.request


def sample():
    start = datetime.datetime(2020, 7, 1)
    return [{'timestamp': (start + datetime.timedelta(seconds=10 * i)).isoformat(),
             'Reservoirs': 9.0 + i * 0.01, 'Motor_current': 1.5,
             'Oil_temperature': 60.0, 'COMP': 1} for i in range(32)]


def main():
    payload = json.dumps({'readings': sample()}).encode()
    req = urllib.request.Request('http://localhost:8000/predict', data=payload,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        print(response.read().decode())


if __name__ == '__main__':
    main()
