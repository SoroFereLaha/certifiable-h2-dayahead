"""Download the public Energinet series used in the paper (no key, no account).

Licence: "Conditions for use of Danish public-sector data" (Energi Data Service).
Writes one CSV per dataset and year under data/raw/, plus manifest.json (request URL, row count and
SHA-256 of each file).
"""
import csv
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = 'https://api.energidataservice.dk/dataset/'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
RAW = os.path.join(ROOT, 'data', 'raw')
AREAS = ['DK1', 'DK2']
YEARS = range(2020, 2027)

DATASETS = {
    'Forecasts_Hour': {
        'time': 'HourUTC',
        'columns': ['HourUTC', 'PriceArea', 'ForecastType', 'ForecastDayAhead', 'ForecastIntraday',
                    'ForecastCurrent', 'TimestampUTC'],
    },
    'ProductionConsumptionSettlement': {
        'time': 'HourUTC',
        'columns': ['HourUTC', 'PriceArea', 'OffshoreWindLt100MW_MWh', 'OffshoreWindGe100MW_MWh',
                    'OnshoreWindLt50kW_MWh', 'OnshoreWindGe50kW_MWh', 'SolarPowerLt10kW_MWh',
                    'SolarPowerGe10Lt40kW_MWh', 'SolarPowerGe40kW_MWh', 'SolarPowerSelfConMWh'],
        'years': range(2019, 2027),
    },
    'Elspotprices': {
        'time': 'HourUTC',
        'columns': ['HourUTC', 'PriceArea', 'SpotPriceEUR'],
        'years': range(2020, 2026),
    },
    'DayAheadPrices': {
        'time': 'TimeUTC',
        'columns': ['TimeUTC', 'PriceArea', 'DayAheadPriceEUR'],
        'years': range(2025, 2027),
    },
}


def fetch(dataset, spec, year):
    params = {
        'start': '%d-01-01T00:00' % year,
        'end': '%d-01-01T00:00' % (year + 1),
        'filter': json.dumps({'PriceArea': AREAS}),
        'columns': ','.join(spec['columns']),
        'sort': spec['time'] + ' asc',
        'timezone': 'UTC',
        'limit': 0,
    }
    url = API + dataset + '?' + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                payload = json.load(response)
            if 'records' in payload:
                return url, payload['records']
            wait = 200
        except urllib.error.HTTPError as error:
            wait = 200 if error.code == 429 else 20 * (attempt + 1)
        except (urllib.error.URLError, TimeoutError):
            wait = 20 * (attempt + 1)
        print('retry %s %d in %ds' % (dataset, year, wait), flush=True)
        time.sleep(wait)
    raise RuntimeError('failed: %s %d' % (dataset, year))


def main():
    os.makedirs(RAW, exist_ok=True)
    manifest_path = os.path.join(RAW, 'manifest.json')
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    jobs = [(d, s, y) for d, s in DATASETS.items() for y in s.get('years', YEARS)]
    for index, (dataset, spec, year) in enumerate(jobs, 1):
        name = '%s_%d.csv' % (dataset, year)
        path = os.path.join(RAW, name)
        if name in manifest and os.path.exists(path):
            continue
        url, records = fetch(dataset, spec, year)
        with open(path, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=spec['columns'])
            writer.writeheader()
            for record in records:
                writer.writerow({c: record.get(c) for c in spec['columns']})
        digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        manifest[name] = {'url': url, 'rows': len(records), 'sha256': digest,
                          'fetched_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        json.dump(manifest, open(manifest_path, 'w'), indent=1)
        print('{"progress": %d, "total": %d}' % (index, len(jobs)), flush=True)
        print('%s rows=%d' % (name, len(records)), flush=True)
        time.sleep(15)
    return 0


if __name__ == '__main__':
    sys.exit(main())
