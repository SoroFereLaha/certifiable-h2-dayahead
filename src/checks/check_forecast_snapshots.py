"""Do day-ahead forecasts change after publication? Compares snapshots of the same hours taken
at different times, and with the archived file in data/raw/. A morning snapshot holds 22 hours (UTC 00-21
of the current Danish day): the next day is published at 18:00 Danish time."""
import glob, itertools, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sources = {}
for path in sorted(glob.glob(os.path.join(ROOT, 'data', 'forecast_snapshots', 'dayahead_snapshot_*.json'))):
    recs = json.load(open(path))['records']
    sources[os.path.basename(path)[18:-5]] = pd.Series({r['HourUTC']: r['ForecastDayAhead'] for r in recs
                                                       if r['ForecastDayAhead'] is not None})
arch = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'Forecasts_Hour_2026.csv'))
arch = arch[(arch['PriceArea'] == 'DK1') & (arch['ForecastType'] == 'Onshore Wind')]
sources['archive_fetch'] = arch.set_index('HourUTC')['ForecastDayAhead'].dropna()
out = {'hours_per_source': {k: int(len(v)) for k, v in sources.items() if k != 'archive_fetch'}, 'pairs': {}}
for a, b in itertools.combinations(sources, 2):
    common = sources[a].index.intersection(sources[b].index)
    if a == 'archive_fetch' or b == 'archive_fetch':
        common = common[common >= '2026-10-21']
    if len(common):
        diff = (sources[a][common] - sources[b][common]).abs()
        out['pairs']['%s vs %s' % (a, b)] = {'common_hours': int(len(common)), 'first': common.min(),
                                             'last': common.max(), 'max_abs_diff': float(diff.max())}
out['pass'] = all(p['max_abs_diff'] == 0 for p in out['pairs'].values()) and len(out['pairs']) > 0
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results', 'data_checks', 'forecast_snapshots.json'), 'w'), indent=1)
