"""Share of Danish days with at least one missing hour in the DK1 onshore day-ahead forecast or
settled production, per data split. The study excludes these days."""
import glob, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'raw')
fc = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, 'Forecasts_Hour_*.csv'))))
fc = fc[(fc['PriceArea'] == 'DK1') & (fc['ForecastType'] == 'Onshore Wind')].set_index('HourUTC')['ForecastDayAhead']
real = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, 'ProductionConsumptionSettlement_*.csv'))))
real = real[real['PriceArea'] == 'DK1'].set_index('HourUTC')
real = real[['OnshoreWindLt50kW_MWh', 'OnshoreWindGe50kW_MWh']].sum(axis=1, min_count=2)
hours = pd.date_range('2020-01-01', '2026-06-30 23:00', freq='h', tz='UTC')
key = hours.strftime('%Y-%m-%dT%H:%M:%S')
ok = pd.Series(fc.reindex(key).notna().values & real.reindex(key).notna().values, index=hours)
day = hours.tz_convert('Europe/Copenhagen').normalize().tz_localize(None)
daily = ok.groupby(day).agg(['all', 'size'])
daily['ok'] = daily['all'] & daily['size'].isin([23, 24, 25])
periods = {'train': ('2020-01-02', '2023-12-31'), 'calib': ('2024-01-01', '2024-12-31'),
           'test_2025': ('2025-01-01', '2025-12-31'), 'test_2026H1': ('2026-01-01', '2026-06-30')}
out = {}
for name, (a, b) in periods.items():
    d = daily.loc[a:b]
    out[name] = {'days': int(len(d)), 'excluded': int((~d['ok']).sum()),
                 'share': float((~d['ok']).mean()),
                 'first_excluded': [str(x.date()) for x in d.index[~d['ok']][:5]]}
out['pass_partial'] = all(v['share'] <= 0.05 for v in out.values() if isinstance(v, dict))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results', 'data_checks', 'missing_hours.json'), 'w'), indent=1)
