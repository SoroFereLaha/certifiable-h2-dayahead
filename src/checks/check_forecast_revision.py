"""Were the stored day-ahead forecasts revised after publication? On 2020-2024 (DK1 onshore),
the day-ahead forecast should have a higher MAE than the later "current" forecast."""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data', 'raw')

fc = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, 'Forecasts_Hour_202[0-4].csv'))))
fc = fc[(fc['PriceArea'] == 'DK1') & (fc['ForecastType'] == 'Onshore Wind')].set_index('HourUTC')
real = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, 'ProductionConsumptionSettlement_202[0-4].csv'))))
real = real[real['PriceArea'] == 'DK1'].set_index('HourUTC')
real = real['OnshoreWindLt50kW_MWh'] + real['OnshoreWindGe50kW_MWh']
j = fc.join(real.rename('real'), how='inner').dropna(subset=['ForecastDayAhead', 'ForecastCurrent', 'real'])
out = {'hours': int(len(j))}
for year, g in j.groupby(j.index.str[:4]):
    out[year] = {'mae_da': float((g['ForecastDayAhead'] - g['real']).abs().mean()),
                 'mae_current': float((g['ForecastCurrent'] - g['real']).abs().mean()),
                 'identical_share': float((g['ForecastDayAhead'] == g['ForecastCurrent']).mean())}
out['pass'] = all(v['mae_da'] >= v['mae_current'] for k, v in out.items() if k[:2] == '20')
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results', 'data_checks', 'forecast_revision.json'), 'w'), indent=1)
