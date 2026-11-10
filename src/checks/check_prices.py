"""Prices: share of excluded days once prices are added (per split and per year), and no missing
hour after converting the 15-min prices to hourly, including the 30 Sep / 1 Oct 2025 switch."""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
price = core.hourly_prices()
hours = pd.date_range('2020-01-01', '2026-06-30 23:00', freq='h', tz='UTC')
p = price.reindex(hours)
out['price_gaps'] = {'missing_hours_2020_2026H1': int(p.isna().sum()),
              'missing_hours_switch_week': int(p['2025-09-27':'2025-10-04'].isna().sum()),
              'first_hourly_from_quarters': str(price[price.index >= '2025-10-01'].index.min()),
              'min_price': float(p.min()), 'max_price': float(p.max())}
frame = core.load_hourly()
table = core.build_days(frame)
out['k_floor_count'] = table.attrs['k_floor_count']
out['excluded_days'] = {}
for split in ('train', 'calib', 'pretest', 'test', 'c2_train', 'c2_calib', 'c2_test'):
    part = core.select(table, split)
    out['excluded_days'][split] = {'days': int(len(part)), 'excluded': int((~part['complete']).sum()),
                         'share': float((~part['complete']).mean())}
for year in core.TEST_YEARS:
    part = core.select(core.select(table, 'test'), year)
    out['excluded_days'][year] = {'days': int(len(part)), 'excluded': int((~part['complete']).sum()),
                        'share': float((~part['complete']).mean())}
out['pass'] = all(v['share'] <= 0.05 for v in out['excluded_days'].values()) and out['price_gaps']['missing_hours_switch_week'] == 0
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results', 'data_checks', 'prices.json'), 'w'), indent=1)
