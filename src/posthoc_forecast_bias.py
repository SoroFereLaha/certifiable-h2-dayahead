"""Yearly mean bias of the TSO day-ahead forecast (forecast minus settled production),
DK1 onshore, in MWh and relative to mean production. Writes results/posthoc_forecast_bias.json."""
import json, os, sys
import core
f = core.load_hourly()
out = {}
for y in range(2020, 2027):
    g = f.loc[str(y)]
    d = (g['fc_DK1_on'] - g['real_DK1_on']).dropna()
    out[str(y)] = {'mean_fc_minus_real_MWh': float(d.mean()), 'relative_to_mean_real': float(d.mean() / g['real_DK1_on'].mean())}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(core.ROOT, 'results', 'posthoc_forecast_bias.json'), 'w'), indent=1)
