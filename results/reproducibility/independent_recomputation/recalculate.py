"""Recomputes the stage-2 metrics with numpy only (no import of the study code) and compares them with results/."""
import json
import math
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
original = root.parent.parent  # results/
comparisons = {}
for stem in ('stage2_selection', 'stage2_final_test'):
    a = json.loads((original / (stem + '.json')).read_text())
    b = json.loads((root / (stem + '.json')).read_text())
    a.pop('elapsed_s', None)
    b.pop('elapsed_s', None)
    comparisons[stem] = a == b

h = json.loads((original / 'stage2_final_test.json').read_text())
d = h['ratios']['0.5']['daily']
q = np.array(d['Q'])
n = len(q)
losses, metrics = {}, {}
z = 1.959963984540054
for name in ('C3', 'R'):
    c = np.array(d[name])
    residual = q - c
    losses[name] = np.where(residual >= 0, .05 * residual, -.95 * residual)
    k = int(np.count_nonzero(c - q > 1e-6))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2*n)) / denom
    radius = z * math.sqrt(p*(1-p)/n + z**2/(4*n*n)) / denom
    metrics[name] = {'n': n, 'k': k, 'pinball': float(losses[name].mean()),
                     'shortfall_rate': p, 'wilson95': [centre-radius, centre+radius]}
    src = h['ratios']['0.5']['pooled'][name]
    assert abs(metrics[name]['pinball'] - src['M1_pinball']) < 1e-14
    assert p == src['M2_shortfall_rate']
    assert np.allclose(metrics[name]['wilson95'], src['M2_wilson95'], atol=1e-14, rtol=0)

# Paired moving-block bootstrap: same blocks for both methods, no wrap-around, cut to n days.
rng = np.random.default_rng(20261023)
starts = rng.integers(0, n-7+1, size=(5000, math.ceil(n/7)))
indices = (starts[..., None] + np.arange(7)).reshape(5000, -1)[:, :n]
draws = 1 - losses['C3'][indices].mean(axis=1) / losses['R'][indices].mean(axis=1)
gain = {'G': float(1-losses['C3'].mean()/losses['R'].mean()),
        'ci95': np.quantile(draws, [.025, .975]).tolist()}
assert np.allclose(gain['ci95'], h['ratios']['0.5']['gain']['C3_vs_R']['ci95'], atol=1e-14, rtol=0)

s = json.loads((original / 'stage2_selection.json').read_text())
m = s['ratios']['0.5']['methods']
valid = [(f, w) for f, w in s['configs'] if m[f+'/'+w]['M2_wilson95'][0] <= .05]
ranked = sorted(valid, key=lambda fw: m['/'.join(fw)]['M1_pinball'])
winner = ranked[0] if ranked else ('hgb', 'w24')
assert list(winner) == s['selection']['C3']
assert [list(v) for v in valid] == s['selection']['valid']
result = {'exact_json_without_elapsed_s': comparisons, 'metrics': metrics,
          'gain': gain, 'date_range': [d['day'][0], d['day'][-1]],
          'selection': {'valid': valid, 'minimum': winner,
                        'ranking': [(list(v),m['/'.join(v)]['M1_pinball']) for v in ranked]},
          'H6': gain['G'] >= .02 and gain['ci95'][0] > 0,
          'H7': metrics['C3']['wilson95'][0] <= .05}
assert all(comparisons.values())
(root / 'independent_metrics.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
