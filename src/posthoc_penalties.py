"""Multiple comparisons and penalty ratios (paper Section 6.2). Reads the daily Q, C3 and R
values in results/stage2_final_test.json; fits nothing.
1. recomputes the reported gain and its 95% interval (same bootstrap, same seed);
2. Bonferroni interval for the 3 electrolyser sizes (98.33% two-sided) and Holm correction on one-sided
   bootstrap p-values;
3. gain of the same commitments under other shortfall/surplus penalty ratios.
Writes results/posthoc_penalties.json."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import core  # noqa: E402  (pinball, block_indices, seeds)

H = json.load(open(os.path.join(os.path.dirname(HERE), 'results', 'stage2_final_test.json')))
out = {'source': 'results/stage2_final_test.json', 'note': 'post hoc; commitments unchanged'}


def draws_gain(la, lb):
    rng = np.random.Generator(np.random.PCG64(core.SEED_BOOT))
    d = np.empty(core.N_BOOT)
    for i in range(core.N_BOOT):
        idx = core.block_indices(la.size, rng)
        d[i] = 1 - la[idx].mean() / lb[idx].mean()
    return d


pv = {}
for r, v in H['ratios'].items():
    q, c3, rr = (np.asarray(v['daily'][k], float) for k in ('Q', 'C3', 'R'))
    la, lb = core.pinball(q, c3), core.pinball(q, rr)
    d = draws_gain(la, lb)
    g = 1 - la.mean() / lb.mean()
    rep = v['gain']['C3_vs_R']
    a = 0.05 / 3
    res = {'G': g, 'ci95': [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
           'reproduces_reported': bool(abs(g - rep['G']) < 1e-12 and np.allclose(
               [np.quantile(d, .025), np.quantile(d, .975)], rep['ci95'])),
           'ci_bonferroni3': [float(np.quantile(d, a / 2)), float(np.quantile(d, 1 - a / 2))],
           'p_one_sided': float((np.sum(d <= 0) + 1) / (d.size + 1)),
           'M1_C3': float(la.mean()), 'M1_R': float(lb.mean()), 'n_days': int(q.size)}
    # same commitments, scored with other penalty ratios (shortfall : surplus = (1-a)/a)
    res['penalty_ratios'] = {}
    for alpha in (0.02, 0.05, 0.10, 0.20):
        la2, lb2 = core.pinball(q, c3, alpha), core.pinball(q, rr, alpha)
        d2 = draws_gain(la2, lb2)
        res['penalty_ratios']['%g' % ((1 - alpha) / alpha)] = {
            'alpha': alpha, 'G': float(1 - la2.mean() / lb2.mean()),
            'ci95': [float(np.quantile(d2, .025)), float(np.quantile(d2, .975))]}
    out[r] = res
    pv[r] = res['p_one_sided']

# Holm correction across the three electrolyser sizes
order = sorted(pv, key=pv.get)
holm, running = {}, 0.0
for i, r in enumerate(order):
    running = max(running, min(1.0, (3 - i) * pv[r]))
    holm[r] = running
out['holm_adjusted_p'] = holm
json.dump(out, open(os.path.join(os.path.dirname(HERE), 'results', 'posthoc_penalties.json'), 'w'), indent=1)
for r in H['ratios']:
    o = out[r]
    print(r, 'G %.3f' % o['G'], 'ci95', np.round(o['ci95'], 3), 'repro', o['reproduces_reported'],
          'bonf', np.round(o['ci_bonferroni3'], 3), 'p1 %.4f holm %.4f' % (o['p_one_sided'], holm[r]),
          'M1 C3 %.4f R %.4f' % (o['M1_C3'], o['M1_R']))
    print('   ratios', {k: (round(x['G'], 3), [round(y, 3) for y in x['ci95']]) for k, x in o['penalty_ratios'].items()})
