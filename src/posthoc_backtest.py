"""Backward test (paper Section 6.2): the selected C3 configuration and R are refitted as of
2023 and as of 2024 and evaluated on those years.
Usage: python3 posthoc_backtest.py --output ../results/new_backtest.json"""
import json
import os
import sys
import time

import numpy as np

import stage2_common as c
import core

ROOT = os.path.dirname(c.HERE)  # repository root
C3 = ('lin', 'rw')
BACK = {
    'Y2023': ('2021-12-31', '2022-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    'Y2024': ('2022-12-31', '2023-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
}


def main():
    out_path = sys.argv[sys.argv.index('--output') + 1]
    if os.path.exists(out_path):
        sys.exit('refusing to overwrite ' + out_path)
    t0 = time.time()
    sel = json.load(open(os.path.join(ROOT, 'results/stage2_selection.json')))
    assert sel['selection']['C3'] == list(C3)
    hold = json.load(open(os.path.join(ROOT, 'results/stage2_final_test.json')))
    frame = core.load_hourly()
    table = c.build_days_until(frame, c.HOLDOUT[1])
    out = {'status': 'post hoc', 'C3': list(C3),
           'years': {}, 'pooled': {}}
    for ratio in core.RATIOS:
        days = core.day_features(table, ratio)
        # Consistency check: the same calls on the final fold give the new-day commitments of stage2_final_test.py.
        h = hold['ratios'][str(ratio)]['daily']
        c3_final = c.run_candidate(days, *C3, *c.FINAL)[0]
        r_final = c.run_baselines(days, ratio, *c.FINAL[1:])['R']
        assert np.allclose(c3_final, h['C3']) and np.allclose(r_final, h['R'])
        qs, l3s, lrs = [], [], []
        for year, fold in BACK.items():
            ev = c.between(days, fold[3], fold[4])
            q = ev['Q'].values
            c3, delta, n_train = c.run_candidate(days, *C3, *fold)
            r = c.run_baselines(days, ratio, *fold[1:])['R']
            l3, lr = core.pinball(q, c3), core.pinball(q, r)
            out['years'].setdefault(year, {})[str(ratio)] = {
                'n_train': n_train, 'delta': delta, 'C3': core.metrics(q, c3), 'R': core.metrics(q, r),
                'C3_vs_R': core.bootstrap_gain(l3, lr)}
            print(year, ratio, n_train, json.dumps(out['years'][year][str(ratio)]['C3_vs_R']), flush=True)
            qs.append(q)
            l3s.append(l3)
            lrs.append(lr)
        out['pooled'][str(ratio)] = {'n_days': int(sum(len(x) for x in qs)),
                                     'C3_vs_R': core.bootstrap_gain(np.concatenate(l3s), np.concatenate(lrs))}
        print('pooled', ratio, json.dumps(out['pooled'][str(ratio)]), flush=True)
    out['elapsed_s'] = time.time() - t0
    json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
