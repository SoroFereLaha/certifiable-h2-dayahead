"""Stage 2, final test (paper Section 5): one pass of the selected C3 and of the
baselines on the new days 2026-07-01 -> 2026-09-30.
Usage: python3 stage2_final_test.py --selection ../results/stage2_selection.json --output ../results/new_final_test.json"""
import argparse
import json
import os
import sys
import time

import numpy as np

import stage2_common as c
import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        sys.exit('refusing to overwrite ' + args.output)
    t0 = time.time()
    family, window = json.load(open(args.selection))['selection']['C3']
    frame = core.load_hourly()
    table = c.build_days_until(frame, c.HOLDOUT[1])
    dev = core.build_days(frame)
    for ratio in core.RATIOS:  # the extended day table must match core.build_days on development days
        a = core.day_features(dev, ratio)
        b = c.between(core.day_features(table, ratio), '2020-01-01', c.DEV_END)
        assert a.index.equals(b.index) and np.allclose(a[['Q'] + core.FEATURES].values, b[['Q'] + core.FEATURES].values)
    out = {'C3': [family, window], 'holdout': list(c.HOLDOUT), 'ratios': {}}
    hold_all = c.between(table, *c.HOLDOUT)
    out['excluded_days'] = {'total': int(len(hold_all)), 'excluded': int((~hold_all['complete']).sum())}
    for ratio in core.RATIOS:
        days = core.day_features(table, ratio)
        ev = c.between(days, *c.HOLDOUT)
        q = ev['Q'].values
        commits = c.run_baselines(days, ratio, *c.FINAL[1:])
        commits['C3'], delta, n_train = c.run_candidate(days, family, window, *c.FINAL)
        # C1 and B1' fitted as in stage 1 (trained to 2023, calibrated on 2024 / 2020-2024).
        train, cal, pre = core.select(days, 'train'), core.select(days, 'calib'), core.select(days, 'pretest')
        model = core.fit_learned(train, core.FEATURES)
        commits['C1_stage1'] = core.apply_learned(model, core.calibrate_learned(model, cal, core.FEATURES), ev, core.FEATURES)
        commits['B1p_stage1'] = core.apply_b1prime(core.fit_b1prime(pre, ratio), ev, ratio)
        commits['oracle'] = q
        losses = {k: core.pinball(q, v) for k, v in commits.items()}
        res = {'n_days': int(q.size), 'C3_delta': delta, 'C3_n_train': n_train,
               'pooled': {k: core.metrics(q, v) for k, v in commits.items()},
               'gain': {'C3_vs_R': core.bootstrap_gain(losses['C3'], losses['R']),
                        'C3_vs_C1_stage1': core.bootstrap_gain(losses['C3'], losses['C1_stage1']),
                        'C1_stage1_vs_R': core.bootstrap_gain(losses['C1_stage1'], losses['R']),
                        'C1_stage1_vs_B1p_stage1': core.bootstrap_gain(losses['C1_stage1'], losses['B1p_stage1'])},
               'daily': {'day': [d.strftime('%Y-%m-%d') for d in ev.index], 'Q': q.tolist(),
                         'C3': commits['C3'].tolist(), 'R': commits['R'].tolist()}}
        out['ratios'][str(ratio)] = res
    p = out['ratios'][str(core.PRIMARY_RATIO)]
    g = p['gain']['C3_vs_R']
    out['H6_pass'] = bool(g['G'] >= 0.02 and g['ci95'][0] > 0)
    out['H7_pass'] = bool(p['pooled']['C3']['M2_wilson95'][0] <= core.ALPHA)
    out['elapsed_s'] = time.time() - t0
    with open(args.output, 'w') as stream:
        json.dump(out, stream, indent=1)
    print(json.dumps({'H6_pass': out['H6_pass'], 'H7_pass': out['H7_pass'], 'G': g}))


if __name__ == '__main__':
    main()
