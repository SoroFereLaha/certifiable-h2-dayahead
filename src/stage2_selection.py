"""Stage 2, selection (paper Section 5): choose the C3 configuration (model family and
training window) on the development folds F1-F3. No day after 2026-06-30 is built.
Usage: python3 stage2_selection.py --output ../results/new_selection.json"""
import argparse
import json
import os
import sys
import time

import numpy as np

import stage2_common as c
import core


def progress(i, n):
    print('{"progress": %d, "total": %d}' % (i, n), flush=True)
    if os.environ.get('RUN_PROGRESS'):
        with open(os.environ['RUN_PROGRESS'], 'w') as stream:
            stream.write('%d %d' % (i, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        sys.exit('refusing to overwrite ' + args.output)
    t0 = time.time()
    table = core.build_days(core.load_hourly())
    assert table.index.max() <= c.core.pd.Timestamp(c.DEV_END)
    out = {'configs': [list(k) for k in c.CONFIGS],
           'ratios': {}}
    total, step = len(core.RATIOS) * (len(c.CONFIGS) + 1), 0
    for ratio in core.RATIOS:
        days = core.day_features(table, ratio)
        q = np.concatenate([c.between(days, f[3], f[4])['Q'].values for f in c.FOLDS.values()])
        res = {'n_eval': int(q.size), 'methods': {}, 'per_fold': {}}
        base = {k: [] for k in ('R', 'B1_12m', 'B0')}
        for name, f in c.FOLDS.items():
            for k, v in c.run_baselines(days, ratio, *f[1:]).items():
                base[k].append(v)
        commits = {k: np.concatenate(v) for k, v in base.items()}
        step += 1
        progress(step, total)
        for family, window in c.CONFIGS:
            parts, info = [], {}
            for name, f in c.FOLDS.items():
                cm, delta, n_train = c.run_candidate(days, family, window, *f)
                parts.append(cm)
                info[name] = {'delta': delta, 'n_train': n_train}
            commits['%s/%s' % (family, window)] = np.concatenate(parts)
            res['per_fold']['%s/%s' % (family, window)] = info
            step += 1
            progress(step, total)
        losses = {k: core.pinball(q, v) for k, v in commits.items()}
        for k, v in commits.items():
            res['methods'][k] = core.metrics(q, v)
            if k != 'R':
                res['methods'][k]['G_vs_R_dev'] = core.bootstrap_gain(losses[k], losses['R'])
        out['ratios'][str(ratio)] = res
    # Selection rule (E/W = 0.5): valid configurations, lowest pooled pinball, ties by list order.
    m = out['ratios'][str(core.PRIMARY_RATIO)]['methods']
    valid = [k for k in c.CONFIGS if m['%s/%s' % k]['M2_wilson95'][0] <= core.ALPHA]
    if valid:
        best = min(valid, key=lambda k: (round(m['%s/%s' % k]['M1_pinball'] / 1e-6), c.CONFIGS.index(k)))
        rule = 'lowest pooled pinball among valid configurations'
    else:
        best, rule = c.FALLBACK, 'no valid configuration: fixed fallback'
    out['selection'] = {'C3': list(best), 'rule': rule, 'valid': [list(k) for k in valid]}
    out['elapsed_s'] = time.time() - t0
    with open(args.output, 'w') as stream:
        json.dump(out, stream, indent=1)
    print(json.dumps(out['selection']))


if __name__ == '__main__':
    main()
