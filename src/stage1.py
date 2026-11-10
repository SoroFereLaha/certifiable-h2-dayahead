"""Stage 1 (paper Section 4): fit C1 and the baselines, evaluate them on the test period
(H1-H3, sensitivities, C2/H5, optional registry replay H4) and write one results file.

Usage: python3 stage1.py --output ../results/new_stage1.json [--registry]
Models, bin edges and margins are fitted on data before the test period; test rows are only used
for prediction and scoring. The script refuses to overwrite an existing output file.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import core
import weather


def progress(i, n):
    print('{"progress": %d, "total": %d}' % (i, n), flush=True)
    if os.environ.get('RUN_PROGRESS'):
        with open(os.environ['RUN_PROGRESS'], 'w') as stream:
            stream.write('%d %d' % (i, n))


def evaluate_ratio(table, ratio, margin_split='pretest'):
    days = core.day_features(table, ratio)
    train, cal, pre = core.select(days, 'train'), core.select(days, 'calib'), core.select(days, margin_split)
    fitted = {
        'clim': core.fit_clim(train),
        'B1': core.fit_b1(pre),
        'B1p': core.fit_b1prime(pre, ratio),
    }
    model = core.fit_learned(train, core.FEATURES)
    fitted['C1_delta'] = core.calibrate_learned(model, cal, core.FEATURES)
    # The test period is read only from here on.
    test = core.select(days, 'test')
    commits = {
        'B_clim': core.apply_clim(fitted['clim'], test),
        'B0': test['Qhat'].values,
        'B1': core.apply_b1(fitted['B1'], test),
        'B1p': core.apply_b1prime(fitted['B1p'], test, ratio),
        'C1': core.apply_learned(model, fitted['C1_delta'], test, core.FEATURES),
        'oracle': test['Q'].values,
    }
    out = {'ratio': ratio, 'fitted': {k: (v if not isinstance(v, float) else float(v)) for k, v in fitted.items()},
           'n': {s: int(len(core.select(days, s))) for s in ('train', 'calib', 'pretest', 'test')},
           'pooled': {}, 'by_year': {}, 'b0_bias_test': float((test['Qhat'] - test['Q']).mean())}
    for name, c in commits.items():
        out['pooled'][name] = core.metrics(test['Q'].values, c)
    for year in core.TEST_YEARS:
        sub = core.select(test, year)
        mask = test.index.isin(sub.index)
        out['by_year'][year] = {name: core.metrics(test['Q'].values[mask], c[mask]) for name, c in commits.items()}
    losses = {name: core.pinball(test['Q'].values, c) for name, c in commits.items()}
    out['gain'] = {'C1_vs_B1p': core.bootstrap_gain(losses['C1'], losses['B1p']),
                   'C1_vs_B1': core.bootstrap_gain(losses['C1'], losses['B1']),
                   'C1_vs_B0': core.bootstrap_gain(losses['C1'], losses['B0'])}
    monthly = (test['Qhat'] - test['Q']).groupby([test.index.year, test.index.month])
    out['monthly_cancellation_B0'] = float(monthly.sum().abs().sum() / monthly.apply(lambda e: e.abs().sum()).sum())
    out['_losses'] = losses
    out['_commits'] = commits
    out['_test_index'] = [d.strftime('%Y-%m-%d') for d in test.index]
    return out


SENSITIVITIES = {  # primary ratio only, same rules, no re-tuning
    'S1_price25': {'exception_price': 25},
    'S1b_price30': {'exception_price': 30},
    'S2_alpha10': {'alpha': 0.10},
    'S3_DK1_offshore': {'asset': ('DK1', 'off')},
    'S4_DK2_onshore': {'asset': ('DK2', 'on')},
    'S5_margins_2024': {'margin_split': 'calib'},
}


def evaluate_sensitivity(frame, table, cfg):
    if 'exception_price' in cfg or 'asset' in cfg:
        table = core.build_days(frame, exception_price=cfg.get('exception_price', core.EXCEPTION_PRICE),
                              asset=cfg.get('asset', ('DK1', 'on')))
    saved = core.ALPHA
    core.ALPHA = cfg.get('alpha', saved)
    try:
        r = evaluate_ratio(table, core.PRIMARY_RATIO, cfg.get('margin_split', 'pretest'))
    finally:
        core.ALPHA = saved
    test = core.select(table, 'test')
    return {'config': {k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()},
            'excluded_test_days': int((~test['complete']).sum()), 'n': r['n'], 'pooled': r['pooled'],
            'gain': r['gain'], 'b0_bias_test': r['b0_bias_test']}


def evaluate_c2(table, ratio, wx):
    base = core.day_features(table, ratio)
    # Weather exists from 2024-03 only: C1 and B1' keep their full pre-test history;
    # C2 training, the shared calibration of C1/C2 and the test use the days that have weather.
    days = base.join(wx, how='inner')
    feats = core.FEATURES + weather.FEATURES
    c1 = core.fit_learned(core.select(base, 'train'), core.FEATURES)
    c2_train, c2_cal = core.select(days, 'c2_train'), core.select(days, 'c2_calib')
    c2 = core.fit_learned(c2_train, feats)
    fitted = {'C1_delta': core.calibrate_learned(c1, c2_cal, core.FEATURES),
              'C2_delta': core.calibrate_learned(c2, c2_cal, feats),
              'B1p': core.fit_b1prime(core.select(base, 'c2_pretest_b'), ratio)}
    test = core.select(days, 'c2_test')
    commits = {'B1p': core.apply_b1prime(fitted['B1p'], test, ratio),
               'C1': core.apply_learned(c1, fitted['C1_delta'], test, core.FEATURES),
               'C2': core.apply_learned(c2, fitted['C2_delta'], test, feats)}
    losses = {k: core.pinball(test['Q'].values, c) for k, c in commits.items()}
    return {'ratio': ratio, 'n': {'c2_train': len(c2_train), 'c2_calib': len(c2_cal), 'c2_test': len(test)},
            'fitted': {k: (float(v) if isinstance(v, float) else v) for k, v in fitted.items()},
            'pooled': {k: core.metrics(test['Q'].values, c) for k, c in commits.items()},
            'gain': {'C2_vs_B1p': core.bootstrap_gain(losses['C2'], losses['B1p']),
                     'C2_vs_C1': core.bootstrap_gain(losses['C2'], losses['C1'])}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--registry', action='store_true')
    args = parser.parse_args()
    if os.path.exists(args.output):
        sys.exit('refusing to overwrite ' + args.output)
    started = time.time()
    steps = 5 + len(core.RATIOS) + (1 if args.registry else 0)
    frame = core.load_hourly()
    progress(1, steps)
    table = core.build_days(frame)
    progress(2, steps)
    results = {'k_floor_count': table.attrs['k_floor_count'], 'excluded_days': {}, 'ratios': {}}
    for split in ('train', 'calib', 'test'):
        part = core.select(table, split)
        results['excluded_days'][split] = {'total': int(len(part)), 'excluded': int((~part['complete']).sum())}
    for i, ratio in enumerate(core.RATIOS):
        results['ratios'][str(ratio)] = evaluate_ratio(table, ratio)
        progress(3 + i, steps)
    r = {k: results['ratios'][str(k)] for k in core.RATIOS}
    bias = {k: abs(v['b0_bias_test']) for k, v in r.items()}
    hi, lo = max(bias, key=bias.get), min(bias, key=bias.get)
    results['H3'] = {'abs_b0_bias': {str(k): v for k, v in bias.items()},
                     'gain_C1_vs_B1p': {str(k): v['gain']['C1_vs_B1p']['G'] for k, v in r.items()},
                     'largest_bias_ratio': hi, 'smallest_bias_ratio': lo,
                     'diff_hi_minus_lo': core.bootstrap_gain_difference(
                         (r[hi]['_losses']['C1'], r[hi]['_losses']['B1p']),
                         (r[lo]['_losses']['C1'], r[lo]['_losses']['B1p']))}
    results['sensitivities'] = {}
    for name, cfg in SENSITIVITIES.items():
        results['sensitivities'][name] = evaluate_sensitivity(frame, table, cfg)
    progress(3 + len(core.RATIOS), steps)
    wx = weather.load_daily()
    results['C2'] = evaluate_c2(table, core.PRIMARY_RATIO, wx)
    progress(4 + len(core.RATIOS), steps)
    if args.registry:
        import registry
        primary = r[core.PRIMARY_RATIO]
        test = core.select(core.day_features(table, core.PRIMARY_RATIO), 'test')
        results['H4'] = registry.run(test, {k: primary['_commits'][k] for k in ('C1', 'B1p')},
                                     os.path.join(os.path.dirname(os.path.abspath(args.output)), 'registry_replay'))
        progress(steps, steps)
    for v in r.values():
        for key in ('_losses', '_test_index', '_commits'):
            v.pop(key)
    results['elapsed_s'] = time.time() - started
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'x') as stream:
        json.dump(results, stream, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    print('written', args.output)


if __name__ == '__main__':
    main()
