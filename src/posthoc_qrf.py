"""Quantile regression forest, added after the final test (paper Section 6.2): selected on
the same folds, then run once on the new days and compared with R and C3. Does not change the stage-2 selection.
Usage: python3 posthoc_qrf.py --output ../results/new_qrf.json"""
import json
import os
import sys
import time

import numpy as np

import stage2_common as c
import core

ROOT = os.path.dirname(c.HERE)  # repository root
WINDOWS = ['all', 'w24', 'rw']
FALLBACK = 'w24'


class QRF:
    def fit(self, train, weights=None):
        from sklearn.ensemble import RandomForestRegressor
        x, self.y = train[core.FEATURES].values, train['Q'].values
        self.w = np.ones(len(self.y)) if weights is None else np.asarray(weights, float)
        self.model = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5,
                                           random_state=core.SEED_MODEL, n_jobs=-1)
        self.model.fit(x, self.y, sample_weight=weights)
        self.leaves = self.model.apply(x)  # (n_train, n_trees)
        order = np.argsort(self.y, kind='stable')
        self.y_sorted, self.order = self.y[order], order
        return self

    def predict(self, frame):
        leaves_test = self.model.apply(frame[core.FEATURES].values)
        n_trees = self.leaves.shape[1]
        weights = np.zeros((len(leaves_test), len(self.y)))
        for t in range(n_trees):
            lt, lx = self.leaves[:, t], leaves_test[:, t]
            mass = np.bincount(lt, weights=self.w, minlength=max(lt.max(), lx.max()) + 1)
            same = lx[:, None] == lt[None, :]
            weights += same * (self.w[None, :] / np.where(mass[lx] > 0, mass[lx], 1.0)[:, None])
        weights /= n_trees
        cum = np.cumsum(weights[:, self.order], axis=1)
        cum /= cum[:, -1:]
        idx = (cum < core.ALPHA).sum(axis=1)
        return self.y_sorted[np.minimum(idx, len(self.y) - 1)]


def run_qrf(days, window, train_end, calib_start, calib_end, eval_start, eval_end):
    train, weights = c.train_window(days, window, train_end)
    cal = c.between(days, calib_start, calib_end)
    model = QRF().fit(train, weights)
    delta = core.conformal_shift(model.predict(cal) - cal['Q'].values)
    ev = c.between(days, eval_start, eval_end)
    return core.commit(model.predict(ev), delta), float(delta)


def main():
    out_path = sys.argv[sys.argv.index('--output') + 1]
    if os.path.exists(out_path):
        sys.exit('refusing to overwrite ' + out_path)
    t0 = time.time()
    sel = json.load(open(os.path.join(ROOT, 'results/stage2_selection.json')))
    hold = json.load(open(os.path.join(ROOT, 'results/stage2_final_test.json')))
    frame = core.load_hourly()
    dev_table = core.build_days(frame)
    out = {'status': 'post hoc', 'dev': {}, 'holdout': {}}
    # Development: same folds and reference R as stage 2.
    for ratio in core.RATIOS:
        days = core.day_features(dev_table, ratio)
        q = np.concatenate([c.between(days, f[3], f[4])['Q'].values for f in c.FOLDS.values()])
        r = np.concatenate([c.run_baselines(days, ratio, *f[1:])['R'] for f in c.FOLDS.values()])
        assert np.isclose(core.metrics(q, r)['M1_pinball'], sel['ratios'][str(ratio)]['methods']['R']['M1_pinball'])
        res = {}
        for w in WINDOWS:
            cm = np.concatenate([run_qrf(days, w, *f)[0] for f in c.FOLDS.values()])
            res['qrf/' + w] = core.metrics(q, cm)
            res['qrf/' + w]['G_vs_R_dev'] = core.bootstrap_gain(core.pinball(q, cm), core.pinball(q, r))
            print(ratio, w, json.dumps(res['qrf/' + w]['G_vs_R_dev']), flush=True)
        out['dev'][str(ratio)] = res
    m = out['dev'][str(core.PRIMARY_RATIO)]
    valid = [w for w in WINDOWS if m['qrf/' + w]['M2_wilson95'][0] <= core.ALPHA]
    best = min(valid, key=lambda w: (round(m['qrf/' + w]['M1_pinball'] / 1e-6), WINDOWS.index(w))) if valid else FALLBACK
    out['selected'] = 'qrf/' + best
    # New days: one pass of the selected QRF, compared with R and C3.
    table = c.build_days_until(frame, c.HOLDOUT[1])
    for ratio in core.RATIOS:
        days = core.day_features(table, ratio)
        ev = c.between(days, *c.HOLDOUT)
        h = hold['ratios'][str(ratio)]['daily']
        q = ev['Q'].values
        assert np.allclose(q, h['Q']) and [d.strftime('%Y-%m-%d') for d in ev.index] == h['day']
        cm, delta = run_qrf(days, best, *c.FINAL)
        lq, lr, l3 = (core.pinball(q, v) for v in (cm, np.asarray(h['R']), np.asarray(h['C3'])))
        out['holdout'][str(ratio)] = {'delta': delta, 'QRF': core.metrics(q, cm),
                                      'QRF_vs_R': core.bootstrap_gain(lq, lr), 'QRF_vs_C3': core.bootstrap_gain(lq, l3)}
        print('holdout', ratio, json.dumps(out['holdout'][str(ratio)]['QRF_vs_R']),
              json.dumps(out['holdout'][str(ratio)]['QRF_vs_C3']), flush=True)
    out['elapsed_s'] = time.time() - t0
    json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
