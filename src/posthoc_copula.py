"""Gaussian-copula scenarios of hourly forecast errors, added after the final test (paper
Section 6.2): selected on the same folds, then run once on the new days and compared with R and C3.
Usage: python3 posthoc_copula.py --output ../results/new_copula.json"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import norm

import stage2_common as c
import core

ROOT = os.path.dirname(c.HERE)  # repository root
WINDOWS = ['all', 'w24', 'rw']
FALLBACK = 'w24'
N_CLASSES = 10
N_SCEN = 1000
N_POS = 24
JITTER = 1e-4
U_CLIP = 1e-4


def _hours(rows):
    """Concatenate hourly arrays of several days; returns per-hour w, w_hat, day index, position."""
    w = np.concatenate(list(rows['w']))
    wh = np.concatenate(list(rows['w_hat']))
    day = np.concatenate([np.full(len(x), i) for i, x in enumerate(rows['w'])])
    pos = np.concatenate([np.minimum(np.arange(len(x)), N_POS - 1) for x in rows['w']])
    return w, wh, day, pos


class Copula:
    def __init__(self, independent=False):
        self.independent = independent

    def fit(self, rows, weights=None):
        w, wh, day, pos = _hours(rows)
        dw = np.ones(len(rows)) if weights is None else np.asarray(weights, float)
        hw = dw[day]
        self.edges = np.quantile(wh, np.linspace(0, 1, N_CLASSES + 1)[1:-1])
        cls = np.searchsorted(self.edges, wh, side='right')
        err = w - wh
        u = np.empty(err.size)
        self.sorted_err, self.cum = [], []
        for k in range(N_CLASSES):
            m = np.flatnonzero(cls == k)
            order = m[np.argsort(err[m], kind='stable')]
            ww = hw[order]
            total = ww.sum()
            cum = np.cumsum(ww) / total
            u[order] = cum - 0.5 * ww / total  # weighted mid-rank
            self.sorted_err.append(err[order])
            self.cum.append(cum)
        z = norm.ppf(np.clip(u, U_CLIP, 1 - U_CLIP))
        if self.independent:
            self.corr = np.eye(N_POS)
        else:
            full = [i for i, x in enumerate(rows['w']) if len(x) == N_POS]
            zd = np.zeros((len(full), N_POS))
            for j, i in enumerate(full):
                zd[j] = z[day == i]
            wd = dw[full] / dw[full].sum()
            cov = (zd * wd[:, None]).T @ zd
            sd = np.sqrt(np.diag(cov))
            self.corr = cov / np.outer(sd, sd)
        rng = np.random.Generator(np.random.PCG64(core.SEED_MODEL))
        self.normals = rng.standard_normal((N_SCEN, N_POS + 1))
        return self

    def _inverse(self, k, u):
        idx = np.searchsorted(self.cum[k], u, side='left')
        return self.sorted_err[k][np.minimum(idx, self.sorted_err[k].size - 1)]

    def predict(self, rows, ratio):
        out = np.empty(len(rows))
        for j, (wh, exc) in enumerate(zip(rows['w_hat'], rows['exc'])):
            n = len(wh)
            pos = np.minimum(np.arange(n), N_POS - 1)
            sigma = self.corr[np.ix_(pos, pos)] + JITTER * np.eye(n)
            chol = np.linalg.cholesky(sigma)
            u = norm.cdf(self.normals[:, :n] @ chol.T)
            cls = np.searchsorted(self.edges, wh, side='right')
            ws = np.empty((N_SCEN, n))
            for h in range(n):
                ws[:, h] = np.maximum(0.0, wh[h] + self._inverse(cls[h], u[:, h]))
            q = np.where(exc[None, :], ratio, np.minimum(ratio, ws)).sum(axis=1)
            out[j] = np.quantile(q, core.ALPHA)
        return out


def run_cop(table, days, ratio, window, train_end, calib_start, calib_end, eval_start, eval_end, independent=False):
    train, weights = c.train_window(days, window, train_end)
    cal = c.between(days, calib_start, calib_end)
    ev = c.between(days, eval_start, eval_end)
    model = Copula(independent).fit(table.loc[train.index], weights)
    delta = core.conformal_shift(model.predict(table.loc[cal.index], ratio) - cal['Q'].values)
    return core.commit(model.predict(table.loc[ev.index], ratio), delta), float(delta)


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
            cm = np.concatenate([run_cop(dev_table, days, ratio, w, *f)[0] for f in c.FOLDS.values()])
            res['cop/' + w] = core.metrics(q, cm)
            res['cop/' + w]['G_vs_R_dev'] = core.bootstrap_gain(core.pinball(q, cm), core.pinball(q, r))
            print(ratio, w, json.dumps(res['cop/' + w]['G_vs_R_dev']), flush=True)
        out['dev'][str(ratio)] = res
    m = out['dev'][str(core.PRIMARY_RATIO)]
    valid = [w for w in WINDOWS if m['cop/' + w]['M2_wilson95'][0] <= core.ALPHA]
    best = min(valid, key=lambda w: (round(m['cop/' + w]['M1_pinball'] / 1e-6), WINDOWS.index(w))) if valid else FALLBACK
    out['selected'] = 'cop/' + best
    # New days: one pass of the selected copula, compared with R and C3.
    table = c.build_days_until(frame, c.HOLDOUT[1])
    for ratio in core.RATIOS:
        days = core.day_features(table, ratio)
        ev = c.between(days, *c.HOLDOUT)
        h = hold['ratios'][str(ratio)]['daily']
        q = ev['Q'].values
        assert np.allclose(q, h['Q']) and [d.strftime('%Y-%m-%d') for d in ev.index] == h['day']
        cm, delta = run_cop(table, days, ratio, best, *c.FINAL)
        ci, _ = run_cop(table, days, ratio, best, *c.FINAL, independent=True)
        lq, li, lr, l3 = (core.pinball(q, v) for v in (cm, ci, np.asarray(h['R']), np.asarray(h['C3'])))
        out['holdout'][str(ratio)] = {'delta': delta, 'COP': core.metrics(q, cm), 'IND': core.metrics(q, ci),
                                      'COP_vs_R': core.bootstrap_gain(lq, lr), 'COP_vs_C3': core.bootstrap_gain(lq, l3),
                                      'COP_vs_IND': core.bootstrap_gain(lq, li)}
        print('holdout', ratio, json.dumps(out['holdout'][str(ratio)]['COP_vs_R']),
              json.dumps(out['holdout'][str(ratio)]['COP_vs_C3']),
              json.dumps(out['holdout'][str(ratio)]['COP_vs_IND']), flush=True)
    out['elapsed_s'] = time.time() - t0
    json.dump(out, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()
