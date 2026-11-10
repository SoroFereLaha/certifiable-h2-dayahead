"""Stage 2 (paper Sections 4-5): candidate grid, development folds F1-F3 and helpers
shared by stage2_selection.py, stage2_final_test.py and the posthoc_* scripts."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
import core  # noqa: E402

DEV_END = '2026-06-30'
HOLDOUT = ('2026-07-01', '2026-09-30')
FAMILIES = ['hgb', 'gbr', 'lin', 'knn']
WINDOWS = ['all', 'w24', 'rw']
CONFIGS = [(f, w) for f in FAMILIES for w in WINDOWS if not (f == 'knn' and w == 'rw')]
FALLBACK = ('hgb', 'w24')
KNN_K = 60
HALF_LIFE_DAYS = 365.0
# Each fold: (train_end, calib_start, calib_end, eval_start, eval_end)
FOLDS = {
    'F1': ('2023-12-31', '2024-01-01', '2024-12-31', '2025-01-01', '2025-06-30'),
    'F2': ('2024-06-30', '2024-07-01', '2025-06-30', '2025-07-01', '2025-12-31'),
    'F3': ('2024-12-31', '2025-01-01', '2025-12-31', '2026-01-01', '2026-06-30'),
}
FINAL = ('2025-06-30', '2025-07-01', '2026-06-30') + HOLDOUT


def between(frame, start, end):
    return frame[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]


def train_window(days, window, train_end):
    end = pd.Timestamp(train_end)
    start = pd.Timestamp('2020-01-01') if window != 'w24' else end - pd.DateOffset(months=24) + pd.Timedelta(days=1)
    train = between(days, start, end)
    weights = None
    if window == 'rw':
        age = (end - train.index).days.values.astype(float)
        weights = 0.5 ** (age / HALF_LIFE_DAYS)
    return train, weights


class Candidate:
    def __init__(self, family):
        self.family = family

    def fit(self, train, weights=None):
        x, y = train[core.FEATURES].values, train['Q'].values
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        if self.family == 'hgb':
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model = HistGradientBoostingRegressor(loss='quantile', quantile=core.ALPHA, max_iter=400,
                                                       learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
                                                       l2_regularization=1.0, random_state=core.SEED_MODEL)
            self.model.fit(x, y, sample_weight=weights)
        elif self.family == 'gbr':
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(loss='quantile', alpha=core.ALPHA, n_estimators=300,
                                                   learning_rate=0.05, max_depth=3, min_samples_leaf=30,
                                                   subsample=0.8, random_state=core.SEED_MODEL)
            self.model.fit(x, y, sample_weight=weights)
        elif self.family == 'lin':
            from sklearn.linear_model import QuantileRegressor
            self.model = QuantileRegressor(quantile=core.ALPHA, alpha=0.001, solver='highs')
            self.model.fit((x - self.mu) / self.sd, y, sample_weight=weights)
        elif self.family == 'knn':
            if weights is not None:
                raise ValueError('knn takes no weights')
            self.xs, self.y = (x - self.mu) / self.sd, y
        return self

    def predict(self, frame):
        x = frame[core.FEATURES].values
        if self.family in ('hgb', 'gbr'):
            return self.model.predict(x)
        z = (x - self.mu) / self.sd
        if self.family == 'lin':
            return self.model.predict(z)
        d = ((z[:, None, :] - self.xs[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argsort(d, axis=1, kind='stable')[:, :KNN_K]
        return np.quantile(self.y[nearest], core.ALPHA, axis=1)


def run_candidate(days, family, window, train_end, calib_start, calib_end, eval_start, eval_end):
    """Fit on the train window, conformal shift on the calibration window, commit on the evaluation window."""
    train, weights = train_window(days, window, train_end)
    cal = between(days, calib_start, calib_end)
    model = Candidate(family).fit(train, weights)
    delta = core.conformal_shift(model.predict(cal) - cal['Q'].values)
    ev = between(days, eval_start, eval_end)
    return core.commit(model.predict(ev), delta), float(delta), len(train)


def run_baselines(days, ratio, calib_start, calib_end, eval_start, eval_end):
    cal = between(days, calib_start, calib_end)
    ev = between(days, eval_start, eval_end)
    return {'R': core.apply_b1prime(core.fit_b1prime(cal, ratio), ev, ratio),
            'B1_12m': core.apply_b1(core.fit_b1(cal), ev),
            'B0': ev['Qhat'].values}


def build_days_until(frame, end, exception_price=core.EXCEPTION_PRICE, asset=('DK1', 'on')):
    """core.build_days with its fixed end date 2026-06-30 replaced by `end`, to extend the day table to the
    new days. stage2_final_test.py checks that both give identical rows up to 2026-06-30."""
    import inspect
    src = inspect.getsource(core.build_days)
    literal = "pd.Timestamp('2026-06-30')"
    if src.count(literal) != 1:
        raise RuntimeError('unexpected core.build_days source')
    src = src.replace('def build_days(', 'def _build_days_until(_end, ', 1).replace(literal, 'pd.Timestamp(_end)')
    namespace = dict(vars(core))
    exec(src, namespace)
    return namespace['_build_days_until'](end, frame, exception_price=exception_price, asset=asset)
