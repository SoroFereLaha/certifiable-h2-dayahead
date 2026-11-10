"""Shared definitions: data loading, certifiable energy Q_D under hourly matching, baselines
(B_clim, B0, B1, B1'), learned model C1, conformal shift, pinball loss and block bootstrap.

All quantities are in p.u.*h of the PPA asset (DK1 onshore wind, normalised by K_D). The operating
policy is the same for every method; methods only differ in the commitment c_D fixed at 18:15 Danish
time on D-1.
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
RAW = os.path.join(ROOT, 'data', 'raw')
TZ = 'Europe/Copenhagen'
ALPHA = 0.05
SEED_MODEL = 20261022
SEED_BOOT = 20261023
RATIOS = (0.3, 0.5, 0.8)
PRIMARY_RATIO = 0.5
EXCEPTION_PRICE = 20.0
K_QUANTILE = 0.999
K_LAG_DAYS = 16
K_WINDOW_DAYS = 365
K_FLOOR_MWH = 1.0
N_BINS = 5
BLOCK_DAYS = 7
N_BOOT = 5000

SPLITS = {
    'train': ('2020-01-01', '2023-12-31'),
    'calib': ('2024-01-01', '2024-12-31'),
    'pretest': ('2020-01-01', '2024-12-31'),
    'test': ('2025-01-01', '2026-06-30'),
    'c2_train': ('2024-03-10', '2024-12-31'),
    'c2_calib': ('2025-01-01', '2025-06-30'),
    'c2_pretest_b': ('2020-01-01', '2025-06-30'),
    'c2_test': ('2025-07-01', '2026-06-30'),
}
TEST_YEARS = {'2025': ('2025-01-01', '2025-12-31'), '2026H1': ('2026-01-01', '2026-06-30')}

FC_TYPES = {'Onshore Wind': 'on', 'Offshore Wind': 'off', 'Solar': 'sol'}
REAL_COLUMNS = {
    'on': ['OnshoreWindLt50kW_MWh', 'OnshoreWindGe50kW_MWh'],
    'off': ['OffshoreWindLt100MW_MWh', 'OffshoreWindGe100MW_MWh'],
    'sol': ['SolarPowerLt10kW_MWh', 'SolarPowerGe10Lt40kW_MWh', 'SolarPowerGe40kW_MWh'],
}
# Series used: primary asset plus the forecast features (paper Section 4).
SERIES = [('DK1', 'on'), ('DK1', 'off'), ('DK2', 'on'), ('DK2', 'off'), ('DK1', 'sol')]


# ---------------------------------------------------------------- loading

def _read(pattern):
    files = sorted(glob.glob(os.path.join(RAW, pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_hourly():
    """One row per UTC hour: day-ahead forecasts, realised production, DK1 and DK2 prices."""
    fc = _read('Forecasts_Hour_*.csv')
    fc = fc[fc['ForecastType'].isin(FC_TYPES)]
    fc['col'] = 'fc_' + fc['PriceArea'] + '_' + fc['ForecastType'].map(FC_TYPES)
    fc['HourUTC'] = pd.to_datetime(fc['HourUTC'], utc=True)
    fc = fc.drop_duplicates(['HourUTC', 'col']).pivot(index='HourUTC', columns='col', values='ForecastDayAhead')

    pr = _read('ProductionConsumptionSettlement_*.csv')
    pr['HourUTC'] = pd.to_datetime(pr['HourUTC'], utc=True)
    real = {}
    for area in ('DK1', 'DK2'):
        part = pr[pr['PriceArea'] == area].drop_duplicates('HourUTC').set_index('HourUTC')
        for kind, cols in REAL_COLUMNS.items():
            # A row counts only if every component is present (no silent zero-fill).
            real['real_%s_%s' % (area, kind)] = part[cols].sum(axis=1, min_count=len(cols))
    real = pd.DataFrame(real)

    frame = real.join(fc, how='outer').join(hourly_prices('DK1'), how='outer').join(hourly_prices('DK2'), how='outer')
    frame = frame.sort_index()
    frame.index.name = 'HourUTC'
    return frame


def hourly_prices(area='DK1'):
    """Day-ahead price of one area per UTC hour; 15-min values (from 2025-10-01) are averaged per hour."""
    parts = []
    try:
        spot = _read('Elspotprices_*.csv')
        spot = spot[spot['PriceArea'] == area]
        spot = pd.Series(spot['SpotPriceEUR'].values, index=pd.to_datetime(spot['HourUTC'], utc=True))
        parts.append(spot[spot.index < pd.Timestamp('2025-09-30T22:00', tz='UTC')])
    except FileNotFoundError:
        pass
    try:
        dap = _read('DayAheadPrices_*.csv')
        dap = dap[dap['PriceArea'] == area]
        quarter = pd.Series(dap['DayAheadPriceEUR'].values, index=pd.to_datetime(dap['TimeUTC'], utc=True))
        quarter = quarter[quarter.index >= pd.Timestamp('2025-09-30T22:00', tz='UTC')]
        grouped = quarter.groupby(quarter.index.floor('h'))
        hourly = grouped.mean()[grouped.count() == 4]  # an hour needs its four quarters
        parts.append(hourly)
    except FileNotFoundError:
        pass
    price = pd.concat(parts).sort_index()
    price = price[~price.index.duplicated()]
    return price.rename('price_' + area)


# ---------------------------------------------------------------- daily construction

def local_day(index):
    return index.tz_convert(TZ).normalize().tz_localize(None)


def capacity_proxy(hourly_real, days):
    """K_D: 0.999 quantile of hourly realised values over the 365 days ending D-16 (local days)."""
    day_of_hour = local_day(hourly_real.index)
    values = hourly_real.values
    out = {}
    for day in days:
        end = day - pd.Timedelta(days=K_LAG_DAYS)
        start = end - pd.Timedelta(days=K_WINDOW_DAYS - 1)
        mask = (day_of_hour >= start) & (day_of_hour <= end)
        window = values[mask]
        window = window[~np.isnan(window)]
        out[day] = np.quantile(window, K_QUANTILE) if window.size >= 0.9 * K_WINDOW_DAYS * 24 else np.nan
    return pd.Series(out)


def certifiable(w, exc, ratio):
    """Q = sum_t [E if exception hour else min(E, w_t)] in p.u.*h."""
    w = np.asarray(w, dtype=float)
    exc = np.asarray(exc, dtype=bool)
    return float(np.sum(np.where(exc, ratio, np.minimum(ratio, w))))


def build_days(frame, exception_price=EXCEPTION_PRICE, asset=('DK1', 'on')):
    """Daily table: realised Q, plug-in Q-hat and the closed feature list (paper Section 4)."""
    frame = frame.copy()
    frame['day'] = local_day(frame.index)
    days = pd.DatetimeIndex(sorted(frame['day'].unique()))
    days = days[(days >= pd.Timestamp('2020-01-01')) & (days <= pd.Timestamp('2026-06-30'))]
    capacity = {}
    for area, kind in set(SERIES) | {asset}:
        capacity[(area, kind)] = capacity_proxy(frame['real_%s_%s' % (area, kind)], days)

    area, kind = asset
    rows, floor_count = [], 0
    groups = frame.groupby('day')
    for day in days:
        hours = groups.get_group(day) if day in groups.groups else None
        row = {'day': day, 'n_hours': 0 if hours is None else len(hours), 'complete': False}
        if hours is None:
            rows.append(row)
            continue
        k = capacity[asset][day]
        price_col = 'price_' + area  # exception hours use the bidding zone of the PPA asset
        needed = hours[['real_%s_%s' % asset, 'fc_%s_%s' % asset, price_col]]
        if np.isnan(k) or needed.isna().any().any() or len(hours) not in (23, 24, 25):
            rows.append(row)
            continue
        w = hours['real_%s_%s' % asset].values / k
        w_hat = hours['fc_%s_%s' % asset].values / k
        price = hours[price_col].values
        exc = price <= exception_price
        row.update(complete=True, w=w, w_hat=w_hat, exc=exc, price_mean=price.mean(), price_min=price.min())
        # Secondary forecasts: the four SERIES other than the asset, in SERIES order (sec0..sec3).
        for i, other in enumerate(o for o in SERIES if o != asset):
            series = hours['fc_%s_%s' % other]
            k_other = capacity[other][day]
            if series.isna().any() or np.isnan(k_other):
                row['complete'] = False
                break
            if k_other <= K_FLOOR_MWH:
                floor_count += 1
                row['sec%d' % i] = 0.0
            else:
                row['sec%d' % i] = float(series.sum() / k_other)
        rows.append(row)
    table = pd.DataFrame(rows).set_index('day')
    table.attrs['k_floor_count'] = floor_count
    return table


def day_features(table, ratio):
    """Q, Q-hat and features for one electrolyser ratio (complete days only)."""
    t = table[table['complete']].copy()
    out = pd.DataFrame(index=t.index)
    out['n_hours'] = t['n_hours']
    out['Q'] = [certifiable(w, e, ratio) for w, e in zip(t['w'], t['exc'])]
    out['Qhat'] = [certifiable(w, e, ratio) for w, e in zip(t['w_hat'], t['exc'])]
    capped = [np.minimum(ratio, wh)[~e] for wh, e in zip(t['w_hat'], t['exc'])]
    out['f_sum_free'] = [c.sum() for c in capped]
    out['f_std_free'] = [c.std() if c.size else 0.0 for c in capped]
    out['f_n_exc'] = [int(e.sum()) for e in t['exc']]
    out['f_n_above'] = [int(np.sum(wh > ratio)) for wh in t['w_hat']]
    out['f_first6'] = [wh[:6].mean() for wh in t['w_hat']]
    out['f_last6'] = [wh[-6:].mean() for wh in t['w_hat']]
    for i in range(len(SERIES) - 1):
        out['f_sec%d' % i] = t['sec%d' % i].astype(float)
    out['f_price_mean'] = t['price_mean']
    out['f_price_min'] = t['price_min']
    doy = out.index.dayofyear
    out['f_doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    out['f_doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    return out


FEATURES = ['Qhat', 'f_sum_free', 'f_std_free', 'f_n_exc', 'f_n_above', 'f_first6', 'f_last6',
            'f_sec0', 'f_sec1', 'f_sec2', 'f_sec3', 'f_price_mean', 'f_price_min',
            'f_doy_sin', 'f_doy_cos']


def select(frame, split):
    start, end = SPLITS[split] if split in SPLITS else TEST_YEARS[split]
    return frame[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]


# ---------------------------------------------------------------- methods

def conformal_shift(scores, alpha=None):
    """delta = ceil((n+1)(1-alpha))-th smallest score; infinite if n is too small."""
    alpha = ALPHA if alpha is None else alpha
    scores = np.sort(np.asarray(scores, dtype=float))
    rank = math.ceil((scores.size + 1) * (1 - alpha))
    return np.inf if rank > scores.size else float(scores[rank - 1])


def commit(qhat, delta):
    return np.maximum(0.0, np.asarray(qhat, dtype=float) - delta)


def fit_b1(cal):
    return {'delta': conformal_shift(cal['Qhat'] - cal['Q'])}


def apply_b1(model, frame):
    return commit(frame['Qhat'], model['delta'])


def _bin_value(frame, ratio):
    return frame['Qhat'] / (frame['n_hours'] * ratio)


def fit_b1prime(cal, ratio):
    value = _bin_value(cal, ratio)
    edges = np.quantile(value, np.linspace(0, 1, N_BINS + 1)[1:-1])
    bins = np.searchsorted(edges, value, side='right')
    deltas = [conformal_shift((cal['Qhat'] - cal['Q'])[bins == b]) for b in range(N_BINS)]
    return {'edges': edges.tolist(), 'deltas': deltas}


def apply_b1prime(model, frame, ratio):
    bins = np.searchsorted(np.asarray(model['edges']), _bin_value(frame, ratio), side='right')
    return commit(frame['Qhat'], np.asarray(model['deltas'])[bins])


def fit_clim(train):
    return {int(m): float(np.quantile(g['Q'], ALPHA)) for m, g in train.groupby(train.index.month)}


def apply_clim(model, frame):
    return np.array([model[m] for m in frame.index.month], dtype=float)


def fit_learned(train, features, seed=SEED_MODEL):
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(loss='quantile', quantile=ALPHA, max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0,
                                          random_state=seed)
    model.fit(train[features].values, train['Q'].values)
    return model


def calibrate_learned(model, cal, features):
    return conformal_shift(model.predict(cal[features].values) - cal['Q'].values)


def apply_learned(model, delta, frame, features):
    return commit(model.predict(frame[features].values), delta)


# ---------------------------------------------------------------- metrics

def pinball(q, c, alpha=None):
    alpha = ALPHA if alpha is None else alpha
    q, c = np.asarray(q, float), np.asarray(c, float)
    return alpha * np.maximum(0, q - c) + (1 - alpha) * np.maximum(0, c - q)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def metrics(q, c):
    q, c = np.asarray(q, float), np.asarray(c, float)
    short = np.maximum(0, c - q)
    k = int(np.sum(short > 1e-6))
    return {'M1_pinball': float(pinball(q, c).mean()), 'M2_shortfall_rate': k / q.size,
            'M2_wilson95': wilson(k, q.size), 'M3_commit_ratio': float(c.mean() / q.mean()),
            'M4_short_over_commit': float(short.sum() / c.sum()) if c.sum() > 0 else np.nan, 'n_days': int(q.size)}


def block_indices(n, rng, block=BLOCK_DAYS):
    starts = rng.integers(0, n - block + 1, size=math.ceil(n / block))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def bootstrap_gain(losses_a, losses_b, n_boot=N_BOOT, seed=SEED_BOOT):
    """Gain G = 1 - mean(a)/mean(b) with paired moving-block bootstrap (same blocks)."""
    a, b = np.asarray(losses_a, float), np.asarray(losses_b, float)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_indices(a.size, rng)
        draws[i] = 1 - a[idx].mean() / b[idx].mean()
    point = 1 - a.mean() / b.mean()
    return {'G': float(point), 'ci95': [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


def bootstrap_gain_difference(pair_x, pair_y, n_boot=N_BOOT, seed=SEED_BOOT):
    """Difference of gains between two ratios on the same days (H3)."""
    (ax, bx), (ay, by) = [(np.asarray(a, float), np.asarray(b, float)) for a, b in (pair_x, pair_y)]
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_indices(ax.size, rng)
        draws[i] = (1 - ax[idx].mean() / bx[idx].mean()) - (1 - ay[idx].mean() / by[idx].mean())
    point = (1 - ax.mean() / bx.mean()) - (1 - ay.mean() / by.mean())
    return {'diff': float(point), 'ci95': [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}
