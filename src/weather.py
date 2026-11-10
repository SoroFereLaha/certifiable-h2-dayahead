"""Weather features for the secondary variant C2 only (paper Section 4).

Open-Meteo Previous Runs API, model ecmwf_ifs025, *_previous_day2 = value predicted 48 h before the
valid hour, hence issued before the 18:15 D-1 decision. Data licence: CC BY 4.0 (Open-Meteo).
"""
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
OUT = os.path.join(ROOT, 'data', 'weather')
POINTS = [(57.0, 9.9), (56.5, 8.4), (56.0, 9.5), (55.5, 8.7), (55.3, 9.4)]
START, END = '2024-03-01', '2026-06-30'
VARS = ['wind_speed_100m_previous_day2', 'wind_direction_100m_previous_day2']
FEATURES = ['wx_ws_mean', 'wx_ws_std', 'wx_ws_min', 'wx_ws_max', 'wx_dir_sin', 'wx_dir_cos']


def path(lat, lon):
    return os.path.join(OUT, 'p_%.1f_%.1f.csv' % (lat, lon))


def fetch():
    os.makedirs(OUT, exist_ok=True)
    manifest = {}
    for lat, lon in POINTS:
        params = {'latitude': lat, 'longitude': lon, 'start_date': START, 'end_date': END,
                  'hourly': ','.join(VARS), 'models': 'ecmwf_ifs025', 'timezone': 'GMT', 'wind_speed_unit': 'ms'}
        url = 'https://previous-runs-api.open-meteo.com/v1/forecast?' + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=300) as response:
            hourly = json.load(response)['hourly']
        frame = pd.DataFrame(hourly)
        frame.to_csv(path(lat, lon), index=False)
        manifest['%.1f,%.1f' % (lat, lon)] = {'url': url, 'rows': len(frame),
                                              'sha256': hashlib.sha256(open(path(lat, lon), 'rb').read()).hexdigest()}
        time.sleep(5)
    json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)


def load_daily():
    """Daily features on Danish local days; a day is kept only if every hour is present at every grid point."""
    frames = []
    for lat, lon in POINTS:
        f = pd.read_csv(path(lat, lon))
        f['time'] = pd.to_datetime(f['time'], utc=True)
        frames.append(f.set_index('time'))
    speed = pd.concat([f[VARS[0]] for f in frames], axis=1)
    direction = np.deg2rad(pd.concat([f[VARS[1]] for f in frames], axis=1))
    ok = speed.notna().all(axis=1) & direction.notna().all(axis=1)
    hourly = pd.DataFrame({'ws': speed.mean(axis=1), 'sin': np.sin(direction).mean(axis=1),
                           'cos': np.cos(direction).mean(axis=1), 'ok': ok})
    hourly['day'] = hourly.index.tz_convert('Europe/Copenhagen').normalize().tz_localize(None)
    groups = hourly.groupby('day')
    daily = pd.DataFrame({'wx_ws_mean': groups['ws'].mean(), 'wx_ws_std': groups['ws'].std(),
                          'wx_ws_min': groups['ws'].min(), 'wx_ws_max': groups['ws'].max(),
                          'wx_dir_sin': groups['sin'].mean(), 'wx_dir_cos': groups['cos'].mean(),
                          'ok': groups['ok'].all(), 'n': groups.size()})
    return daily[daily['ok'] & daily['n'].isin([23, 24, 25])][FEATURES]


if __name__ == '__main__':
    fetch()
    sys.exit(0)
