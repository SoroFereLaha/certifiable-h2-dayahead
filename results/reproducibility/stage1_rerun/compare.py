"""Rerun check: exact (zero-tolerance) comparison of main run vs rerun, excluding only elapsed_s."""
import hashlib
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(H))  # results/
a = json.load(open(os.path.join(R, 'stage1.json')))
b = json.load(open(os.path.join(H, 'stage1.json')))
diffs, n = [], [0]


def walk(x, y, p):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if p == '' and k == 'elapsed_s':
                continue
            if k not in x or k not in y:
                diffs.append((p + '/' + k, 'missing', k in x, k in y))
            else:
                walk(x[k], y[k], p + '/' + k)
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, '%s[%d]' % (p, i))
    else:
        n[0] += 1
        if type(x) is not type(y) or x != y:
            if not (isinstance(x, float) and x != x and isinstance(y, float) and y != y):  # NaN == NaN
                diffs.append((p, x, y))


walk(a, b, '')


def canon(d):
    d = {k: v for k, v in d.items() if k != 'elapsed_s'}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


print(json.dumps({'leaves_compared': n[0], 'n_diffs': len(diffs), 'diffs': diffs[:50],
                  'canon_sha_main': canon(a), 'canon_sha_rerun': canon(b),
                  'registry_csv_sha_main': fsha(os.path.join(R, 'registry_replay', 'published_commitments.csv')),
                  'registry_csv_sha_rerun': fsha(os.path.join(H, 'registry_replay', 'published_commitments.csv')),
                  'published_sha256_main': a['H4']['published_sha256'],
                  'published_sha256_rerun': b['H4']['published_sha256'],
                  'gas_median': [a['H4']['gas_median'], b['H4']['gas_median']],
                  'elapsed_s': [a['elapsed_s'], b['elapsed_s']]},
                 indent=1, default=str))
