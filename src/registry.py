"""Commitment registry (H4): replay of the commitments on a local EVM (Anvil).

The script writes each day's commitment to the CommitmentRegistry contract, as the producer would at D-1,
then the certified quantity, as the meter adapter would at settlement. It reads the chain state back and
compares it with registry_replay/published_commitments.csv. It also checks that the contract rejects a
second commitment or settlement for the same day, a settlement without a commitment, and calls from an
outside account.
"""
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = 1_000_000


def to_int(x):
    return int(round(float(x) * SCALE))


class Chain:
    def __init__(self):
        from web3 import Web3
        for exe in ('anvil', 'solc'):
            if not shutil.which(exe):
                raise RuntimeError(exe + ' is required')
        source = os.path.join(HERE, 'CommitmentRegistry.sol')
        out = subprocess.run(['solc', '--optimize', '--evm-version', 'cancun', '--combined-json', 'abi,bin', source],
                             check=True, capture_output=True, text=True)
        compiled = next(iter(json.loads(out.stdout)['contracts'].values()))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(['anvil', '--silent', '--port', str(port)], stdout=self.log, stderr=self.log)
        self.w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:%d' % port, request_kwargs={'timeout': 30}))
        for _ in range(200):
            if self.w3.is_connected():
                break
            time.sleep(0.05)
        else:
            self.close()
            raise RuntimeError('anvil did not start')
        self.admin, self.producer, self.adapter, self.outsider = self.w3.eth.accounts[:4]
        factory = self.w3.eth.contract(abi=compiled['abi'], bytecode=compiled['bin'])
        receipt = self.wait(factory.constructor().transact({'from': self.admin}))
        self.c = self.w3.eth.contract(address=receipt.contractAddress, abi=compiled['abi'])
        self.wait(self.c.functions.setProducer(self.producer, True).transact({'from': self.admin}))
        self.wait(self.c.functions.setAdapter(self.adapter, True).transact({'from': self.admin}))

    def wait(self, tx):
        return self.w3.eth.wait_for_transaction_receipt(tx)

    def try_tx(self, fn, sender):
        """Return True if the transaction succeeded, False if it reverted."""
        try:
            fn.call({'from': sender})
        except Exception:
            return False
        receipt = self.wait(fn.transact({'from': sender, 'gas': 300000}))
        return bool(receipt.status)

    def close(self):
        self.process.terminate()
        self.process.wait(timeout=10)


def method_id(name):
    return hashlib.sha256(name.encode()).digest()


def run(test, commitments, out_dir):
    """test: DataFrame with Q indexed by day; commitments: {method: array aligned with test}."""
    os.makedirs(out_dir, exist_ok=True)
    published = os.path.join(out_dir, 'published_commitments.csv')
    with open(published, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['day', 'method', 'commit_micro', 'certified_micro'])
        for name, values in commitments.items():
            for day, c, q in zip(test.index, values, test['Q'].values):
                writer.writerow([day.strftime('%Y-%m-%d'), name, to_int(c), to_int(q)])
    chain = Chain()
    gas, refusals = [], {}
    try:
        with open(published, newline='') as stream:
            rows = list(csv.DictReader(stream))
        day_no = {d: i for i, d in enumerate(sorted({r['day'] for r in rows}))}
        # Commitments are written first (decision at D-1), settlements later (metered data at ~D+15).
        for r in rows:
            fn = chain.c.functions.commit(method_id(r['method']), day_no[r['day']], int(r['commit_micro']),
                                          hashlib.sha256(r['day'].encode()).digest())
            gas.append(chain.wait(fn.transact({'from': chain.producer})).gasUsed)
        for r in rows:
            fn = chain.c.functions.settle(method_id(r['method']), day_no[r['day']], int(r['certified_micro']),
                                          hashlib.sha256((r['day'] + 'meter').encode()).digest())
            gas.append(chain.wait(fn.transact({'from': chain.adapter})).gasUsed)
        # Read the chain state back and compare it with the published file.
        mismatches = 0
        for r in rows:
            entry = chain.c.functions.entries(method_id(r['method']), day_no[r['day']]).call()
            expected = max(0, int(r['commit_micro']) - int(r['certified_micro']))
            if not (entry[3] and entry[4] and entry[0] == int(r['commit_micro'])
                    and entry[1] == int(r['certified_micro']) and entry[2] == expected):
                mismatches += 1
        first = rows[0]
        m, d = method_id(first['method']), day_no[first['day']]
        refusals['double_commit'] = not chain.try_tx(chain.c.functions.commit(m, d, 1, b'\0' * 32), chain.producer)
        refusals['double_settle'] = not chain.try_tx(chain.c.functions.settle(m, d, 1, b'\0' * 32), chain.adapter)
        fresh = len(day_no) + 1
        refusals['settle_without_commit'] = not chain.try_tx(
            chain.c.functions.settle(m, fresh, 1, b'\0' * 32), chain.adapter)
        refusals['commit_by_outsider'] = not chain.try_tx(
            chain.c.functions.commit(m, fresh, 1, b'\0' * 32), chain.outsider)
        refusals['settle_by_outsider'] = not chain.try_tx(
            chain.c.functions.settle(m, fresh, 1, b'\0' * 32), chain.outsider)
    finally:
        chain.close()
    return {'rows': len(rows), 'mismatches': mismatches, 'refusals': refusals,
            'pass': mismatches == 0 and all(refusals.values()),
            'gas_median': float(np.median(gas)), 'published_sha256': sha256_file(published)}


def sha256_file(path):
    with open(path, 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()
