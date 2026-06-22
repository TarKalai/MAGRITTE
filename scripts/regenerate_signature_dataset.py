"""Structural behavioural-signature fraud for BPIC 2012. The signature is a RARE CONSECUTIVE TRANSITION
between two COMMON activities (a 'forbidden edge' / back-loop a->b->a) injected with a probability that
escalates toward the surrogate transaction. Because both activities are common, *presence* is not a tell;
only models that capture ORDER / transitions / paths can detect it -> exercises GINE's inductive bias and
mirrors real fraud (abnormal navigation paths). Class overlap keeps peak detectability non-trivial."""
import sys, types
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent       # scripts/
REPO = ROOT.parent
APRIL = REPO / 'lib' / 'april'                # vendored april subset
for name, sub in [('april', ''), ('april.processmining', 'processmining'), ('april.generation', 'generation')]:
    m = types.ModuleType(name); m.__path__ = [str(APRIL / sub)] if sub else [str(APRIL)]; sys.modules[name] = m
fs = types.ModuleType('april.fs'); fs.EVENTLOG_DIR = REPO/'local'/'.out'/'eventlogs'; fs.EVENTLOG_CACHE_DIR = REPO/'local'/'.out'/'.cache'/'eventlogs'; sys.modules['april.fs'] = fs
from april.processmining.log import EventLog
from april.processmining.event import Event

SEED = 42
FRAUD_RATE = 0.05
SURROGATE = 'O_ACCEPTED+COMPLETE'
ESC_MAX = 0.15           # modest escalation toward the transaction
MIN_PREFIX = 4
CONTAM_FRAC = 0.20       # legit traces also showing the pattern (caps peak AUC)
XES = REPO / 'data' / 'raw' / 'BPI_Challenge_2012.xes'
OUT = REPO / 'data' / 'processed' / 'bpic_2012_signature.parquet'
EPOCH = pd.Timestamp('2011-10-01 00:00:00', tz='UTC')

log = EventLog.from_xes(str(XES))

# --- choose a structural motif: rare consecutive transition (a->b) between two COMMON activities ---
ntr = len(log.cases)
presence = Counter(); bigram = Counter()
for case in log:
    names = [e.name for e in case.events]
    for a in set(names):
        presence[a] += 1
    for a, b in zip(names, names[1:]):
        bigram[(a, b)] += 1
common = [a for a, c in presence.items() if c / ntr >= 0.30 and a != SURROGATE]   # appear in >=30% of traces
# pick the ordered common pair with the FEWEST consecutive occurrences in the log (rarest transition)
cand = sorted(((bigram.get((a, b), 0), a, b) for a in common for b in common if a != b))
_, A, B = cand[0]
MOTIF = [A, B, A]   # a->b->a back-loop (transitions a->b and b->a)
print(f"common activities: {len(common)} | chosen rare transition: {A}  ->  {B}  "
      f"(consecutive in normal: {bigram.get((A, B), 0)} times)")
print(f"motif (structural loop): {MOTIF}")

deltas_pool = []
for case in log:
    ts = pd.to_datetime([e.timestamp for e in case.events], utc=True, errors='coerce')
    d = ts.to_series().diff().dt.total_seconds().dropna().values
    deltas_pool.extend(d[d > 0].tolist())
deltas_pool = np.array(deltas_pool)

def inject(case, cut):
    """Inject the structural loop (consecutive a->b->a) with probability escalating toward the transaction."""
    prefix, suffix = list(case.events[:cut]), case.events[cut:]
    P = max(len(prefix) - 1, 1)
    new_prefix = []
    for i, ev in enumerate(prefix):
        new_prefix.append(ev)
        if np.random.uniform(0, 1) < ESC_MAX * (i / P):
            new_prefix.extend(Event(name=m, timestamp=None) for m in MOTIF)
    case.events = new_prefix + suffix

np.random.seed(SEED)
case_label, cut_after = [], []
for case in log:
    names = [e.name for e in case.events]
    cut = names.index(SURROGATE) if SURROGATE in names else None
    if cut is not None and cut >= MIN_PREFIX and np.random.uniform(0, 1) <= FRAUD_RATE:
        inject(case, cut); case_label.append('fraud')
    elif cut is not None and cut >= MIN_PREFIX and np.random.uniform(0, 1) <= CONTAM_FRAC:
        inject(case, cut); case_label.append('normal')
    else:
        case_label.append('normal')
    nm = [e.name for e in case.events]
    cut_after.append(nm.index(SURROGATE) if SURROGATE in nm else np.nan)

rows = []
for case, lab in zip(log, case_label):
    n = len(case.events)
    if n == 0: continue
    d = np.random.choice(deltas_pool, size=max(n - 1, 0))
    ts = EPOCH + pd.to_timedelta(np.concatenate([[0.0], np.cumsum(d)]), unit='s')
    for e, t in zip(case.events, ts):
        rows.append((str(case.id), e.name, t, lab, 'signature' if lab == 'fraud' else 'normal'))

df = pd.DataFrame(rows, columns=['case_id', 'activity', 'timestamp', 'label', 'anomaly_type'])
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False, engine='pyarrow', compression='snappy')

per = df.groupby('case_id').agg(label=('label', 'first'))
nf = (per.label == 'fraud').sum()
ln = df.groupby('case_id').size()
fr_idx = set(per[per.label == 'fraud'].index)
fr_cut = [c for sid, c in zip(per.index, cut_after) if sid in fr_idx and not np.isnan(c)]
print(f'traces={len(per)} | fraud={nf} ({nf/len(per)*100:.2f}%) | events={len(df)}')
print('fraud transaction position (post-injection) pct:',
      {p: int(np.percentile(fr_cut, p)) for p in (25, 50, 75)})
print('eligible (>=L):', {L: int((ln >= L).sum()) for L in (30, 40, 50)})
