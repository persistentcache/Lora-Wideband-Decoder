#!/usr/bin/env python3
"""Per-preset end-to-end validation harness.

Usage: preset_test.py PRESET [--set] [--msgs N]
  PRESET : Meshtastic modem preset name (e.g. SHORT_FAST, LONG_SLOW)
  --set  : set the preset on both radios first (else assume already set)
  --msgs : number of sendtest messages (default 60)

Cycle: [set preset on both radios] -> record full 28 MHz while sendtest sends
N messages -> process the recording through lora_detect (detect pool + decode)
-> tally unique hops.  Prints a final RESULT line:
    RESULT <PRESET>: messages X/N both-hop Y/N hops Z/2N FP F
"""
import sys, subprocess, time, os, signal, re

PRESET = sys.argv[1]
DO_SET = '--set' in sys.argv
MSGS = 60
if '--msgs' in sys.argv:
    MSGS = int(sys.argv[sys.argv.index('--msgs') + 1])
# Inter-message send delay (sendtest -w) and post-send trailing record time.
# High SF (11/12) has multi-second airtime; at -w 0 the relay's hop1 rebroadcast
# queue backs up unbounded (CSMA defers on the busy channel), so late messages'
# hop1 never make it on-air before recording stops.  Spacing the sends lets the
# relay keep pace, and extra trailing time drains any residual queue.
WAIT = 0
if '--wait' in sys.argv:
    WAIT = int(sys.argv[sys.argv.index('--wait') + 1])
TRAIL = 15
if '--trail' in sys.argv:
    TRAIL = int(sys.argv[sys.argv.index('--trail') + 1])

# Project root, computed from this file's location (tools/ is one level below
# the root) so the harness is portable — no machine-specific absolute path.
HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.dirname(os.path.abspath(__file__))
RATE = 28_000_000
RAW = f"/tmp/preset_{PRESET}.sc16"
CAPS = f"/tmp/caps_{PRESET}"
DECLOG = f"/tmp/preset_{PRESET}_decode.log"
SENDLOG = f"/tmp/preset_{PRESET}_send.log"
PORTS = ['/dev/ttyUSB0', '/dev/ttyUSB1']


def sh(cmd, timeout=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout)


def log(*a):
    print(*a, flush=True)


# ---- 1. set + verify preset ----
# Meshtastic enum value for each preset name (modem_preset reads back as the
# integer).  After a --set the node REBOOTS (~22 s) and only then applies the
# new preset; a fixed sleep(25) is not enough — the verify came back empty
# (node still rebooting) and recording then captured STALE traffic on the old
# preset (e.g. a "LONG_TURBO" run recorded SF11/250 because the node hadn't
# switched to SF11/500 yet).  Poll --get lora until BOTH nodes report the
# expected enum before recording; bail if it never confirms.
PRESET_ENUM = {
    'LONG_FAST': 0, 'LONG_SLOW': 1, 'VERY_LONG_SLOW': 2, 'MEDIUM_SLOW': 3,
    'MEDIUM_FAST': 4, 'SHORT_SLOW': 5, 'SHORT_FAST': 6, 'LONG_MODERATE': 7,
    'SHORT_TURBO': 8, 'LONG_TURBO': 9,
}
_want = PRESET_ENUM.get(PRESET)
if _want is None:
    log(f"ERROR: unknown preset {PRESET}"); sys.exit(1)

if DO_SET:
    for p in PORTS:
        log(f"[set] {PRESET} on {p}")
        sh(f"meshtastic --port {p} --set lora.modem_preset {PRESET}", timeout=120)

def _preset_of(port):
    r = sh(f"meshtastic --port {port} --get lora", timeout=60)
    for l in r.stdout.splitlines():
        if 'modem_preset' in l:
            try:
                return int(l.split(':')[1].strip())
            except (ValueError, IndexError):
                return None
    return None

_deadline = time.time() + 180
_confirmed = set()
while time.time() < _deadline and len(_confirmed) < len(PORTS):
    for p in PORTS:
        if p in _confirmed:
            continue
        cur = _preset_of(p)
        if cur == _want:
            _confirmed.add(p)
            log(f"[verify] {p}: modem_preset={cur} == {PRESET} OK")
        else:
            log(f"[verify] {p}: modem_preset={cur} (want {_want}); waiting...")
    if len(_confirmed) < len(PORTS):
        time.sleep(8)
if len(_confirmed) < len(PORTS):
    log(f"ERROR: preset {PRESET}({_want}) not confirmed on all nodes "
        f"(confirmed {sorted(_confirmed)}) — aborting to avoid recording stale traffic")
    sys.exit(1)

# ---- 2. record (continuous) + sendtest ----
os.system(f"rm -f {RAW}; rm -rf {CAPS}; mkdir -p {CAPS}")
blade_cmd = (
    f'bladeRF-cli -e "set frequency rx 915000000; set samplerate rx {RATE}; '
    f'set bandwidth rx {RATE}; set agc rx on; '
    f'rx config file={RAW} format=bin n=0 buffers=512 samples=32768 xfers=64; '
    f'rx start; rx wait"')
blade = subprocess.Popen(blade_cmd, shell=True, preexec_fn=os.setsid,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
t0 = time.time()
while not (os.path.exists(RAW) and os.path.getsize(RAW) > 50_000_000):
    if time.time() - t0 > 30:
        log("ERROR: recording did not start"); sys.exit(1)
    time.sleep(1)
log(f"[rec] recording active (wait={WAIT}s trail={TRAIL}s)")
st = sh(f"cd '{HOME}' && python3 '{TOOLS}/sendtest.py' -i {MSGS} -w {WAIT} -y", timeout=9000)
open(SENDLOG, 'w').write(st.stdout + st.stderr)
sent = st.stdout.count("Node A sent")
recv = st.stdout.count("Node B received")
log(f"[send] sent {sent}/{MSGS}, Node B recv {recv}/{MSGS}")
log(f"[trail] draining relay queue for {TRAIL}s")
time.sleep(TRAIL)  # capture trailing relays (drain hop1 queue)
try:
    os.killpg(os.getpgid(blade.pid), signal.SIGTERM)
except Exception:
    pass
time.sleep(2)
sz = os.path.getsize(RAW)
log(f"[rec] raw {sz/1e9:.1f} GB ({sz/4/RATE:.0f}s)")

# ---- 3. process ----
log("[proc] running pipeline...")
env = dict(os.environ); env['LORA_DECODE_BUDGET_S'] = '40.0'
det = (f"python3 '{HOME}/scripts/lora_detect.py' -f {RAW} "
       f"-r {RATE} -b {RATE} -c 915.0 -t sc16 "
       f"--threshold 0.55 --dc-notch 0.1 --overlap 0.5 --energy-threshold 5.0 "
       f"--detect-workers 6 --decode --export-iq {CAPS} -d 1")
with open(DECLOG, 'w') as f:
    subprocess.run(det, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)

# ---- 4. tally ----
# Dedup-aware, keyed by UNIQUE (pktid, hops_taken).  The decode log holds the
# verbose subprocess output of EVERY decode of EVERY (redundant) capture, so a
# real packet appears many times and one of those can be a chase-garble (correct
# pktid+hop, wrong text — e.g. "VE1t49" for Test49 hop1).  Counting distinct
# Message strings reports that garble as an FP, but the live pipeline's
# (pkt_id, hops_taken) dedup emits only the first/correct decode and suppresses
# the garble, so it never reaches output.  Build (pktid,hop)->set(messages): a
# (pktid,hop) with ANY TestN is a real hop; one whose decodes are ALL non-Test
# is a true FP.  (Line-based pairing: track the most recent (pktid,hop) from a
# Flags line, bind to the next Message line — avoids the DOTALL mis-pairing that
# undercounted hops.)
ph = {}
_cur = None
for line in open(DECLOG, errors='ignore'):
    m = re.search(r'PacketID: (0x[0-9a-f]+)\s+Flags: 0x[0-9a-f]+ '
                  r'\(hop_limit=\d+ hop_start=\d+ hops_taken=(\d+)\)', line)
    if m:
        _cur = (m.group(1), int(m.group(2)))
        continue
    m = re.search(r'Message: "([^"]*)"', line)
    if m and _cur is not None:
        ph.setdefault(_cur, set()).add(m.group(1))
        _cur = None
mh = {}
fp = []
for (pid, hop), msgs in ph.items():
    tests = [x for x in msgs if re.fullmatch(r'Test\d+', x)]
    if tests:
        mh.setdefault(tests[0], set()).add(hop)
    else:
        fp.append((pid, hop, sorted(msgs)))
got = set(mh)
both = [m for m in mh if {0, 1} <= mh[m]]
hops = sum(len(v) for v in mh.values())
exp = set(f"Test{i}" for i in range(1, MSGS + 1))
inc = sorted(((m, sorted({0, 1} - mh[m])) for m in mh if mh[m] != {0, 1}),
             key=lambda x: int(x[0][4:]))
log(f"missing: {sorted(exp - got, key=lambda s: int(s[4:]))}")
log(f"incomplete: {inc}")
if fp:
    log(f"FP detail: {fp[:10]}")
log(f"RESULT {PRESET}: messages {len(got)}/{MSGS}  both-hop {len(both)}/{MSGS}  "
    f"hops {hops}/{2*MSGS}  FP {len(fp)}")
