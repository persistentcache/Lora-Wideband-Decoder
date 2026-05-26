#!/usr/bin/env python3
# Compute send->decoded latency per preset from live log [LAT] markers + send log.
# Usage: lat_analyze.py <live_log> <send_log> <nmsgs>
import sys, re, datetime
live_log, send_log, N = sys.argv[1], sys.argv[2], int(sys.argv[3])

# midnight epoch today (for HH:MM:SS -> epoch conversion of the send log)
now = datetime.datetime.now()
midnight = datetime.datetime(now.year, now.month, now.day).timestamp()

def hhmmss_to_epoch(s):
    h, m, sec = s.split(':')
    return midnight + int(h)*3600 + int(m)*60 + int(sec)

# send log: pktid -> (send_epoch, msg)
sent = {}
for ln in open(send_log, errors='ignore'):
    m = re.search(r'\[(\d\d:\d\d:\d\d)\] Node A sent: "(Test\d+)"\s+\(id=(0x[0-9a-f]+)\)', ln)
    if m:
        sent[m.group(3)] = (hhmmss_to_epoch(m.group(1)), m.group(2))

# live log [LAT] epoch pktid hopN -> earliest emit per (pktid,hop)
emit = {}
for ln in open(live_log, errors='ignore'):
    m = re.search(r'\[LAT\] ([0-9.]+) (0x[0-9a-f]+) hop(\d)', ln)
    if m:
        key = (m.group(2), int(m.group(3)))
        t = float(m.group(1))
        if key not in emit or t < emit[key]:
            emit[key] = t

# hop0 latency = emit(hop0) - send (direct A->gate path)
lats = []
got_msgs = set()
for pid, (st, msg) in sent.items():
    got_msgs_h0 = (pid, 0) in emit
    if got_msgs_h0:
        lats.append(emit[(pid, 0)] - st)
# tally hops decoded (by pktid+hop across all emits, matched to a sent msg)
hops = 0
both = 0
for pid, (st, msg) in sent.items():
    h0 = (pid, 0) in emit
    h1 = (pid, 1) in emit
    hops += (1 if h0 else 0) + (1 if h1 else 0)
    if h0 and h1:
        both += 1
    if h0 or h1:
        got_msgs.add(msg)

lats.sort()
def pct(p):
    return lats[min(len(lats)-1, int(len(lats)*p))] if lats else float('nan')
print(f"TALLY: messages {len(got_msgs)}/{N}  both-hop {both}/{N}  hops {hops}/{2*N}")
if lats:
    print(f"HOP0 send->decoded latency (n={len(lats)}): "
          f"median={pct(0.5):.2f}s  p90={pct(0.9):.2f}s  min={lats[0]:.2f}s  max={lats[-1]:.2f}s")
else:
    print("HOP0 latency: no matched packets")
