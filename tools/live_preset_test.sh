#!/bin/bash
# Live per-preset validation: set preset, stream 28 MHz live, send, measure
# 100% + send->decoded latency.  Usage:
#   live_preset_test.sh PRESET NMSGS WAIT TRAIL DEDUP_KEEP [--set]
set -u
HOME_DIR="/home/david/Stuff/LORA Project/lora_ml"; cd "$HOME_DIR"
PRESET="$1"; NMSGS="${2:-20}"; WAIT="${3:-0}"; TRAIL="${4:-30}"; KEEP="${5:-1}"
DOSET="${6:-}"
PORTS=(/dev/ttyUSB0 /dev/ttyUSB1)
declare -A ENUM=( [LONG_FAST]=0 [LONG_SLOW]=1 [VERY_LONG_SLOW]=2 [MEDIUM_SLOW]=3 \
  [MEDIUM_FAST]=4 [SHORT_SLOW]=5 [SHORT_FAST]=6 [LONG_MODERATE]=7 [SHORT_TURBO]=8 [LONG_TURBO]=9 )
WANT=${ENUM[$PRESET]}
CAPS=/dev/shm/live_caps; LOG=/tmp/live_${PRESET}.log; SENDLOG=/tmp/live_${PRESET}_send.log; ERR=/tmp/live_${PRESET}.err

if [ "$DOSET" = "--set" ]; then
  for p in "${PORTS[@]}"; do echo "[set] $PRESET on $p"; timeout 120 meshtastic --port $p --set lora.modem_preset $PRESET >/dev/null 2>&1; done
fi
# verify-wait: both nodes must report WANT before recording.
# SKIP_VERIFY=1 bypasses (use when the preset was already confirmed out-of-band
# — the meshtastic --get CLI is slow/flaky ~60s/call and can time out the loop
# even when the nodes are correctly set).
if [ "${SKIP_VERIFY:-0}" = "1" ]; then
  echo "[verify] SKIPPED (SKIP_VERIFY=1) — assuming $PRESET (enum $WANT) is set"
else
  deadline=$(( $(date +%s) + 240 )); conf=0
  while [ $(date +%s) -lt $deadline ] && [ $conf -lt 2 ]; do
    conf=0
    for p in "${PORTS[@]}"; do
      cur=$(timeout 60 meshtastic --port $p --get lora 2>/dev/null | grep -oE 'modem_preset: [0-9]+' | grep -oE '[0-9]+')
      [ "$cur" = "$WANT" ] && conf=$((conf+1))
    done
    [ $conf -lt 2 ] && sleep 8
  done
  [ $conf -lt 2 ] && { echo "[ERR] preset $PRESET not confirmed"; exit 1; }
  echo "[verify] $PRESET (enum $WANT) confirmed on both nodes"
fi

rm -rf "$CAPS"; mkdir -p "$CAPS"; rm -f "$LOG" "$SENDLOG" "$ERR"
export LORA_DECODE_BUDGET_S=${LORA_BUDGET:-10.0} LORA_DEDUP_KEEP=$KEEP
# LORA_DECODE_WORKERS unset → auto-scale.  Override via `LORA_DECODE_WORKERS=N` env if needed.
bladeRF-cli -e "set frequency rx 915000000; set samplerate rx 28000000; set bandwidth rx 28000000; set agc rx on; rx config file=/dev/stdout format=bin n=0 buffers=512 samples=32768 xfers=64; rx start; rx wait" 2>"$ERR" \
  | python3 scripts/lora_detect.py -r 28000000 -b 28000000 -c 915.0 -t sc16 \
      --threshold 0.55 --overlap 0.5 --energy-threshold 5.0 \
      --detect-workers "${LORA_DW:--1}" --buf-seconds 16 --decode --export-iq "$CAPS" -d 1 > "$LOG" 2>&1 &
echo "[live] pipeline up (keep=$KEEP wait=$WAIT trail=$TRAIL)"; sleep 18
echo "[live] sending $NMSGS msgs"
python3 tools/sendtest.py -i "$NMSGS" -w "$WAIT" -y > "$SENDLOG" 2>&1
echo "[send] sent=$(grep -c 'Node A sent' "$SENDLOG") recv=$(grep -c 'Node B received' "$SENDLOG")"
sleep "$TRAIL"
pkill -f "bladeRF-cli" 2>/dev/null
# Let the gate hit EOF and DRAIN its decode backlog before force-killing, so
# captures aren't killed mid-decode.  The gate exits on its own once drained;
# wait up to ~160 s for that, then force-kill.
for _i in $(seq 1 40); do pgrep -f "lora_detect.py -r 28000000" >/dev/null || break; sleep 4; done
pkill -f "lora_detect.py -r 28000000" 2>/dev/null; sleep 2
echo "=== $PRESET RESULT ==="
echo "$(grep 'Done:' "$LOG")"
echo "dropped: $(grep -oE 'skipped=[0-9.]+M' "$LOG" || echo NONE)  saved/decoded: $(grep -c 'Saved SF' "$LOG")  capSF: $(ls "$CAPS"/*.cf32 2>/dev/null | grep -oE 'SF[0-9]+_[0-9]+k' | sort -u | tr '\n' ' ')"
python3 tools/lat_analyze.py "$LOG" "$SENDLOG" "$NMSGS"
echo "=== $PRESET DONE ==="
