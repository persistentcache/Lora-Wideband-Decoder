#!/usr/bin/env python3
"""
sendtest.py — Send numbered test messages between two Meshtastic USB-serial nodes.

Usage:
  python3 sendtest.py -i 3 -w 15
  python3 sendtest.py -i 5 -w 10 -p LONG_FAST

Options:
  -i N       Number of messages to send (Test1, Test2, ... TestN)  [default: 3]
  -w SECS    Seconds to wait between messages                       [default: 15]
  -p PRESET  LoRa preset to apply to both nodes before sending (optional)
  -y         Auto-close connections at end without prompting

Valid presets: SHORT_FAST, SHORT_SLOW, MEDIUM_FAST, MEDIUM_SLOW,
               LONG_FAST, LONG_MODERATE, LONG_SLOW, VERY_LONG_SLOW
"""

import argparse
import os
import sys
import time
import threading
from datetime import datetime

try:
    import meshtastic.serial_interface
    from meshtastic import BROADCAST_ADDR
    from pubsub import pub
except ImportError as exc:
    sys.exit(f"Missing dependency: {exc}\n  pip install meshtastic")


# ---------------------------------------------------------------------------
# Device serial ports
# ---------------------------------------------------------------------------

DEV1_PORT = "/dev/ttyUSB0"   # DEV1_783c — sender  (Node A)
DEV2_PORT = "/dev/ttyUSB1"   # DEV2_2c88 — receiver (Node B)


# ---------------------------------------------------------------------------
# LoRa preset table
# ---------------------------------------------------------------------------

PRESETS = {
    "SHORT_FAST":     0,
    "SHORT_SLOW":     1,
    "MEDIUM_FAST":    2,
    "MEDIUM_SLOW":    3,
    "LONG_FAST":      4,
    "LONG_MODERATE":  5,
    "LONG_SLOW":      6,
    "VERY_LONG_SLOW": 7,
}
PRESET_BY_VAL = {v: k for k, v in PRESETS.items()}

RECV_TIMEOUT = 30   # seconds to wait for Node B to receive the channel broadcast
ACK_TIMEOUT  = 15   # seconds to wait for the routing ACK after receive


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ts():
    return datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)

def node_id_str(num):
    return f"!{num:08x}"


# Primary channel on both devices (index 0, default encryption AQ==)
CHANNEL_INDEX = 0
CHANNEL_NAME  = "primary"


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------

def get_preset(iface):
    val = iface.localConfig.lora.modem_preset
    return val, PRESET_BY_VAL.get(val, f"#{val}")

def apply_preset_if_needed(iface, label, target_val):
    """Write preset to iface if it differs. Returns True if a write was needed."""
    cur_val, cur_name = get_preset(iface)
    target_name = PRESET_BY_VAL.get(target_val, f"#{target_val}")
    if cur_val == target_val:
        log(f"  {label}: already {cur_name} — no change needed")
        return False
    log(f"  {label}: {cur_name} -> {target_name}")
    iface.localConfig.lora.modem_preset = target_val
    iface.writeConfig("lora")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Send test messages between DEV1 (Node A) and DEV2 (Node B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-i", "--interval", type=int, default=3, metavar="N",
                    help="number of messages to send (default: 3)")
    ap.add_argument("-w", "--wait", type=float, default=15.0, metavar="SECS",
                    help="seconds to wait between messages (default: 15)")
    ap.add_argument("-p", "--preset", metavar="PRESET",
                    help="LoRa preset to apply to both nodes: " + ", ".join(PRESETS))
    ap.add_argument("-y", "--yes", action="store_true",
                    help="auto-close connections at the end without prompting")
    ap.add_argument("--dm", action="store_true",
                    help="send as DIRECT MESSAGES to Node B (PKI-encrypted) "
                         "instead of channel broadcasts. ROUTING_APP ACK detection "
                         "is unchanged — works for both modes (relay echo for broadcast, "
                         "B's ack for DM).")
    args = ap.parse_args()

    # Validate preset
    preset_val = None
    if args.preset:
        key = args.preset.upper().replace("-", "_")
        if key not in PRESETS:
            ap.error(f"Unknown preset '{args.preset}'. Valid: {', '.join(PRESETS)}")
        preset_val = PRESETS[key]

    # --- Connect ---
    log(f"Connecting to Node A (DEV1_783c) {DEV1_PORT} ...")
    try:
        iface1 = meshtastic.serial_interface.SerialInterface(DEV1_PORT)
    except Exception as exc:
        sys.exit(f"Node A connection failed: {exc}")

    log(f"Connecting to Node B (DEV2_2c88) {DEV2_PORT} ...")
    try:
        iface2 = meshtastic.serial_interface.SerialInterface(DEV2_PORT)
    except Exception as exc:
        try:
            iface1.close()
        except Exception:
            pass
        sys.exit(f"Node B connection failed: {exc}")

    node1_num = iface1.myInfo.my_node_num
    node2_num = iface2.myInfo.my_node_num
    log(f"Node A: {node_id_str(node1_num)}")
    log(f"Node B: {node_id_str(node2_num)}")

    log(f"Channel: [{CHANNEL_INDEX}] '{CHANNEL_NAME}'")
    print()

    # --- Apply LoRa preset if requested ---
    if preset_val is not None:
        log(f"Checking LoRa preset (target: {PRESET_BY_VAL[preset_val]})")
        ch1 = apply_preset_if_needed(iface1, "Node A", preset_val)
        ch2 = apply_preset_if_needed(iface2, "Node B", preset_val)
        if ch1 or ch2:
            log("Preset written — waiting 15s for node(s) to reboot ...")
            time.sleep(2)
            for changed, iface in [(ch1, iface1), (ch2, iface2)]:
                if changed:
                    try:
                        iface.close()
                    except Exception:
                        pass
            time.sleep(13)
            log("Reconnecting ...")
            try:
                if ch1:
                    iface1 = meshtastic.serial_interface.SerialInterface(DEV1_PORT)
                    node1_num = iface1.myInfo.my_node_num
                if ch2:
                    iface2 = meshtastic.serial_interface.SerialInterface(DEV2_PORT)
                    node2_num = iface2.myInfo.my_node_num
                log("Reconnected.")
            except Exception as exc:
                sys.exit(f"Reconnect after preset change failed: {exc}")
        print()

    # --- Pubsub event tracking ---
    # _recv_evts[pkt_id] — set when Node B receives the channel broadcast
    # _ack_evts[pkt_id]  — set when Node A receives the routing ACK from Node B
    _lock      = threading.Lock()
    _recv_evts = {}
    _ack_evts  = {}

    def _on_packet(packet, interface):
        decoded  = packet.get("decoded", {})
        portnum  = decoded.get("portnum", "")
        from_num = packet.get("from", 0)

        if portnum == "TEXT_MESSAGE_APP" and interface is iface2:
            if from_num == node1_num:
                text   = decoded.get("text", "")
                pkt_id = packet.get("id", 0)
                snr    = packet.get("rxSnr", "?")
                rssi   = packet.get("rxRssi", "?")
                log(f"Node B received: \"{text}\""
                    f"  (SNR={snr} dB, RSSI={rssi} dBm)")
                with _lock:
                    ev = _recv_evts.get(pkt_id)
                if ev:
                    ev.set()

        elif portnum == "TEXT_MESSAGE_APP" and interface is iface1:
            # Relay-ACK: Node B re-broadcast our packet with decremented hop_limit.
            # Node A receives its own packet back — this is how Meshtastic signals "Acknowledged".
            rx_id     = packet.get("id", 0)
            hop_limit = packet.get("hopLimit", "?")
            with _lock:
                ev = _ack_evts.get(rx_id)
            if ev:
                log(f"ACK (relay): Node B relayed \"{decoded.get('text','')}\""
                    f"  (id=0x{rx_id:08x} hop_limit={hop_limit})")
                ev.set()

        elif portnum == "ROUTING_APP" and interface is iface1:
            # Firmware sends ROUTING_APP from own node (err=NONE) once the relay echo
            # is received back — this is how Meshtastic signals delivery-ACK for broadcasts.
            req_id = decoded.get("requestId", 0)
            error  = decoded.get("routing", {}).get("errorReason", "NONE")
            ok     = error in ("NONE", "", 0, None)
            if ok and req_id:
                log(f"ACK: relay echo received for id=0x{req_id:08x}")
                with _lock:
                    ev = _ack_evts.get(req_id)
                if ev:
                    ev.set()
            else:
                log(f"[DBG] ROUTING_APP from=!{from_num:08x}"
                    f" req_id=0x{req_id:08x} err={error}")

    pub.subscribe(_on_packet, "meshtastic.receive")

    # --- Send loop ---
    exit_code = 0
    try:
        for i in range(1, args.interval + 1):
            text    = f"Test{i}"
            recv_ev = threading.Event()
            ack_ev  = threading.Event()

            log(f"--- Message {i}/{args.interval} ---")
            log(f"Requesting send: \"{text}\"  Node A -> {('DM to '+node_id_str(node2_num)) if args.dm else 'channel broadcast'}")

            pkt    = iface1.sendText(
                text,
                destinationId=(node2_num if args.dm else BROADCAST_ADDR),
                wantAck=True,
                channelIndex=CHANNEL_INDEX,
            )
            pkt_id = pkt.id if pkt and hasattr(pkt, "id") else 0
            log(f"Node A sent: \"{text}\"  (id=0x{pkt_id:08x})")

            if pkt_id:
                with _lock:
                    _recv_evts[pkt_id] = recv_ev
                    _ack_evts[pkt_id]  = ack_ev

                if not recv_ev.wait(RECV_TIMEOUT):
                    log(f"  ! No receive confirmation from Node B (waited {RECV_TIMEOUT}s)")
                    exit_code = 1
                if not ack_ev.wait(ACK_TIMEOUT):
                    log(f"  ! No relay-ACK from Node B (waited {ACK_TIMEOUT}s)")
                    exit_code = 1

                with _lock:
                    _recv_evts.pop(pkt_id, None)
                    _ack_evts.pop(pkt_id, None)
            else:
                log("  ! Could not read packet ID — skipping ACK wait")
                time.sleep(5)

            if i < args.interval:
                log(f"Waiting {args.wait:.0f}s ...")
                print()
                time.sleep(args.wait)

    except KeyboardInterrupt:
        print("\nAborted.", flush=True)
        exit_code = 130
    finally:
        try:
            pub.unsubscribe(_on_packet, "meshtastic.receive")
        except Exception:
            pass

    print()
    if args.yes:
        answer = "y"
    else:
        try:
            answer = input("Close connections? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "y"

    if answer == "n":
        log("Leaving connections open.")
    else:
        log("Closing connections ...")
        for iface in (iface1, iface2):
            try:
                iface.close()
            except Exception:
                pass

    log("Done.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
