#!/usr/bin/env bash
# ============================================================================
# lora_ml installer — Debian/Ubuntu.
#
# Installs everything the receiver needs to run on a fresh machine:
#   * SoapySDR + python bindings + per-device modules  (universal SDR support)
#   * native bladeRF / HackRF / RTL-SDR / Airspy tools (optional)
#   * the Python requirements (requirements.txt)
#   * SDR device access (plugdev group)
#
# Usage:   ./install.sh
# (You'll be prompted for sudo to install system packages.)
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> lora_ml installer"

if ! command -v apt-get >/dev/null 2>&1; then
    cat <<'EOF'
This installer targets Debian/Ubuntu (apt).  On other systems, install manually:
  - SoapySDR + its python3 bindings + your SDR's SoapySDR module
  - a C toolchain + libusb
  - then:  pip install -r requirements.txt
EOF
    exit 1
fi

echo "==> Installing system packages (sudo required) ..."
sudo apt-get update
# Essentials — abort if these fail.
sudo apt-get install -y python3 python3-pip build-essential libusb-1.0-0 \
    soapysdr-tools python3-soapysdr

# SDR device modules + native tools.  Any package not present in this distro's
# repo is skipped (not fatal) so the install still completes everywhere.
#   * soapysdr-module-all = EVERY Soapy device module in the repo (airspy, bladerf,
#     hackrf, rtlsdr, lms7/LimeSDR, uhd/USRP, osmosdr, mirisdr, rfspace, redpitaya,
#     remote, audio) — robust + future-proof vs. a hand-listed set.
#   * soapysdr-module-xtrx is NOT pulled by -all, so add it explicitly.
#   * native CLI tools (incl. airspyhf) for direct-capture profiles.
#   * libiio/libad9361 so a from-source SoapyPlutoSDR build works later.
# NOTE: SoapyPlutoSDR (ADALM-Pluto) and SoapyAirspyHF are NOT packaged in
# Debian/Ubuntu — for those, build the Soapy module from source.
echo "==> Installing SDR device support (skips anything not in your repo) ..."
for pkg in soapysdr-module-all soapysdr-module-xtrx \
           bladerf hackrf rtl-sdr airspy airspyhf \
           libiio-utils libad9361-0; do
    sudo apt-get install -y "$pkg" >/dev/null 2>&1 && echo "   + $pkg" \
        || echo "   - $pkg (not available — skipped)"
done

echo "==> Installing Python requirements ..."
# --user keeps it out of the system site; --break-system-packages satisfies PEP 668
# (it only touches ~/.local).  The system python3 — which the pipeline spawns —
# sees ~/.local, so the gate/decoder get these too.
pip3 install --user --break-system-packages -r "$HERE/requirements.txt"

echo "==> SDR device access ..."
sudo usermod -aG plugdev "$USER" 2>/dev/null || true   # device packages add udev rules

echo "==> Verifying ..."
echo "   SoapySDR device modules loaded:"
SoapySDRUtil --info 2>/dev/null | grep -oiE "lib[A-Za-z0-9]+Support\.so" | sort -u | sed 's/^/     - /' \
    || echo "     (none — check soapysdr install)"
echo "   Connected SDRs:"
SoapySDRUtil --find 2>/dev/null | grep -iE "driver|serial" | sed 's/^/   /' \
    || echo "     (no SDR detected yet — plug one in)"
python3 - <<'PY'
import numpy, scipy, flask
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
print("   python deps OK — numpy %s, scipy %s, flask %s" % (numpy.__version__, scipy.__version__, flask.__version__))
PY

cat <<EOF

Done.
  Run the receiver:   python3 lora_web/lora_web.py
  Open the web UI, go to Config → SDR/Radio, pick your SDR, click Detect.

If your SDR isn't found, re-plug it or log out/in once (so the 'plugdev' group
membership takes effect).
EOF
