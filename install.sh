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
# Most SDR tools live in the 'universe' component.  Make sure it's enabled —
# without it `apt-cache show bladerf` returns nothing and every device package
# below will skip.  Idempotent: if universe is already in sources.list this is
# a no-op.
if ! grep -qhE '(^|[[:space:]])universe([[:space:]]|$)' \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.list \
        /etc/apt/sources.list.d/*.sources 2>/dev/null; then
    echo "==> Enabling 'universe' apt component (needed for SDR packages) ..."
    sudo apt-get install -y software-properties-common >/dev/null 2>&1 || true
    sudo add-apt-repository -y universe 2>&1 | tail -2
    sudo apt-get update
fi

echo "==> Installing SDR device support (skips anything not in your repo) ..."
# Two-stage: (1) check apt cache so 'not in any source' is reported clearly;
# (2) install with apt-get, surfacing real errors instead of swallowing them.
# Without this, a missing 'universe' or a held package looks the same as "doesn't
# exist anywhere" — and we couldn't diagnose either.
for pkg in soapysdr-module-all soapysdr-module-uhd soapysdr-module-xtrx \
           bladerf hackrf rtl-sdr airspy airspyhf uhd-host \
           libiio-utils libad9361-0; do
    if ! apt-cache show "$pkg" >/dev/null 2>&1; then
        echo "   - $pkg  (not found in any apt source)"
        continue
    fi
    if out=$(sudo apt-get install -y "$pkg" 2>&1); then
        echo "   + $pkg"
    else
        err=$(echo "$out" | grep -E "^E:|Unable to locate|held|Depends:" | head -1)
        echo "   - $pkg  (install failed: ${err:-see apt output})"
    fi
done

# UHD (USRP B200/B205mini/B210) FPGA + firmware images.  uhd-host ships the
# downloader; the images themselves are pulled separately (~70 MB).  Without
# these, B205mini fails to enumerate on first connect.  Idempotent.
if command -v uhd_images_downloader >/dev/null 2>&1; then
    echo "==> Downloading UHD FPGA/firmware images (USRP devices) ..."
    sudo uhd_images_downloader 2>&1 | tail -4
fi

echo "==> Installing Python requirements ..."
# --user keeps it out of the system site; --break-system-packages is only used
# when this pip supports it (added in pip 23.0.1; PEP 668 enforcement applies
# from Debian 12 / Ubuntu 23.04 onwards).  Ubuntu 22.04 ships pip 22.0.2 which
# doesn't recognize the flag and doesn't need it — so probe first.  The system
# python3 — which the pipeline spawns — sees ~/.local, so the gate/decoder
# get these too.
PIP_FLAGS="--user"
if pip3 install --help 2>/dev/null | grep -q -- '--break-system-packages'; then
    PIP_FLAGS="$PIP_FLAGS --break-system-packages"
fi
pip3 install $PIP_FLAGS -r "$HERE/requirements.txt"

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
