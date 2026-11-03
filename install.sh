#!/usr/bin/env bash
# ============================================================================
# LORA Wideband Decoder installer — Debian/Ubuntu.
#
# Installs everything the receiver needs to run on a fresh machine:
#   * SoapySDR + python bindings + per-device modules  (universal SDR support)
#   * native bladeRF / HackRF / RTL-SDR / Airspy tools (optional)
#   * the Python requirements (requirements.txt)
#   * SDR device access (plugdev group)
#
# Usage:   ./install.sh [--install-service] [--uninstall-service]
# (You'll be prompted for sudo to install system packages.)
#
#   --install-service   (aliases: --installservice, --install-autorun,
#                        --installautorun)  Also install + enable a systemd
#                        unit so the web UI AND the SDR pipeline start by
#                        themselves at boot, with the settings saved in the
#                        UI (lora_web/web_settings.json: SDR, radio, ...).
#   --uninstall-service Disable and remove that unit; nothing else changes.
#   --print-service     Print the unit that --install-service would write.
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

SERVICE_NAME="lora-wideband-decoder"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
INSTALL_SERVICE=0
for arg in "$@"; do
    case "$arg" in
        --install-service|--installservice|--install-autorun|--installautorun)
            INSTALL_SERVICE=1 ;;
        --uninstall-service|--uninstallservice)
            INSTALL_SERVICE=-1 ;;
        --print-service)
            INSTALL_SERVICE=2 ;;
        -h|--help)
            sed -n '3,/^# =====/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)
            echo "install.sh: unknown option '$arg' (try --help)" >&2; exit 2 ;;
    esac
done

# The unit runs as the installing user (pip --user puts the Python deps in
# that HOME; the plugdev membership gives SDR access) from this checkout,
# under the system python that the SoapySDR bindings are built against.
# --autostart makes the server launch the pipeline with the saved settings
# and keep retrying until it is healthy (USB SDRs often enumerate after the
# service starts).  systemd restarts the server itself if it ever dies, and
# stopping the unit stops the whole process tree (the SDR is released).
# ExecStart arguments are quoted because a checkout path may contain spaces;
# WorkingDirectory= takes the whole value as one path.
render_service() {
    local user="${SUDO_USER:-${USER:-$(whoami)}}"
    local home; home="$(getent passwd "$user" | cut -d: -f6)"
    cat <<UNIT
[Unit]
Description=LORA Wideband Decoder (web UI + SDR pipeline, autostart)
After=network-online.target systemd-udev-settle.service
Wants=network-online.target

[Service]
Type=simple
User=${user}
WorkingDirectory=${HERE}
Environment=HOME=${home}
Environment=LORA_AUTOSTART=1
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 "${HERE}/run/web.py" --autostart
Restart=always
RestartSec=5
TimeoutStopSec=25
KillMode=control-group

[Install]
WantedBy=multi-user.target
UNIT
}

install_service() {
    if ! command -v systemctl >/dev/null 2>&1; then
        echo "==> --install-service: systemd not found on this host; skipping the unit." >&2
        return 1
    fi
    echo "==> Installing systemd service ${SERVICE_NAME} ..."
    render_service | sudo tee "$SERVICE_PATH" >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME" >/dev/null
    sudo systemctl restart "$SERVICE_NAME"
    sleep 2
    sudo systemctl --no-pager --lines=5 status "$SERVICE_NAME" || true
    cat <<MSG

Service installed: the web UI and the configured SDR pipeline now start at boot.
  status:   systemctl status ${SERVICE_NAME}
  logs:     journalctl -u ${SERVICE_NAME} -f
  stop:     sudo systemctl stop ${SERVICE_NAME}      (releases the SDR)
  disable:  ./install.sh --uninstall-service
Settings changed in the web UI (SDR, radio, protocols) are saved to
${HERE}/lora_web/web_settings.json and used on the next boot.
MSG
}

uninstall_service() {
    if [ ! -f "$SERVICE_PATH" ]; then
        echo "==> ${SERVICE_NAME} is not installed."; return 0
    fi
    echo "==> Removing systemd service ${SERVICE_NAME} ..."
    sudo systemctl disable --now "$SERVICE_NAME" >/dev/null 2>&1 || true
    sudo rm -f "$SERVICE_PATH"
    sudo systemctl daemon-reload
    echo "   removed; start the receiver by hand with: python3 run/web.py"
}

case "$INSTALL_SERVICE" in
    2)  render_service; exit 0 ;;
    -1) uninstall_service; exit 0 ;;
esac

echo "==> LORA Wideband Decoder installer"

# If the user chose to run this installer from inside a venv, check
# whether the venv was created with --system-site-packages.  The apt
# python3-soapysdr package ships SoapySDR as a C extension bound to
# /usr/bin/python3 — it is NOT pip-installable into a venv.  Without
# the flag, the project won't be able to import SoapySDR at runtime
# even though apt reports it installed.  pyvenv.cfg is the canonical
# source of truth (line reads `include-system-site-packages = true|false`).
# Not-in-a-venv is the COMMON path — the installer's default
# `pip install --user` works fine there; no warning needed.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    _cfg="$VIRTUAL_ENV/pyvenv.cfg"
    if [ -f "$_cfg" ] && grep -q "^include-system-site-packages[[:space:]]*=[[:space:]]*true" "$_cfg"; then
        echo "==> venv detected with --system-site-packages (good)."
    else
        cat <<EOF >&2

==> WARNING: You're in a venv that lacks --system-site-packages:
       VIRTUAL_ENV=$VIRTUAL_ENV

    The apt python3-soapysdr package is a C extension bound to the
    system python and is NOT pip-installable.  Without
    --system-site-packages, the project won't be able to import
    SoapySDR at runtime even though apt reports it installed.

    Recommended fix BEFORE continuing:
        deactivate
        rm -rf "\$VIRTUAL_ENV"
        python3 -m venv --system-site-packages "\$VIRTUAL_ENV"
        source "\$VIRTUAL_ENV/bin/activate"

    Then re-run this installer.  Continuing in 10 seconds (Ctrl-C to abort)…

EOF
        sleep 10
    fi
fi

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
sudo apt-get install -y python3 python3-pip python3-dev build-essential libusb-1.0-0 \
    libfftw3-dev soapysdr-tools python3-soapysdr

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
# 'universe' is an UBUNTU component (not present on Debian / Raspberry Pi OS,
# where SDR packages live in main).  Only try to enable it on Ubuntu-family
# distros — guarding by /etc/os-release prevents the script from calling
# `add-apt-repository universe` on Debian, where the tool isn't installed by
# default and the component doesn't exist anyway (set -e otherwise bails the
# whole install on a non-issue).  Tested: Pi 4 / Raspberry Pi OS bookworm/trixie
# now finish the install; behaviour on Ubuntu 22.04 unchanged.
_is_ubuntu_family=false
if [ -r /etc/os-release ] && grep -qE '^ID(_LIKE)?=.*\bubuntu\b' /etc/os-release; then
    _is_ubuntu_family=true
fi
if $_is_ubuntu_family && ! grep -qhE '(^|[[:space:]])universe([[:space:]]|$)' \
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
sudo usermod -aG plugdev "${USER:-$(whoami)}" 2>/dev/null || true   # device packages add udev rules

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
try:
    import pyfftw
    fftw_msg = "pyfftw %s" % pyfftw.__version__
except ImportError:
    fftw_msg = "pyfftw MISSING (decoder will fall back to scipy.fft, slower)"
print("   python deps OK — numpy %s, scipy %s, flask %s, %s" %
      (numpy.__version__, scipy.__version__, flask.__version__, fftw_msg))
PY

cat <<EOF

Done.
  Run the receiver:   python3 run/web.py
  Open the web UI, go to Config → SDR/Radio, pick your SDR, click Detect.

If your SDR isn't found, re-plug it or log out/in once (so the 'plugdev' group
membership takes effect).
EOF

if [ "$INSTALL_SERVICE" = 1 ]; then
    install_service
else
    echo "  Start at boot:      ./install.sh --install-service   (web UI + SDR pipeline, saved settings)"
fi
