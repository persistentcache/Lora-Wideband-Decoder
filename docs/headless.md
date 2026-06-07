# Headless mode

Skip the web UI and pipe the SDR straight into the detector.

## bladeRF, 28 Msps

The `buffers=512 samples=32768 xfers=64` values aren't optional. Drop
them and the bladeRF will start dropping samples under load. The
web-UI launcher (`src/sdr_profiles.py`) uses the same numbers.

```bash
bladeRF-cli -e "set frequency rx 915000000; set samplerate rx 28000000;
    set bandwidth rx 28000000; set agc rx on;
    rx config file=/dev/stdout format=bin n=0 buffers=512 samples=32768
        xfers=64; rx start; rx wait" \
| python3 run/headless.py -r 28000000 -b 28000000 -c 915.0 -t sc16 \
    --decode --export-iq captures/
```

## USRP B210 / B205mini

Should handle 28 Msps over USB 3.0:

```bash
python3 src/soapy_rx.py --driver uhd -f 915000000 -s 28000000 -b 28000000 \
| python3 run/headless.py -r 28000000 -b 28000000 -c 915.0 -t sc16 --decode
```

## HackRF or any other SoapySDR device

HackRF tops out at 20 Msps. Adjust `-s` for whatever your device can
do.

```bash
python3 src/soapy_rx.py --driver hackrf -f 915000000 -s 20000000 -b 20000000 \
| python3 run/headless.py -r 20000000 -b 20000000 -c 915.0 -t sc16 --decode
```

Replace `hackrf` with `lime`, `plutosdr`, `airspy`, `rtlsdr`, etc.

## Replaying a recording

```bash
python3 run/headless.py -f recording.sc16 \
    -r 28000000 -b 28000000 -c 915.0 -t sc16 --decode
```
