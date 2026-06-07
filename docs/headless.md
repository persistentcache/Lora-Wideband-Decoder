# Headless mode

If you don't want the web UI, pipe the SDR directly into the detector.

## bladeRF at 28 Msps wideband

`buffers=512 samples=32768` match what the web-UI launcher uses in
production (`src/sdr_profiles.py`) — without them the bladeRF runs with
default buffering and can drop samples under load.

```bash
bladeRF-cli -e "set frequency rx 915000000; set samplerate rx 28000000;
    set bandwidth rx 28000000; set agc rx on;
    rx config file=/dev/stdout format=bin n=0 buffers=512 samples=32768
        xfers=64; rx start; rx wait" \
| python3 run/headless.py -r 28000000 -b 28000000 -c 915.0 -t sc16 \
    --decode --export-iq captures/
```

## USRP B210 / B205mini via SoapyUHD (~30 Msps capable)

```bash
python3 src/soapy_rx.py --driver uhd -f 915000000 -s 28000000 -b 28000000 \
| python3 run/headless.py -r 28000000 -b 28000000 -c 915.0 -t sc16 --decode
```

## HackRF (20 Msps cap) or other SoapySDR devices

```bash
python3 src/soapy_rx.py --driver hackrf -f 915000000 -s 20000000 -b 20000000 \
| python3 run/headless.py -r 20000000 -b 20000000 -c 915.0 -t sc16 --decode
```

Substitute `--driver` with `lime`, `plutosdr`, `airspy`, `rtlsdr`, …
as needed.

## Replay a recorded file

```bash
python3 run/headless.py -f recording.sc16 \
    -r 28000000 -b 28000000 -c 915.0 -t sc16 --decode
```
