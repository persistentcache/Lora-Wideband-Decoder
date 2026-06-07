# Headless mode

If you don't want the web UI, pipe the SDR directly into the detector.

## bladeRF at 28 Msps wideband

```bash
bladeRF-cli -e "set frequency rx 915000000; set samplerate rx 28000000;
    set bandwidth rx 28000000; set agc rx on;
    rx config file=/dev/stdout format=bin n=0; rx start; rx wait" \
| python3 run/headless.py -r 28000000 -b 28000000 -c 915.0 -t sc16 \
    --decode --export-iq captures/
```

## Any SoapySDR device (HackRF / RTL-SDR / LimeSDR / USRP / …)

```bash
python3 src/soapy_rx.py --driver hackrf -f 915000000 -s 20000000 -b 20000000 \
| python3 run/headless.py -r 20000000 -b 20000000 -c 915.0 -t sc16 --decode
```

## Replay a recorded file

```bash
python3 run/headless.py -f recording.sc16 \
    -r 28000000 -b 28000000 -c 915.0 -t sc16 --decode
```
