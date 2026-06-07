# Security & deployment

The web UI has no authentication. Anyone who can reach the port can
read decoded traffic, read your keys, change settings, and stop the
receiver. So the shipped `lora.toml` binds it to `127.0.0.1`, and the
UI logs a warning if you change that.

One thing to watch: if `lora.toml` is missing or `[web].host` isn't
set, the fallback in `src/lora_config.py` is `0.0.0.0`, which binds to
every interface. Keep the file present and the host line set.

If you want LAN access, change `[web] host = "0.0.0.0"` in
`lora.toml`. Don't expose the port directly to the internet. If you
need remote access, put it behind something that adds auth and TLS
(reverse proxy, VPN, SSH tunnel).

Run as your normal user in the `plugdev` group. There's no reason to
run this as root.

Werkzeug's dev server is fine for one person on localhost. If you're
hosting for others, use a real WSGI server (gunicorn behind nginx, or
similar).

You're responsible for whether passive RF reception is legal where
you are. Don't republish traffic, node identities, or keys you don't
have the right to share.
