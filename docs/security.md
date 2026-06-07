# Security & deployment

This is a self-hosted local tool. It needs a physical SDR, so you run
your own instance and view it in your own browser.

## Network exposure

The web UI has **no authentication**. Anyone who can reach the port
can start and stop the receiver, change settings, read decoded
traffic, and read keys. For that reason the shipped `lora.toml` binds
to `127.0.0.1` by default and the UI prints a warning if you change
that.

> **Foot-gun**: if `lora.toml` is missing or `[web].host` is unset,
> `src/lora_config.py:22` codes the fallback as `0.0.0.0` — the UI
> would bind to every interface. Keep the file present and the host
> line set.

If you need LAN access, set `[web] host = "0.0.0.0"` in `lora.toml` —
but only on a trusted network, and ideally behind a reverse proxy
that adds auth and TLS. **Never expose it directly to the public
internet.**

## Privileges

Run as a normal user in the `plugdev` group. **Not as root.**

## Legal & ethical

You're responsible for the legal and ethical side of passive RF
reception wherever you live. Don't publish traffic, node identities,
or keys you don't have the right to share.

## Production hosting

The bundled Werkzeug server is fine for local single-user use. Reach
for a real WSGI server (e.g. gunicorn behind nginx) only if you're
deliberately hosting for others.
