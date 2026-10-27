# Mobile receiver workspace

User intent: phone users operating a receiver on a Pi should see results and basic status first, with full settings still available. Design judgment remains delegated from the UI redesign.

Architecture: keep the same Flask API, event stream and desktop UI. At widths up to 900px, render a bounded intercept card feed instead of the wide table. A compact receiver summary reports connection, running state, receive frequency, totals and last reception. Intercepts, Nodes, Map and More form the bottom navigation; Settings is inside More. Sessions sits beside Start/Stop receiver; More holds secondary views, settings, clear-session and theme controls. Filters and packet details use progressive disclosure. No new runtime dependency or extra server polling.

- [x] Add browser regression coverage for phone feed, status, disclosure, navigation, reconnect and desktop preservation; observe baseline failure.
- [x] Add mobile markup, presenter and responsive CSS. Preserve filtering, merge semantics and full packet details. Limit initial cards to 60; reveal more on demand. Hold display while reading and offer an explicit refresh, without pausing capture.
- [x] Integrate existing stats/health/stream updates; don't report a stale connection as live. Keep all settings usable, with scoped horizontal scrolling for dense tables.
- [x] Check 360/390/900px phones, desktop, light/dark, long text, encrypted/unknown traffic, empty results and session/setting actions. Run Python tests and review.

Validation command (existing local Flask preview): `PUPPETEER_MODULE=/tmp/lora-ui-browser/node_modules/puppeteer-core CHROME_BIN=/home/david/.cache/puppeteer/chrome/linux-149.0.7827.22/chrome-linux64/chrome node --test tests/mobile_ui.cjs`. Set UI_URL to use another Flask server. Browser uses synthetic packets and intercepts mutating requests, never starts hardware.

Implementation notes: full state snapshots are opt-in through `/api/state?full=1`, used for reconnect/session recovery; default startup stays bounded to 2,000 packets. Snapshot generations prevent an old response undoing a session change, stream-status generations prevent a late response claiming connectivity, and a stats revision prevents stale receiver state overriding newer stream stats. Arrivals during snapshot fetch are reconciled. Passive recovery preserves open settings and the mobile reading position.

Verification: 33 Python tests passed, plus four browser acceptance scenarios covering 360/390/760px portrait, 844px landscape, desktop, both themes, mixed/long/escaped traffic, lazy details, filtering, earlier intercepts, settings save requests, session controls, decryption menu, stream loss/recovery, delayed response races, explicit session replacement and Clear. Browser mutation requests were intercepted; no hardware was started. Independent review findings fixed and covered. `git diff --check` and JavaScript syntax checks passed. Actual Pi performance and physical-phone browser testing remain unmeasured.

Follow-up: show receiver bandwidth beneath center frequency, supplied by the state snapshot and live spectrum updates. Intercept subtitles use protocol and measured frequency (retaining sub-MHz precision), with channel name available in details. Browser coverage checks bandwidth updates and the revised navigation.

Navigation follow-up: Nodes now contains Node list / Network graph buttons at the top on phone and desktop. The separate Network navigation entry is removed; existing #net links still open the graph with Nodes selected. Reopening Nodes retains the chosen view. Sessions remains accessible beside receiver controls down to 320px. Browser checks cover both views, selected navigation, remembered selection, deep links and session access.

Final follow-ups: Sessions uses pale green for enabled auto-save and pale red when disabled, with accessible state labels. Empty intercept views now refresh on receiver start/stop events without waiting for the first packet; the browser regression covers phone/desktop and preserves the no-matching-results message when filtering. Final validation: 33 Python tests and all four browser scenarios passed.
