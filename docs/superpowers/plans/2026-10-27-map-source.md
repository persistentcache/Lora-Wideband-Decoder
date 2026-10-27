# Map Source Implementation Plan

**Goal:** Default to internet maps when reachable, automatically use local maps without internet, and offer a persistent manual choice.

**Design:** Keep Leaflet, the world overview, and raster MBTiles local. Add Auto (default), Online, and Offline to Map settings, saved per browser. Auto determines availability from actual visible tile loads, falls back on errors or an eight-second loading timeout, and retries every 30 seconds while the map is visible or on a browser connectivity event. Offline never requests external map tiles. Online explicitly uses internet tiles and reports unavailability. Preserve positions, zoom, and pan while switching. Existing pack controls remain in Map settings.

**Architecture:** Extend `src/web/static/offline-map.js`, keeping the installed-pack API unchanged. Use the standard HTTPS OpenStreetMap tile endpoint with visible attribution, browser caching, and no prefetch/download feature. Add the source select to `src/web/templates/index.html` and a small responsive style in `workbench.css`. Update `docs/offline-maps.md`.

**Execution:** Continue inline in the existing UI worktree under the user's delegated judgment. Leave changes unpushed as with the preceding offline-map work.

## Review focus

- Wi-Fi without internet: rely on actual image failure/timeout, not just navigator.onLine.
- Internet returns without a connectivity event: periodic retry while map is visible.
- Mode changes while images are loading: stale events must not override the choice.
- Local pack missing or outside coverage: retain overview and honest status.
- Reload/storage unavailable: persist valid modes when possible, default safely otherwise.

## Task: automatic map source and saved setting

- [x] Add browser tests for online preference, automatic fallback/recovery, timeout, manual mode persistence and no external requests in Offline. Pin prior offline tests to explicit Offline mode.
- [x] Run new tests and observe failure before implementation.
- [x] Implement source selection, layer transitions, bounded loading, reconnect retry and readable status without changing receiver settings.
- [x] Run all map/mobile browser tests and the Python suite; inspect phone layout and review the change.
- [x] Update offline-map usage documentation and refresh the temporary preview.

Validation: 12 browser scenarios passed, 78 Python tests passed, JavaScript syntax and whitespace checks passed. iPhone-sized preview inspected with no horizontal overflow. The initial online-preference test failed against the offline-only implementation. Subsequent test fixture corrections placed the viewport inside its regional pack, including after reload.

Independent review found no material defects. Supplemental browser checks confirmed denied localStorage access still allows Auto/manual Offline, and failed internet tiles outside installed coverage show the overview honestly. These two supplemental checks are not permanent automated cases. Existing backend/upload work remained outside this incremental review; it was reviewed during the preceding task.
