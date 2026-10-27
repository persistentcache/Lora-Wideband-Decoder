# Offline map plan

Goal: phone and receiver need no internet connection to render maps, positions, tracks or waypoints.

- Bundle Leaflet 1.9.4 and its license/assets; replace CDN references.
- Bundle a compact public-domain Natural Earth world overview. It is explicitly a coarse overview, not local street/trail coverage.
- Support one installed raster MBTiles map pack on the receiver, uploaded through Map → Offline map. Validate before atomic replacement; preserve the previous pack on error. Serve tiles locally, including proper TMS/XYZ conversion. No automatic external tile downloads or public tile scraping.
- Show readable offline map status, coverage/zoom limitations, pack attribution and loading errors; fit received positions or optionally fit the installed region.
- Test fresh browser with all external requests blocked, node/track/waypoint popups, map pack upload/serving/TMS/errors, phone layout, and existing regression suite.

Architecture: Flask blueprint in src/web/offline_maps.py, installed under lora_web/maps; Leaflet and Natural Earth static assets; browser integration in static/offline-map.js. No new Python or JS runtime dependencies. One-GiB upload cap; accept PNG/JPEG/WebP MBTiles, reject vector packs with an explanatory error.

Implementation complete. Review identified slow lookups in unindexed packs: uploads now gain a coordinate index in the temporary copy when needed, and validation checks lookups using the serving budget before installing a pack. Regressions cover both indexed flat tables and rejected slow views with 500,000 tiles. Python suite: 78 passed.

Final browser verification: 6 passed across mobile_ui.cjs and offline_map.cjs, including a fresh browser with external requests blocked, received positions and local node-location navigation, regional tiles, failed-upload preservation, missing coverage, and mobile/desktop regressions. Temporary Tailscale previews refreshed to the current implementation. No detailed regional pack is bundled or installed in the demo.
