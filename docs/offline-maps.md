# Online and offline maps

Open **Map → Map settings → Map source** to choose:

- **Auto** (default): uses online OpenStreetMap tiles when they load. If tile
  requests fail or stall for eight seconds, it switches to the installed offline
  pack, or the world overview if no pack is loaded. It retries online maps every
  30 seconds while the map is visible, and when the browser reports reconnection.
- **Online**: uses internet tiles and reports when they are unavailable. Choose
  Auto if you want automatic fallback to your saved regional map.
- **Offline**: uses only maps on the receiver, with no internet tile requests.

Your choice is saved in this browser on this device. Changing map source keeps
your view and received positions. Auto checks actual tile loading because being
connected to Wi-Fi does not guarantee internet access.

The receiver serves the map library, world overview and installed regional tiles
locally. A phone can open Map for the first time without internet access, as long
as it can reach the receiver over local Wi-Fi, Ethernet or an available Tailscale
connection. Tailscale connectivity itself may require internet; a direct local
network is the option for a fully disconnected field setup.

The included Natural Earth overview shows countries/coastlines. Received node
positions, tracks, waypoints, coordinates and position-uncertainty circles work
offline immediately. The overview **does not include streets or trails**.

## Add detailed coverage before going offline

1. Obtain a **raster MBTiles** file for the area you will use, from a source that
   permits offline use. PNG, JPEG and WebP tiles are supported. Vector/PBF MBTiles
   files are not supported by this renderer.
2. Open **Map → Map settings → Offline maps**, choose the `.mbtiles` file, and select
   **Load onto receiver**. Uploads are limited to 1 GiB; choose a smaller area or
   fewer zoom levels if needed.
3. Wait for **Map ready**, select **Offline**, then use **Show map area** to check
   coverage. Return to **Auto** if you want internet maps whenever available.
4. Verify the needed area and zoom levels before disconnecting from the internet.

One pack is stored in `lora_web/maps/offline.mbtiles`, alongside other local
receiver state. It survives restarts and is shared by all connected phones.
Installing another valid pack replaces it atomically; a rejected or interrupted
upload leaves the previous map intact. Large uploads also need free space for the
new temporary file alongside the existing map. Advanced users can copy a valid
raster MBTiles file to that path while the server is stopped.

The map reports when no tiles are saved for the current area/zoom. Zooming past a
pack's highest detail enlarges saved tiles; it cannot create new street detail.
Disable **Use regional map** to return to the world overview. Pack attribution is
shown on the map. Auto and Online request only the online tiles needed for the
current view and use normal browser caching. Offline packs must be supplied
separately; the app does not bulk-download public OpenStreetMap tiles. See the
[OpenStreetMap tile usage policy](https://operations.osmfoundation.org/policies/tiles/).

## Developer checks

Run `python3 -m pytest -q` for map storage and tile-serving tests.
With a local Flask preview running, run
`node --test tests/offline_map.cjs tests/map_source.cjs tests/mobile_ui.cjs` with
`PUPPETEER_MODULE` pointing to a development install of `puppeteer-core` and
`CHROME_BIN` pointing to Chromium. `UI_URL` defaults to `http://127.0.0.1:5057/`.
The offline browser test selects Offline and blocks all external network requests
from a fresh browser; it does not need a warmed cache. Map-source tests intercept
the internet tile service to simulate success, failures and stalled connections,
and check fallback, recovery, manual selection and persistence.
