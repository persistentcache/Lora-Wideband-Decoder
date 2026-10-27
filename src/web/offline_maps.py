"""Local raster MBTiles installation and serving, without network access.

Packs are immutable after installation: uploads are validated in a temporary
file, then atomically replace the installed file. Installed databases are
always read-only; temporary upload copies may gain a tile coordinate index.
Every query has a time/instruction budget.
"""
from contextlib import contextmanager
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time

from flask import Blueprint, Response, jsonify, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge


MAX_UPLOAD_BYTES = 1024 ** 3
MAX_TILE_BYTES = 16 * 1024 ** 2
MAX_ZOOM = 30
_MAX_TILES = 2_000_000


class InvalidMap(ValueError):
    """The supplied database cannot be used as a raster map."""


@contextmanager
def _database(path, *, validation=False, writable=False):
    # as_uri quotes spaces, # and ? in paths; mode=ro cannot create a file.
    mode = '?mode=rw' if writable else '?mode=ro&immutable=1'
    db = sqlite3.connect(path.resolve().as_uri() + mode,
                         uri=True, timeout=0.5)
    deadline = time.monotonic() + (30 if validation else 1)
    remaining = 200_000 if validation else 2_000

    def exhausted():
        nonlocal remaining
        remaining -= 1
        return remaining <= 0 or time.monotonic() > deadline

    try:
        db.set_progress_handler(exhausted, 1000)
        if not writable:
            db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA trusted_schema=OFF')
        # Limit even computed results from an untrusted view, before Python
        # materializes them. Older Python versions do not expose setlimit.
        if hasattr(db, 'setlimit'):
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_TILE_BYTES + 65536)
        yield db
    finally:
        db.close()


def _raster_mime(prefix):
    """Recognize raster headers; image decoding is left to the browser."""
    if not isinstance(prefix, bytes):
        return None
    if prefix.startswith(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR') and len(prefix) >= 24:
        return 'image/png'
    if prefix.startswith(b'\xff\xd8\xff') and len(prefix) >= 4:
        return 'image/jpeg'
    if (len(prefix) >= 20 and prefix[:4] == b'RIFF' and
            prefix[8:12] == b'WEBP' and prefix[12:16] in (b'VP8 ', b'VP8L', b'VP8X')):
        return 'image/webp'
    return None


def _bounds(value):
    try:
        bounds = [float(part) for part in value.split(',')]
        if (len(bounds) == 4 and all(math.isfinite(v) for v in bounds) and
                -180 <= bounds[0] < bounds[2] <= 180 and
                -90 <= bounds[1] < bounds[3] <= 90):
            return bounds
    except (ValueError, AttributeError):
        pass
    return None


def _file_key(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _revision(key):
    return '-'.join(format(part, 'x') for part in key)


def _prepare_tile_index(path):
    """Index flat tables in the temporary copy, never in installed packs."""
    with _database(path, validation=True, writable=True) as db:
        kind = db.execute("SELECT type FROM sqlite_master WHERE name = 'tiles' "
                          "COLLATE NOCASE LIMIT 1").fetchone()
        if kind != ('table',):
            return  # A normalized tiles view must already support fast lookups.
        indexes = db.execute("SELECT name FROM pragma_index_list('tiles') "
                             'WHERE partial = 0 LIMIT 256').fetchall()
        for (name,) in indexes:
            columns = db.execute('SELECT name FROM pragma_index_info(?) '
                                 'ORDER BY seqno LIMIT 3', (name,)).fetchall()
            if {column for (column,) in columns} == {'zoom_level', 'tile_column', 'tile_row'}:
                return
        # A private, generated identifier cannot collide with source schema
        # names or interpolate untrusted metadata into SQL.
        index_name = '_lora_tile_coordinates_' + os.urandom(8).hex()
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=FULL')
        db.execute(f'CREATE INDEX {index_name} ON tiles '
                   '(zoom_level, tile_column, tile_row)')
        db.commit()


def _tile_data(path, z, x, tms_y):
    """Use the same bounded lookup during validation and HTTP serving."""
    with _database(path) as db:
        row = db.execute(
            'SELECT tile_data FROM tiles WHERE zoom_level = ? '
            'AND tile_column = ? AND tile_row = ? '
            'AND length(tile_data) <= ? LIMIT 1',
            (z, x, tms_y, MAX_TILE_BYTES)).fetchone()
    return row[0] if row and isinstance(row[0], bytes) else None


def _inspect(path, *, prepare=False):
    """Validate all tile headers in bounded memory, including tiles views."""
    with _database(path, validation=True) as db:
        if db.execute('PRAGMA quick_check(1)').fetchone() != ('ok',):
            raise InvalidMap('The MBTiles database is damaged.')
        rows = db.execute(
            'SELECT substr(name, 1, 64), substr(value, 1, 4096) '
            'FROM metadata LIMIT 256')
        metadata = {name: value for name, value in rows
                    if isinstance(name, str) and isinstance(value, str)}
        format_ = metadata.get('format', '').strip().lower()
        if format_ in ('pbf', 'mvt', 'vector'):
            raise InvalidMap('Vector MBTiles are not supported. Choose a raster '
                             'MBTiles pack containing PNG, JPEG or WebP tiles.')
        if format_ and format_ not in ('png', 'jpg', 'jpeg', 'webp', 'mixed'):
            raise InvalidMap('Unsupported map format. Choose PNG, JPEG or WebP MBTiles.')
        minzoom, maxzoom, count = MAX_ZOOM, 0, 0
        first_tile, last_per_zoom = None, {}
        tiles = db.execute(
            'SELECT zoom_level, tile_column, tile_row, length(tile_data), '
            'substr(tile_data, 1, 32) FROM tiles LIMIT ?', (_MAX_TILES + 1,))
        for z, x, y, size, prefix in tiles:
            count += 1
            if count > _MAX_TILES:
                raise InvalidMap('The map contains too many tiles; choose a smaller region.')
            if (not all(isinstance(v, int) for v in (z, x, y)) or
                    not 0 <= z <= MAX_ZOOM or not 0 <= x < 1 << z or
                    not 0 <= y < 1 << z):
                raise InvalidMap('The map contains invalid tile coordinates.')
            if (not isinstance(size, int) or not 0 < size <= MAX_TILE_BYTES or
                    _raster_mime(prefix) is None):
                raise InvalidMap('The map contains unsupported or damaged tiles. '
                                 'Choose raster PNG, JPEG or WebP MBTiles, not vector tiles.')
            minzoom, maxzoom = min(minzoom, z), max(maxzoom, z)
            if first_tile is None:
                first_tile = (z, x, y)
            last_per_zoom[z] = (z, x, y)
        if not count:
            raise InvalidMap('The MBTiles pack contains no tiles.')
    if prepare:
        _prepare_tile_index(path)
    # Scanning all headers does not establish that individual tiles can be
    # found within the tighter serving budget. Probe the first tile and the
    # last scanned tile at every zoom, including normalized tiles views.
    for coordinates in {first_tile, *last_per_zoom.values()}:
        try:
            if _tile_data(path, *coordinates) is None:
                raise InvalidMap('A tile in this map could not be retrieved.')
        except sqlite3.Error as exc:
            raise InvalidMap('Tile lookups in this map exceed the serving limit. '
                             'Export a pack with an index on zoom_level, tile_column '
                             'and tile_row, or choose a smaller region.') from exc
    # Actual tile zooms are authoritative; optional metadata may be stale or
    # malformed. Attribution stays plain text for textContent in the browser.
    return {
        'available': True,
        'name': metadata.get('name', '').strip()[:256] or 'Offline map',
        'minzoom': minzoom, 'maxzoom': maxzoom,
        'bounds': _bounds(metadata.get('bounds')),
        'attribution': metadata.get('attribution', ''),
    }


def _unavailable(error=None):
    info = {'available': False, 'name': '', 'minzoom': None, 'maxzoom': None,
            'bounds': None, 'attribution': '', 'revision': ''}
    if error:
        info['error'] = error
    return info


def create_blueprint(data_dir):
    """Create map routes with persistent storage rooted at ``data_dir``."""
    blueprint = Blueprint('offline_maps', __name__)
    maps_dir = Path(data_dir) / 'maps'
    installed = maps_dir / 'offline.mbtiles'
    lock = threading.RLock()
    cached_key, cached_info = None, None

    def info():
        nonlocal cached_key, cached_info
        with lock:
            try:
                key = _file_key(installed)
            except FileNotFoundError:
                return _unavailable()
            except OSError:
                return _unavailable('The installed map could not be read.')
            if key != cached_key:
                try:
                    cached_info = _inspect(installed)
                    cached_info['revision'] = _revision(key)
                except (sqlite3.Error, InvalidMap, OSError):
                    cached_info = _unavailable('The installed map is unreadable or invalid. '
                                               'Upload a raster MBTiles pack to replace it.')
                cached_key = key
            return cached_info

    @blueprint.get('/api/map')
    def map_info():
        return jsonify(info())

    @blueprint.post('/api/map/upload')
    def map_upload():
        nonlocal cached_key, cached_info
        temporary = None
        try:
            if request.content_length is not None and request.content_length > MAX_UPLOAD_BYTES:
                raise RequestEntityTooLarge()
            maps_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=maps_dir, prefix='.upload-',
                                             suffix='.mbtiles', delete=False) as target:
                temporary = Path(target.name)
                total = 0
                while True:
                    chunk = request.stream.read(min(65536, MAX_UPLOAD_BYTES - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise RequestEntityTooLarge()
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if not total:
                raise InvalidMap('The upload is empty. Choose a raster MBTiles file.')
            validated = _inspect(temporary, prepare=True)
            # Compute metadata before replacement: failures must leave the old
            # pack in place. Renaming preserves this inode, size and mtime.
            key = _file_key(temporary)
            validated['revision'] = _revision(key)
            with lock:
                os.replace(temporary, installed)
                temporary = None
                cached_key, cached_info = key, validated
            return jsonify(ok=True, map=validated)
        except RequestEntityTooLarge:
            return jsonify(ok=False, error='The map upload exceeds the 1 GiB limit.'), 413
        except InvalidMap as exc:
            return jsonify(ok=False, error=str(exc)), 400
        except sqlite3.Error:
            return jsonify(ok=False, error='This file is not a readable raster MBTiles '
                           'database, or it is too complex to validate. Choose a smaller '
                           'PNG, JPEG or WebP map pack.'), 400
        except BadRequest:
            return jsonify(ok=False, error='The map upload was interrupted or incomplete.'), 400
        except OSError:
            return jsonify(ok=False, error='The map could not be saved. Check available '
                           'disk space and permissions, then try again.'), 400
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass

    @blueprint.get('/api/map/tiles/<int:z>/<int:x>/<int:y>')
    def map_tile(z, x, y):
        if not 0 <= z <= MAX_ZOOM or not 0 <= x < 1 << z or not 0 <= y < 1 << z:
            return Response(status=404)
        try:
            data = _tile_data(installed, z, x, (1 << z) - 1 - y)
            mime = _raster_mime(data[:32]) if data is not None else None
            if mime:
                return Response(data, mimetype=mime,
                                headers={'X-Content-Type-Options': 'nosniff'})
        except (sqlite3.Error, OSError):
            pass
        return Response(status=404)

    @blueprint.after_request
    def local_map_headers(response):
        # Avoid stale tiles when a different pack replaces the same local URL.
        response.headers['Cache-Control'] = 'no-store'
        return response

    return blueprint
