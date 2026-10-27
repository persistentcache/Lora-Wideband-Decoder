"""Exercise the offline map HTTP API using real, temporary MBTiles files."""
import base64
import io
import sqlite3

import pytest
from flask import Flask

from web import offline_maps


PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/'
    'AAX+Av4N70a4AAAAAElFTkSuQmCC')
JPEG = base64.b64decode(
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof'
    'Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwh'
    'MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/'
    'wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAA'
    'AgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkK'
    'FhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWG'
    'h4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl'
    '5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA'
    'AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYk'
    'NOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOE'
    'hYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk'
    '5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q==')
WEBP = base64.b64decode('UklGRiQAAABXRUJQVlA4IBgAAAAwAQCdASoBAAEAAUAmJaQAA3AA/vz0AAA=')


def make_pack(tmp_path, metadata=None, tiles=None, *, normalized=False, indexed=True):
    """Tiles use MBTiles' bottom-origin TMS coordinates, not URL coordinates."""
    path = tmp_path / 'fixture.mbtiles'
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE metadata (name TEXT, value TEXT)')
        db.executemany('INSERT INTO metadata VALUES (?, ?)', (
            metadata if metadata is not None else {
                'name': 'Test region', 'format': 'png', 'minzoom': '2',
                'maxzoom': '2', 'bounds': '-75,40,-74,41',
                'attribution': 'Example map author',
            }).items())
        table = 'tile_storage' if normalized else 'tiles'
        db.execute(f'CREATE TABLE {table} (zoom_level INTEGER, tile_column INTEGER, '
                   'tile_row INTEGER, tile_data BLOB)')
        if indexed:
            db.execute(f'CREATE UNIQUE INDEX tile_index ON {table} '
                       '(zoom_level, tile_column, tile_row)')
        db.executemany(f'INSERT INTO {table} VALUES (?, ?, ?, ?)',
                       tiles if tiles is not None else [(2, 1, 2, PNG)])
        if normalized:
            db.execute('CREATE VIEW tiles AS SELECT * FROM tile_storage')
    return path.read_bytes()


def make_client(data_dir):
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(offline_maps.create_blueprint(data_dir))
    return app.test_client()


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path / 'receiver')


def upload(client, payload):
    return client.post('/api/map/upload', data=payload,
                       content_type='application/octet-stream')


def test_no_map_is_explicit_and_missing_tiles_do_not_create_database(client, tmp_path):
    state = client.get('/api/map').get_json()
    assert state['available'] is False
    assert client.get('/api/map/tiles/0/0/0').status_code == 404
    assert not (tmp_path / 'receiver/maps/offline.mbtiles').exists()


def test_upload_exposes_metadata_and_serves_xyz_via_tms(client, tmp_path):
    response = upload(client, make_pack(tmp_path))
    assert response.status_code == 200
    data = response.get_json()
    assert data['ok'] is True
    state = data['map']
    assert state == client.get('/api/map').get_json()
    assert state['available'] is True
    assert state['name'] == 'Test region'
    assert (state['minzoom'], state['maxzoom']) == (2, 2)
    assert state['bounds'] == [-75, 40, -74, 41]
    assert state['attribution'] == 'Example map author'
    assert state['revision']
    tile = client.get('/api/map/tiles/2/1/1')
    assert tile.status_code == 200
    assert tile.data == PNG
    assert tile.mimetype == 'image/png'
    assert tile.headers['X-Content-Type-Options'] == 'nosniff'
    assert client.get('/api/map/tiles/2/1/2').status_code == 404
    restarted = make_client(tmp_path / 'receiver')
    assert restarted.get('/api/map').get_json() == state
    assert restarted.get('/api/map/tiles/2/1/1').data == PNG


@pytest.mark.parametrize('data,format_,mime', [
    (PNG, 'png', 'image/png'), (JPEG, 'jpg', 'image/jpeg'),
    (WEBP, 'webp', 'image/webp'),
], ids=['png', 'jpeg', 'webp'])
def test_supported_raster_formats_and_tiles_view(client, tmp_path, data, format_, mime):
    response = upload(client, make_pack(tmp_path, {'format': format_},
                                       [(0, 0, 0, data)], normalized=True))
    assert response.status_code == 200
    tile = client.get('/api/map/tiles/0/0/0')
    assert tile.data == data
    assert tile.mimetype == mime


@pytest.mark.parametrize('metadata,tile_data', [
    ({'format': 'pbf'}, b'\x1a\x08vector'),
    ({'format': 'pbf'}, PNG),
    ({'format': 'png'}, b'<svg onload="alert(1)"></svg>'),
    ({'format': 'png'}, b'not an image'),
])
def test_vector_and_disguised_non_raster_packs_are_rejected(client, tmp_path, metadata, tile_data):
    response = upload(client, make_pack(tmp_path, metadata, [(0, 0, 0, tile_data)]))
    assert response.status_code == 400
    assert response.get_json()['ok'] is False
    if metadata['format'] == 'pbf':
        assert 'vector' in response.get_json()['error'].lower()
    assert client.get('/api/map').get_json()['available'] is False


@pytest.mark.parametrize('invalid', [b'', b'not sqlite', b'SQLite format 3\x00truncated'])
def test_invalid_upload_preserves_previous_pack(client, tmp_path, invalid):
    original = upload(client, make_pack(tmp_path)).get_json()['map']
    response = upload(client, invalid)
    assert response.status_code == 400
    assert response.get_json()['ok'] is False
    assert client.get('/api/map').get_json() == original
    assert client.get('/api/map/tiles/2/1/1').data == PNG
    assert [p.name for p in (tmp_path / 'receiver/maps').iterdir()] == ['offline.mbtiles']


@pytest.mark.parametrize('coords', ['31/0/0', '2/4/1', '2/1/4', '-1/0/0',
                                   '0/0/1', '0/1/0', '2/1/1%27', '../offline.mbtiles'])
def test_tile_coordinates_and_paths_are_bounded(client, tmp_path, coords):
    upload(client, make_pack(tmp_path))
    assert client.get('/api/map/tiles/' + coords).status_code == 404


@pytest.mark.parametrize('metadata', [
    {'minzoom': 'bad', 'maxzoom': '999', 'bounds': 'broken'},
    {'minzoom': '4', 'maxzoom': '1', 'bounds': 'nan,1,2,3'},
    {'minzoom': '-1', 'maxzoom': '2', 'bounds': '-181,0,1,91'},
    {'bounds': '10,20,-10,-20'}, {},
])
def test_malformed_optional_metadata_uses_actual_zoom_and_no_bounds(client, tmp_path, metadata):
    response = upload(client, make_pack(tmp_path, metadata))
    assert response.status_code == 200
    state = response.get_json()['map']
    assert (state['minzoom'], state['maxzoom']) == (2, 2)
    assert state['bounds'] is None
    assert isinstance(state['name'], str)
    assert isinstance(state['attribution'], str)


def test_declared_oversized_upload_is_rejected_without_replacing_map(client, tmp_path, monkeypatch):
    original = upload(client, make_pack(tmp_path)).get_json()['map']
    monkeypatch.setattr(offline_maps, 'MAX_UPLOAD_BYTES', 100)
    response = upload(client, b'x' * 101)
    assert response.status_code == 413
    assert response.get_json()['ok'] is False
    assert client.get('/api/map').get_json() == original


def test_unknown_length_upload_is_still_capped(client, tmp_path, monkeypatch):
    original = upload(client, make_pack(tmp_path)).get_json()['map']
    monkeypatch.setattr(offline_maps, 'MAX_UPLOAD_BYTES', 100)
    response = client.open('/api/map/upload', method='POST', environ_overrides={
        'wsgi.input': io.BytesIO(b'x' * 101), 'wsgi.input_terminated': True,
        'CONTENT_LENGTH': '', 'CONTENT_TYPE': 'application/octet-stream',
    })
    assert response.status_code == 413
    assert response.get_json()['ok'] is False
    assert client.get('/api/map').get_json() == original


def test_corrupt_installed_database_is_reported_without_server_error(client, tmp_path):
    upload(client, make_pack(tmp_path))
    installed = tmp_path / 'receiver/maps/offline.mbtiles'
    installed.write_bytes(b'corrupted')
    state = client.get('/api/map').get_json()
    assert state['available'] is False
    assert state['error']
    assert client.get('/api/map/tiles/2/1/1').status_code == 404


def test_installed_database_is_not_changed_by_read_requests(client, tmp_path):
    upload(client, make_pack(tmp_path))
    installed = tmp_path / 'receiver/maps/offline.mbtiles'
    before = installed.read_bytes()
    installed.chmod(0o444)
    restarted = make_client(tmp_path / 'receiver')
    assert restarted.get('/api/map').get_json()['available'] is True
    assert restarted.get('/api/map/tiles/2/1/1').data == PNG
    assert installed.read_bytes() == before
    assert [p.name for p in installed.parent.iterdir()] == ['offline.mbtiles']


def test_disk_failure_returns_json_and_keeps_previous_pack(client, tmp_path, monkeypatch):
    original = upload(client, make_pack(tmp_path)).get_json()['map']

    def disk_full(*args, **kwargs):
        raise OSError('No space left on device')

    monkeypatch.setattr(offline_maps.os, 'replace', disk_full)
    response = upload(client, make_pack(tmp_path))
    assert response.status_code == 400
    assert response.get_json()['ok'] is False
    assert client.get('/api/map').get_json() == original
    assert client.get('/api/map/tiles/2/1/1').data == PNG


@pytest.mark.parametrize('tiles', [[], [(31, 0, 0, PNG)], [(0, 1, 0, PNG)],
                                   [(0, 0, -1, PNG)], [(None, 0, 0, PNG)],
                                   [(0, 0, 0, None)]],
                         ids=['empty', 'zoom', 'column', 'row', 'null-zoom', 'null-data'])
def test_empty_packs_and_invalid_tile_records_are_rejected(client, tmp_path, tiles):
    response = upload(client, make_pack(tmp_path, tiles=tiles))
    assert response.status_code == 400
    assert response.get_json()['ok'] is False


def test_validation_checks_beyond_the_first_raster_tile(client, tmp_path):
    tiles = [(7, x, 0, PNG) for x in range(100)] + [(7, 100, 0, b'vector')]
    response = upload(client, make_pack(tmp_path, tiles=tiles))
    assert response.status_code == 400
    assert response.get_json()['ok'] is False


def test_schema_missing_required_tile_columns_is_rejected(client, tmp_path):
    pack = tmp_path / 'bad-schema.sqlite'
    with sqlite3.connect(pack) as db:
        db.execute('CREATE TABLE metadata(name TEXT, value TEXT)')
        db.execute('CREATE TABLE tiles(unrelated TEXT)')
    response = upload(client, pack.read_bytes())
    assert response.status_code == 400
    assert response.get_json()['ok'] is False


def test_oversized_tile_is_rejected_without_materializing_its_blob(client, tmp_path, monkeypatch):
    monkeypatch.setattr(offline_maps, 'MAX_TILE_BYTES', 32)
    response = upload(client, make_pack(tmp_path))
    assert response.status_code == 400
    assert response.get_json()['ok'] is False


def test_metadata_html_is_returned_as_bounded_json_text(client, tmp_path):
    metadata = {'name': '<img src=x onerror=alert(1)>' * 1000,
                'attribution': '<a href="https://example.com">Example</a>' * 1000}
    response = upload(client, make_pack(tmp_path, metadata))
    assert response.status_code == 200
    state = response.get_json()['map']
    assert len(state['name']) <= 256
    assert len(state['attribution']) <= 4096
    assert response.mimetype == 'application/json'
    assert state['name'].startswith('<img')


def test_reading_expensive_sqlite_view_has_a_query_budget(client, tmp_path):
    installed = tmp_path / 'receiver/maps/offline.mbtiles'
    installed.parent.mkdir(parents=True)
    with sqlite3.connect(installed) as db:
        # Without an execution budget this recursive aggregate never returns.
        db.execute('''CREATE VIEW tiles AS WITH RECURSIVE n(x) AS
                   (VALUES(1) UNION ALL SELECT x+1 FROM n)
                   SELECT 0 AS zoom_level, 0 AS tile_column, 0 AS tile_row,
                          sum(x) AS tile_data FROM n''')
    response = client.get('/api/map/tiles/0/0/0')
    assert response.status_code == 404


def test_map_directory_with_uri_reserved_characters_works(tmp_path):
    client = make_client(tmp_path / 'receiver #1?local')
    response = upload(client, make_pack(tmp_path))
    assert response.status_code == 200
    assert client.get('/api/map/tiles/2/1/1').data == PNG


def test_successful_replacement_changes_revision_and_tile_bytes(client, tmp_path):
    first = upload(client, make_pack(tmp_path)).get_json()['map']
    second = upload(client, make_pack(tmp_path, {'format': 'webp'},
                                    [(2, 1, 2, WEBP)])).get_json()['map']
    assert second['revision'] != first['revision']
    tile = client.get('/api/map/tiles/2/1/1')
    assert tile.data == WEBP
    assert tile.mimetype == 'image/webp'
    assert tile.headers['Cache-Control'] == 'no-store'


def test_large_unindexed_pack_serves_its_last_tile_and_preserves_original(client, tmp_path):
    payload = make_pack(tmp_path, {'format': 'png'},
                        ((10, i % 1024, i // 1024, PNG) for i in range(500_000)),
                        indexed=False)
    response = upload(client, payload)
    assert response.status_code == 200
    # The 500,000th TMS tile is (10, 287, 488), hence XYZ row 535.
    tile = client.get('/api/map/tiles/10/287/535')
    assert tile.status_code == 200
    assert tile.data == PNG
    assert (tmp_path / 'fixture.mbtiles').read_bytes() == payload
    restarted = make_client(tmp_path / 'receiver')
    assert restarted.get('/api/map').get_json()['available'] is True
    assert restarted.get('/api/map/tiles/10/287/535').data == PNG


def test_unindexed_large_tiles_view_is_rejected_before_replacing_working_map(client, tmp_path):
    original = upload(client, make_pack(tmp_path)).get_json()['map']
    payload = make_pack(tmp_path, {'format': 'png'},
                        ((10, i % 1024, i // 1024, PNG) for i in range(500_000)),
                        normalized=True, indexed=False)
    response = upload(client, payload)
    assert response.status_code == 400
    assert response.get_json()['ok'] is False
    assert 'index' in response.get_json()['error'].lower()
    assert client.get('/api/map').get_json() == original
    assert client.get('/api/map/tiles/2/1/1').data == PNG
    assert [p.name for p in (tmp_path / 'receiver/maps').iterdir()] == ['offline.mbtiles']
