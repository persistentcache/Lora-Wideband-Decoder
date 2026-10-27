from collections import deque

import pytest
from web import app as web


@pytest.mark.parametrize('query,count', [('', 2000), ('?full=1', 2003)])
def test_state_snapshot_retains_full_history_when_requested(monkeypatch, query, count):
    monkeypatch.setattr(web, 'PACKETS', deque({'pktid': i} for i in range(2003)))
    response = web.app.test_client().get('/api/state' + query)
    assert response.status_code == 200
    data = response.get_json()
    assert len(data['packets']) == count
    assert data['packets'][-1]['pktid'] == 2002
    assert data['stats']['packets'] == 2003


def test_state_reports_receiver_bandwidth_separately_from_sample_rate(monkeypatch):
    monkeypatch.setattr(web, '_effective_radio', lambda: (20000000, 12000000, 914.0))
    config = web.app.test_client().get('/api/state').get_json()['config']
    assert config['center_mhz'] == 914.0
    assert config['bandwidth_hz'] == 12000000
