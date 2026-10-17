import base64
import io
import json
import os
from pathlib import Path
import time

from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
import pytest

from backend.jobs import JobStore, JobRunner, QueueFull
from backend.main import create_app
from backend.schemas import AnalysisResult, EvidenceFrame, Violation
from tests.helpers import clean_analyzer, counted_analyzer, slow_analyzer, wait_job

KEY = {"X-API-Key": "test-key"}


def make_app(tmp_path, **kwargs):
    return create_app(api_key="test-key", upload_dir=tmp_path / "uploads", auth_db=tmp_path / "auth.sqlite3", **kwargs)


def test_completed_result_and_owner_survive_restart(tmp_path, mov_bytes):
    headers = {"X-UI-Request": "1"}
    with TestClient(make_app(tmp_path, analyzer=clean_analyzer)) as client:
        client.post('/auth/api/register', headers=headers, json={
            'name': 'QA', 'email': 'persist@example.test', 'password': 'local-test-password-123',
        }).raise_for_status()
        token = client.cookies.get('ui_session')
        created = client.post('/app/api/v1/analyze', headers=headers, files={'file': ('q.mov', mov_bytes)})
        assert created.status_code == 202
        job_id = created.json()['job_id']
        before = wait_job(client, job_id, route='/app/api/v1/jobs/')
        assert before['status'] == 'done'
    with TestClient(make_app(tmp_path, analyzer=clean_analyzer)) as after:
        after.cookies.set('ui_session', token)
        assert after.get('/app/api/v1/jobs/' + job_id).json() == before
        after.cookies.clear()
        assert after.get('/app/api/v1/jobs/' + job_id).status_code == 401
        assert after.get('/api/v1/jobs/' + job_id, headers=KEY).json() == before


def test_timeout_terminates_process_and_removes_video(tmp_path, mov_bytes):
    with TestClient(make_app(tmp_path, analyzer=slow_analyzer, analysis_timeout=3, queue_capacity=1)) as client:
        response = client.post('/api/v1/analyze', headers=KEY, files={'file': ('q.mov', mov_bytes)})
        assert response.status_code == 202
        queued = client.post('/api/v1/analyze', headers=KEY, files={'file': ('q.mov', mov_bytes)})
        assert queued.status_code == 429
        assert queued.headers['Retry-After'] == '5'
        job = wait_job(client, response.json()['job_id'], KEY, timeout=8)
        assert job['status'] == 'failed'
        assert 'лимит' in job['error']
        assert 'verdict' not in job
        assert not list((tmp_path / 'uploads').glob('*.mov'))
        markers = list((tmp_path / 'uploads').glob('*.pid'))
        assert markers, "The blocking analyzer must actually have started"
        for marker in markers:
            with pytest.raises(ProcessLookupError):
                os.kill(int(marker.read_text()), 0)


def test_capacity_is_atomic_and_queue_is_durable(tmp_path, mov_bytes):
    store = JobStore(tmp_path / 'jobs.sqlite3')
    paths = [tmp_path / f'{i}.mov' for i in range(3)]
    for path in paths:
        path.write_bytes(mov_bytes)
    first = store.submit(paths[0], 'owner', 20, 2)
    second = store.submit(paths[1], 'owner', 20, 2)
    with pytest.raises(QueueFull):
        store.submit(paths[2], None, 20, 2)
    assert store.claim()['job_id'] == first
    # Simulate an interrupted running job and a queued job left in the same database.
    runner = JobRunner(JobStore(store.path), clean_analyzer, 1, 86400)
    runner.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and store.get(second)['state'] != 'done':
            time.sleep(.05)
        assert store.get(first)['state'] == 'failed'
        assert 'перезапуском' in store.get(first)['error']
        assert store.get(second)['state'] == 'done'
        assert store.get(second)['owner'] == 'owner'
        assert not paths[0].exists() and not paths[1].exists()
    finally:
        runner.close()


def test_two_api_workers_share_one_concurrency_limit(tmp_path, mov_bytes):
    store = JobStore(tmp_path / 'jobs.sqlite3')
    ids = []
    for i in range(4):
        path = tmp_path / f'{i}.mov'
        path.write_bytes(mov_bytes)
        ids.append(store.submit(path, None, 30, 8))
    runners = [JobRunner(JobStore(store.path), counted_analyzer, 1, 86400) for _ in range(2)]
    for runner in runners:
        runner.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and any(store.get(i)['state'] != 'done' for i in ids):
            time.sleep(.05)
        assert all(store.get(i)['state'] == 'done' for i in ids)
        assert json.loads((tmp_path / 'concurrency.json').read_text()) == {'active': 0, 'peak': 1}
    finally:
        for runner in runners:
            runner.close()


def test_retention_removes_old_results(tmp_path):
    store = JobStore(tmp_path / 'jobs.sqlite3')
    path = tmp_path / 'q.mov'
    path.write_bytes(b'fixture')
    job_id = store.submit(path, None, 20, 2)
    store.finish(job_id, result='{"verdict":"no_violation","violations":[]}')
    with store.connect() as db:
        db.execute('UPDATE jobs SET updated=? WHERE job_id=?', (time.time() - 10, job_id))
    store.maintain(1)
    assert store.get(job_id) is None


@pytest.mark.parametrize('route', ['register', 'login'])
def test_password_is_not_reflected_in_422(tmp_path, route):
    with TestClient(make_app(tmp_path, analyzer=clean_analyzer)) as client:
        secret = 'tiny-secret'
        response = client.post('/auth/api/' + route, headers={'X-UI-Request': '1'}, json={
            'name': 'QA', 'email': 'qa@example.test', 'password': secret,
        })
        assert response.status_code == 422
        assert secret not in response.text
        assert all(set(error) <= {'loc', 'type', 'msg'} for error in response.json()['detail'])


def test_jpeg_requires_decodable_pixels_and_source():
    fake = base64.b64encode(b'\xff\xd8not-jpeg\xff\xd9').decode()
    with pytest.raises(ValidationError):
        EvidenceFrame(timestamp_sec=1, image_base64=fake)
    image = io.BytesIO()
    Image.new('RGB', (32, 32)).save(image, format='JPEG')
    frame = {'timestamp_sec': 1, 'image_base64': base64.b64encode(image.getvalue()).decode()}
    assert EvidenceFrame(**frame).timestamp_sec == 1
    with pytest.raises(ValidationError):
        EvidenceFrame(timestamp_sec=1, image_base64=base64.b64encode(image.getvalue()[:-80]).decode())
    event = {'type': 'QA', 'timecode_sec': 1, 'description': 'QA', 'evidence_frames': [frame]}
    with pytest.raises(ValidationError):
        Violation(**event)
    with pytest.raises(ValidationError):
        Violation(**event, rule_source={'document': '  '})
    parsed = Violation(**event, rule_source={'document': 'Synthetic contract fixture'})
    assert AnalysisResult(verdict='violation', violations=[parsed])


def test_python_plugin_runs_in_worker(tmp_path, mov_bytes, monkeypatch):
    # A real module import and process call; this fixture is not a recognition model.
    monkeypatch.setenv('ANALYZER_CALLABLE', 'tests.helpers:clean_analyzer')
    with TestClient(make_app(tmp_path)) as client:
        assert client.get('/health').json()['analyzer_configured'] is True
        response = client.post('/api/v1/analyze', headers=KEY, files={'file': ('q.mov', mov_bytes)})
        assert response.status_code == 202
        assert wait_job(client, response.json()['job_id'], KEY)['status'] == 'done'


def test_swagger_has_only_local_assets(tmp_path):
    with TestClient(make_app(tmp_path, analyzer=clean_analyzer)) as client:
        html = client.get('/docs').text
        assert 'https://' not in html and 'http://' not in html
        for url in ['/docs-assets/swagger-ui-bundle.js', '/docs-assets/swagger-ui.css', '/assets/favicon.svg', '/openapi.json']:
            assert client.get(url).status_code == 200
        assert '"validatorUrl": null' in html
