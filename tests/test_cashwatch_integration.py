import io
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
import pytest

from backend.analyzer import analyzer_configured, DEFAULT_ANALYZER
from backend.cashwatch_adapter import analyze_video
from backend.main import create_app
from backend.schemas import AnalysisResult, UnmappedRuleSource, Violation
from tests.helpers import wait_job


def test_shipped_detector_is_available_without_configuration(monkeypatch):
    monkeypatch.delenv('ANALYZER_CALLABLE', raising=False)
    assert analyzer_configured()
    monkeypatch.setenv('CASHWATCH_CHECKPOINT', '/missing-detector.pt')
    assert not analyzer_configured()
    monkeypatch.setenv('ANALYZER_CALLABLE', '')
    assert not analyzer_configured()


def test_unmapped_rule_is_explicit_not_a_fabricated_document():
    import base64
    image = io.BytesIO()
    Image.new('RGB', (32, 32)).save(image, format='JPEG')
    event = {
        'type': 'Binary model', 'timecode_sec': 1,
        'description': 'Requires human review',
        'evidence_frames': [{'timestamp_sec': 1, 'image_base64': base64.b64encode(image.getvalue()).decode()}],
    }
    with patch('ml.cashwatch.inference.VideoAnalyzer') as model:
        model.return_value.analyze.return_value = ({'verdict': 'violation', 'violations': [event]}, {})
        result = AnalysisResult.model_validate(analyze_video(Path('synthetic.mov')))
    assert result.verdict == 'violation'
    rule = result.violations[0].rule_source
    assert isinstance(rule, UnmappedRuleSource)
    assert rule.document is None and rule.reason
    with pytest.raises(ValidationError):
        Violation(**{**event, 'rule_source': {'status': 'unmapped', 'reason': ' '}})
    with pytest.raises(ValidationError):
        Violation(**{**event, 'rule_source': {'status': 'unmapped', 'reason': 'Unknown', 'document': 'Invented'}})


def test_actual_checkpoint_completes_api_job(tmp_path, mov_bytes, monkeypatch):
    # Real model + weights on a synthetic video. This verifies integration, not accuracy.
    monkeypatch.setenv('ANALYZER_CALLABLE', DEFAULT_ANALYZER)
    monkeypatch.setenv('CASHWATCH_DEVICE', 'cpu')
    app = create_app(api_key='local-ml-test', upload_dir=tmp_path/'uploads', auth_db=tmp_path/'auth.sqlite3')
    headers = {'X-API-Key': 'local-ml-test'}
    with TestClient(app) as client:
        assert client.get('/health').json()['analyzer_configured'] is True
        response = client.post('/api/v1/analyze', headers=headers, files={'file': ('synthetic.mov', mov_bytes)})
        assert response.status_code == 202
        job = wait_job(client, response.json()['job_id'], headers, timeout=90)
        assert job['status'] == 'done', job.get('error')
        result = AnalysisResult.model_validate(job)
        assert result.verdict in {'violation', 'no_violation'}
        for event in result.violations:
            assert event.rule_source.status == 'unmapped'
            assert 1 <= len(event.evidence_frames) <= 3
