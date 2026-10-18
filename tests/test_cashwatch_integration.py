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
    monkeypatch.delenv('CASHWATCH_CHECKPOINT', raising=False)
    assert analyzer_configured()
    monkeypatch.setenv('CASHWATCH_CHECKPOINT', '/missing-detector.pt')
    assert not analyzer_configured()
    monkeypatch.setenv('ANALYZER_CALLABLE', '')
    assert not analyzer_configured()


def test_default_checkpoint_prefers_localizer_then_refit_then_reviewed_then_binary(tmp_path, monkeypatch):
    import backend.cashwatch_adapter as adapter
    monkeypatch.delenv('CASHWATCH_CHECKPOINT', raising=False)
    monkeypatch.setattr(adapter, 'WEIGHTS_DIR', tmp_path)
    monkeypatch.setattr(adapter, 'DEFAULT_CHECKPOINT', tmp_path / 'event_localizer_detector.pt')
    for filename in ('detector.pt', 'reviewed_detector.pt', 'hard_negative_detector.pt', 'event_localizer_detector.pt'):
        checkpoint = tmp_path / filename
        checkpoint.write_bytes(b'Path selection fixture; never loaded as a model')
        assert adapter.checkpoint_path() == checkpoint
    override = tmp_path / 'explicit.pt'
    monkeypatch.setenv('CASHWATCH_CHECKPOINT', str(override))
    assert adapter.checkpoint_path() == override


def test_shipped_backend_and_python_defaults_use_event_localizer_model(monkeypatch):
    import hashlib
    import json
    import torch
    from backend.cashwatch_adapter import checkpoint_path
    from ml.cashwatch.inference import default_checkpoint_path
    monkeypatch.delenv('CASHWATCH_CHECKPOINT', raising=False)
    checkpoint = checkpoint_path()
    assert checkpoint.name == 'event_localizer_detector.pt'
    assert checkpoint == default_checkpoint_path()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['format_version'] == 4
    assert saved['config']['event_localization']['target'] == 'annotated_event_center'
    report = json.loads((checkpoint.parent / 'event_localizer_report.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == report['checkpoint_sha256']
    assert saved['config'] == json.loads((checkpoint.parent / 'event_localizer_config.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('filename,kind', [
    ('event_localizer_detector.pt', 'semantic'),
    ('hard_negative_detector.pt', 'semantic'),
    ('reviewed_detector.pt', 'semantic'),
    ('detector.pt', 'binary'),
])
def test_health_capability_follows_weights_not_filename(tmp_path, monkeypatch, filename, kind):
    import shutil
    from backend.cashwatch_adapter import analyzer_kind, WEIGHTS_DIR
    renamed = tmp_path / 'renamed.pt'
    shutil.copyfile(WEIGHTS_DIR / filename, renamed)
    monkeypatch.setenv('CASHWATCH_CHECKPOINT', str(renamed))
    assert analyzer_kind() == kind
    renamed.write_bytes(b'not the attested semantic checkpoint')
    assert analyzer_kind() == 'custom_checkpoint'


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
    monkeypatch.delenv('CASHWATCH_CHECKPOINT', raising=False)
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
            assert event.rule_source.status in {'provided', 'unmapped'}
            if event.type_id:
                assert 0 <= event.type_confidence <= 1
                assert event.description_source == 'class_template'
                assert event.rule_context['source_status']
            assert 1 <= len(event.evidence_frames) <= 3
