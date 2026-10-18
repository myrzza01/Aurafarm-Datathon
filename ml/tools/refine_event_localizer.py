"""Refine point proximity with replay and mined off-center training examples.

Training/selection on the same annotated videos measures training fit only.
The cache contains frozen visual features; runtime never reads this cache,
filenames, labels, or annotated timestamps. A separate fresh-video audit is
required before replacing any deployed checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.cashwatch.data import file_hash
from ml.cashwatch.evaluation import detect_intervals
from ml.cashwatch.inference import VideoAnalyzer
from ml.cashwatch.train import set_seed
from ml.cashwatch.typed_evaluation import typed_event_metrics
from ml.cashwatch.video import clip_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--before-report', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu', 'auto'))
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error('epochs and learning rate must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'candidate.pt'
    if output.exists():
        raise FileExistsError('Keep previous candidates; use a new output directory')

    before = json.loads(args.before_report.read_text(encoding='utf-8-sig'))
    source_hash = file_hash(args.checkpoint)
    if before['checkpoint_sha256'] != source_hash:
        raise ValueError('Checkpoint and baseline report do not match')
    cache = torch.load(args.features, map_location='cpu', weights_only=True)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved['manifest_sha256'] != cache['manifest_sha256']:
        raise ValueError('Feature cache and checkpoint use different annotations')
    records = cache['records']
    recorded = {row['id']: row for row in before['videos']}
    if set(recorded) != {row['id'] for row in records}:
        raise ValueError('Training population differs from the regression report')
    for record in records:
        previous = recorded[record['id']]
        if record['sha256'] != previous['sha256']:
            raise ValueError('Video content changed')
        signature = lambda events: [(e['timecode_sec'], e['type_id']) for e in events]
        if signature(record['events']) != signature(previous['truth_events']):
            raise ValueError('Training annotations changed')

    set_seed(args.seed)
    analyzer = VideoAnalyzer(args.checkpoint, device=args.device)
    model = analyzer.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    localizer = model.localizer
    for parameter in localizer.parameters():
        parameter.requires_grad_(True)
    cfg = copy.deepcopy(analyzer.config)
    sigma = cfg['event_localization']['sigma_sec']
    gate = cfg['event_localization']['binary_gate_threshold']
    postprocess = cfg['event_localization']['postprocess']
    frozen = {k: v.clone() for k, v in saved['model'].items() if not k.startswith('localizer.')}
    old_predictions = {row['id']: row['after_predictions'] for row in before['videos']}
    old_metrics = typed_event_metrics(records, old_predictions, 1.0)
    old_matches = {(row['video_id'], e['truth_event_index']): e for row in old_metrics['videos']
                   for e in row['matches'] if e['category'] == 'type_correct'}

    inputs, targets, near_weights, far_weights = [], [], [], []
    timelines, slices, binary, actions = {}, {}, {}, {}
    offset = 0
    for record in records:
        identifier = record['id']
        item = cache['encoded'][identifier]
        times = item['times'].numpy()
        indices = np.stack([clip_indices(times, t, cfg['clip_frames'], cfg['fps']) for t in times])
        windows = item['features'][torch.as_tensor(indices)].to(analyzer.device)
        with torch.inference_mode():
            normalized = model.normalization(windows)
            difference = torch.zeros_like(normalized)
            difference[:, 1:] = normalized[:, 1:] - normalized[:, :-1]
            prepared = torch.cat([normalized, difference], dim=-1).transpose(1, 2).cpu()
        target = np.zeros(len(times), dtype=np.float32)
        for event in record['events']:
            target = np.maximum(target, np.exp(-0.5 * ((times - event['timecode_sec']) / sigma) ** 2))
        target = target.astype(np.float32)
        near, far = target >= 0.5, target < 0.5
        inputs.append(prepared)
        targets.append(torch.from_numpy(target))
        near_weights.extend(near.astype(float) / max(int(near.sum()), 1))
        far_weights.extend(far.astype(float) / max(int(far.sum()), 1))
        timelines[identifier] = times
        slices[identifier] = slice(offset, offset + len(times))
        offset += len(times)
        binary[identifier] = item['binary_logits'].sigmoid().numpy()
        actions[identifier] = item['action_logits'].softmax(-1).numpy()
    X, Y = torch.cat(inputs).to(analyzer.device), torch.cat(targets).to(analyzer.device)
    pw, nw = torch.tensor(near_weights, dtype=torch.double), torch.tensor(far_weights, dtype=torch.double)

    def evaluate():
        model.eval()
        with torch.inference_mode():
            scores = torch.cat([localizer(X[i:i + 64]).squeeze(-1).sigmoid().cpu()
                                for i in range(0, len(X), 64)]).numpy()
        predictions = {}
        for record in records:
            identifier = record['id']
            times = timelines[identifier]
            values = np.where(binary[identifier] >= gate, scores[slices[identifier]], 0.0)
            events = detect_intervals(times, values, **postprocess)
            for event in events:
                position = int(np.abs(times - event['timecode_sec']).argmin())
                distribution = actions[identifier][position]
                event['type_id'] = analyzer.type_ids[int(distribution.argmax())]
            predictions[identifier] = events
        metrics = typed_event_metrics(records, predictions, 1.0)
        matches = {(row['video_id'], e['truth_event_index']): e for row in metrics['videos']
                   for e in row['matches'] if e['category'] == 'type_correct'}
        gates = {
            'all_previous_typed_events_preserved': old_matches.keys() <= matches.keys(),
            'no_previous_timing_error_increased': all(key in matches and matches[key]['error_sec'] <= old['error_sec'] + 1e-8
                                                     for key, old in old_matches.items()),
            'all_previous_binary_verdicts_preserved': all(bool(predictions[r['id']]) == bool(recorded[r['id']]['after_verdict'])
                                                         for r in records),
        }
        return scores, metrics, predictions, gates

    scores, baseline_metrics, _, _ = evaluate()
    if baseline_metrics['typed'] != old_metrics['typed']:
        raise ValueError('Cached feature predictions do not reproduce the current baseline')
    optimizer = torch.optim.AdamW(localizer.parameters(), lr=args.learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    history, selected = [], False
    for epoch in range(1, args.epochs + 1):
        # Replay all videos; mine off-center examples by model output, never by filename.
        hard = nw * torch.from_numpy(((Y.cpu().numpy() < 0.1) & (scores >= 0.5)).astype(float))
        hard = hard if hard.sum() > 0 else nw
        positive = torch.multinomial(pw, 240, replacement=True, generator=generator)
        background = torch.multinomial(nw, 240, replacement=True, generator=generator)
        mined = torch.multinomial(hard, 160, replacement=True, generator=generator)
        localizer.train()
        loss_sum = 0.0
        for step in range(20):
            indices = torch.cat([positive[step * 12:(step + 1) * 12],
                                 background[step * 12:(step + 1) * 12],
                                 mined[step * 8:(step + 1) * 8]]).to(analyzer.device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(localizer(X[indices]).squeeze(-1), Y[indices])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(localizer.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) / 20
        scores, metrics, predictions, gates = evaluate()
        entry = {'epoch': epoch, 'loss': loss_sum, 'typed': metrics['typed'], 'gates': gates}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if all(gates.values()) and metrics['typed']['fp'] == 0 and metrics['typed']['fn'] == 0:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if any(not torch.equal(value, state[k]) for k, value in frozen.items()):
                raise ValueError('An original detector tensor changed')
            candidate = copy.deepcopy(saved)
            total_epochs = saved['epoch'] + epoch
            cfg.update(epochs=total_epochs, stage='frozen_detector_with_refined_point_localizer')
            candidate.update(model=state, config=cfg, epoch=total_epochs,
                             training_steps=saved['training_steps'] + epoch * 20,
                             previous_event_refit=saved['event_refit'],
                             event_refit={'source_sha256': source_hash, 'trained_modules': ['localizer'],
                                          'epoch': epoch, 'total_localizer_epochs': total_epochs,
                                          'updates': epoch * 20, 'seed': args.seed, 'learning_rate': args.learning_rate,
                                          'sampling': '12 near, 12 background, 8 mined off-center per batch; 20 batches per epoch',
                                          'point_labels_are_not_binary_normality_labels': True,
                                          'scope': 'all40_training_fit'},
                             selection='First epoch with zero unmatched events and no lost typed events, worse matched timing or changed binary verdicts on the training population.',
                             independent_quality_established=False,
                             weight_changes={k: float((state[k] - saved['model'][k]).abs().sum())
                                             for k in state if k.startswith('localizer.')})
            torch.save(candidate, output)
            (args.output_dir / 'training-report.json').write_text(json.dumps({
                'source_sha256': source_hash, 'checkpoint_sha256': file_hash(output),
                'training': candidate['event_refit'], 'metrics_1sec': metrics,
                'predictions': predictions, 'gates': gates,
                'independent_quality_established': False}, indent=2) + '\n', encoding='utf-8')
            selected = True
            break
    (args.output_dir / 'training-history.json').write_text(json.dumps(history, indent=2) + '\n', encoding='utf-8')
    if file_hash(args.checkpoint) != source_hash:
        raise ValueError('Source checkpoint changed during training')
    print(json.dumps({'selected': selected, 'epochs_run': len(history)}), flush=True)
    return 0 if selected else 2


if __name__ == '__main__':
    raise SystemExit(main())
