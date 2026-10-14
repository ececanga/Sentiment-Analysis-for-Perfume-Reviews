"""Repeated training on immutable accepted splits; dry-run is the default.

Historical replay entrypoints remain seed-42 protocols. This orchestrator uses
those same validated inputs and training primitives, with explicit training
seed overrides and independent, content-validated completion records.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import Any

# Initialize PyTorch before loading XGBoost. On this macOS environment the
# reverse native-library load order segfaults during BERT embedding creation.
# Keep this at module startup; in-process tests often already imported torch.
import torch
import numpy as np
import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT
from src.training.model_benchmark_runner import (
    select_feature_columns, validate_configured_frozen_inputs, validate_source_group_audit,
)
from src.training.source_grouped_folds import (
    sha256_file, load_and_validate_outer_fold_manifest,
    build_grouped_inner_validation_manifest, validate_grouped_inner_manifest,
)
from src.training.run_source_grouped_feature_ablation import build_feature_families, build_feature_variants
from src.training.run_source_grouped_weighted_synthetic_lr import (
    AUGMENTED, CONTROL, build_training_rows, positive_class_probability, evaluate_oof,
)

DEFAULT_CONFIG = PROJECT_ROOT / 'config/thesis_three_seed_v1.yaml'
METRICS = ('macro_f1', 'macro_roc_auc', 'macro_average_precision')
CLASSICAL = ('logistic_regression', 'xgboost', 'random_forest')
ALIASES = {'logistic_regression': 'lr'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def project_path(value):
    return (PROJECT_ROOT / value).resolve()


@dataclass(frozen=True)
class Job:
    experiment: str
    model: str
    condition: str
    variant: str
    model_seed: int
    data_seed: int = 42

    @property
    def run_id(self):
        return f'{self.experiment}/{self.model}/{self.condition}/{self.variant}/data{self.data_seed}_model{self.model_seed}'


def weighted_config_path(model):
    return PROJECT_ROOT / f'config/evaluation_config_source_grouped_weighted_synthetic_{ALIASES.get(model, model)}_v1.yaml'


def model_module(model):
    return importlib.import_module(f'src.training.run_{model}_benchmark')


def weighted_module(model):
    return importlib.import_module(f'src.training.run_source_grouped_weighted_synthetic_{ALIASES.get(model, model)}')


def variants_for(model, features):
    families = build_feature_families(features, enforce_expected_counts=True)
    if model == 'logistic_regression':
        return build_feature_variants(features, families)
    if model in CLASSICAL:
        cfg = read_yaml(PROJECT_ROOT / f'config/evaluation_config_source_grouped_{model}_feature_combination_ablation_v1.yaml')
        return {name: [c for c in features if any(c in families[f] for f in spec['families'])]
                for name, spec in cfg['variants'].items()}
    return {}


def build_jobs(config, features, *, models=None, conditions=None, seeds=None, experiments=None, variants=None):
    jobs = []
    for experiment in experiments or config['experiments']:
        for model in models or config['models']:
            choices = {('full_hybrid' if model in CLASSICAL else 'standard'): features} if experiment == 'main' else variants_for(model, features)
            for condition in conditions or config['conditions']:
                if experiment == 'structured_ablation' and condition != CONTROL:
                    continue
                for variant in choices:
                    if variants and variant not in variants:
                        continue
                    for seed in seeds or config['model_seeds']:
                        jobs.append(Job(experiment, model, condition, variant, seed, config['data_seed']))
    if not jobs:
        raise ValueError('Selection contains no supported jobs')
    if len({j.run_id for j in jobs}) != len(jobs):
        raise ValueError('Duplicate jobs in selection')
    return jobs


def load_protocol(config):
    if config['data_seed'] != 42 or config['model_seeds'] != [42, 123, 2026]:
        raise ValueError('Thesis protocol requires split seed 42 and training seeds 42, 123, 2026')
    base_path = project_path(config['base_config'])
    base = read_yaml(base_path)
    validate_configured_frozen_inputs(base, keys=['raw_csv', 'labels_csv', 'feature_label_csv'])
    audit_path, audit_sha256 = validate_source_group_audit(base)
    if base['seeds']['data_seed'] != 42:
        raise ValueError('Accepted split seed changed')
    data = pd.read_csv(project_path(base['data']['canonical_input_csv']))
    labels = base['targets']['official_emotion_columns']
    outer_path = project_path(base['splits']['outer_cv']['manifest_path'])
    inner_path = project_path(base['splits']['inner_validation']['manifest_path'])
    outer = load_and_validate_outer_fold_manifest(outer_path, expected_ids=data.perfume_id.astype(str).tolist(), n_splits=5)
    inner = pd.read_csv(inner_path)
    validate_grouped_inner_manifest(inner, outer)
    expected = build_grouped_inner_validation_manifest(outer, inner_n_splits=10, seed=42)
    columns = ['perfume_id', 'outer_fold_id', 'inner_role', 'source_group_id']
    pd.testing.assert_frame_equal(inner[columns].sort_values(columns).reset_index(drop=True),
                                  expected[columns].sort_values(columns).reset_index(drop=True), check_dtype=False)
    reference = outer[['perfume_id', 'source_group_id', 'fold_id']].merge(
        data[['perfume_id', *labels]], on='perfume_id', validate='one_to_one')
    reference = reference.rename(columns={label: f'true_{label}' for label in labels}).sort_values('perfume_id').reset_index(drop=True)
    features = select_feature_columns(data, base['data']['feature_prefixes'])
    files = {str(base_path): sha256_file(base_path), str(outer_path): sha256_file(outer_path), str(inner_path): sha256_file(inner_path)}
    files[str(audit_path)] = audit_sha256
    for value in base['provenance']['frozen_input_paths'].values():
        path = project_path(value); files[str(path)] = sha256_file(path)
    return dict(base=base, base_path=base_path, reference=reference, labels=labels, features=features,
                outer=outer, inner=inner, files=files)


def validate_oof(frame, reference, labels, threshold=0.3):
    if frame.perfume_id.duplicated().any() or len(frame) != len(reference):
        raise ValueError('Incomplete or duplicate OOF rows')
    frame = frame.sort_values('perfume_id').reset_index(drop=True)
    ref = reference.sort_values('perfume_id').reset_index(drop=True)
    for column in ['perfume_id', 'fold_id', 'source_group_id', *[f'true_{l}' for l in labels]]:
        if not np.array_equal(frame[column].astype(str), ref[column].astype(str)):
            raise ValueError(f'OOF evaluation rows/targets differ: {column}')
    probability = frame[[f'prob_{l}' for l in labels]].to_numpy(float)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Incomplete or invalid OOF probabilities')
    pred = frame[[f'pred_{l}' for l in labels]].to_numpy(float)
    if not np.array_equal(pred, (probability >= threshold).astype(int)):
        raise ValueError('OOF threshold predictions differ')
    return frame


def pooled_metrics(frame, labels):
    # Established project metric implementation: AP, not trapezoidal PR area.
    values, _ = evaluate_oof(frame, labels)
    return {metric: float(values[metric]) for metric in METRICS}


def compatible_saved_contract(saved, current, *, state, root):
    """Accept only individually audited old/new contracts for a scoped repair.

    Completed runs keep their original contract and code provenance. The failed
    BERT job can adopt the fix only if training never produced a checkpoint.
    This is not a general permission to resume across code/config changes.
    """
    if saved == current:
        return True
    registry = PROJECT_ROOT / 'config/thesis_three_seed_resume_compatibility.json'
    if not registry.is_file():
        return False
    approved = json.loads(registry.read_text()).get('contracts', [])
    for entry in approved:
        if (entry['previous_sha256'] == digest(saved)
                and entry['current_sha256'] == digest(current)
                and entry['state'] == state):
            if state == 'incomplete':
                # Only the tokenizer cache was created before this specific bug.
                if any(p.name not in {'contract.json', 'tokenized.npz',
                                      'contract_before_bert_control_fix.json', '.running.lock'}
                       for p in root.iterdir()):
                    return False
            return True
    return False


def completion_status(root, contract, reference, labels):
    path = root / 'complete.json'
    if not path.exists():
        if root.exists() and any(root.iterdir()):
            manifest = root / 'contract.json'
            if not manifest.exists() or not compatible_saved_contract(
                    json.loads(manifest.read_text()), contract, state='incomplete', root=root):
                raise ValueError(f'Incompatible/unidentified incomplete output: {root}')
            return 'incomplete'
        return 'pending'
    complete = json.loads(path.read_text())
    saved = json.loads((root / 'contract.json').read_text())
    if complete.get('status') != 'complete' or complete.get('contract_sha256') != digest(saved):
        raise ValueError(f'Completed run configuration mismatch: {root}')
    if not compatible_saved_contract(saved, contract, state='complete', root=root):
        raise ValueError(f'Completed run configuration mismatch: {root}')
    for name in ('oof.csv', 'metrics.json', *[n for n in complete.get('artifacts', {}) if n not in ('oof.csv', 'metrics.json')]):
        if not (root / name).is_file() or sha256_file(root / name) != complete['artifacts'][name]:
            raise ValueError(f'Missing/corrupt completed artifact: {root / name}')
    oof = validate_oof(pd.read_csv(root / 'oof.csv'), reference, labels)
    if pooled_metrics(oof, labels) != json.loads((root / 'metrics.json').read_text()):
        raise ValueError(f'Saved metrics differ from pooled OOF: {root}')
    return 'complete'


def derive_protocol_config(source, destination):
    """Rebind only executable-code hashes, recording old/new hashes explicitly.

    All historical data, config, checkpoint, and result hashes stay frozen.
    Historical YAMLs and replay entrypoints are never modified.
    """
    source = Path(source)
    cfg = read_yaml(source)
    approved = json.loads((PROJECT_ROOT / "config/thesis_three_seed_code_revisions.json").read_text())["files"]
    revisions = {}
    for name, spec in cfg['inputs'].items():
        path = project_path(spec['path'])
        actual = sha256_file(path)
        if actual != spec['sha256']:
            if not path.is_relative_to(PROJECT_ROOT / 'src') or path.suffix != '.py':
                raise ValueError(f'Frozen non-code input changed: {path}')
            revision = approved.get(str(path.relative_to(PROJECT_ROOT)), {})
            if revision != {'historical_sha256': spec['sha256'], 'current_sha256': actual}:
                raise ValueError(f'Unreviewed code revision: {path}')
            revisions[name] = {'path': str(path), 'historical_sha256': spec['sha256'], 'current_sha256': actual}
            spec['sha256'] = actual
    cfg['three_seed_code_revision'] = {'source_config': str(source), 'source_sha256': sha256_file(source), 'revisions': revisions}
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg, sort_keys=False)
    if not destination.exists() or destination.read_text() != text:
        destination.write_text(text)
    return destination


def prepare_model(model, output_root):
    derived = derive_protocol_config(weighted_config_path(model), output_root / 'protocols' / f'{model}.yaml')
    return weighted_module(model).load_and_validate_inputs(derived)


def contract_for(job, protocol, prepared):
    features = (protocol['features'] if job.experiment == 'main' else variants_for(job.model, protocol['features'])[job.variant])
    # Bind all training/helper implementations, original input files, software,
    # checkpoint identity, and the effective configuration. Never trust a seed tag.
    code = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in sorted((PROJECT_ROOT / 'src/training').glob('*.py'))}
    inputs = {str(p): sha256_file(p) for p in prepared.paths.values()}
    packages = {}
    for name in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'torch', 'transformers']:
        packages[name] = version(name)
    return dict(schema=1, job=asdict(job), run_id=job.run_id, config=prepared.config,
                features=features if job.model in CLASSICAL else None, labels=protocol['labels'],
                threshold=0.3, accepted_inputs=protocol['files'], inputs=inputs, code=code,
                software=packages, checkpoint=getattr(prepared, 'checkpoint_provenance', None),
                aggregation='pooled_oof_then_seed_mean_sample_std', bootstrap_seed=42, permutation_seed=42)


def train_classical(job, prepared, features, *, root=None, contract=None, progress=None):
    """Preserve accepted final-fit rows and exact weighting/preprocessing rules."""
    data = prepared.youtube.merge(prepared.manifest, on=['perfume_id', 'Perfume'], validate='one_to_one').sort_values('perfume_id').reset_index(drop=True)
    oof = data[['perfume_id', 'source_group_id', 'source_group_size', 'fold_id']].copy()
    augmented = job.condition == AUGMENTED
    for label in prepared.labels:
        oof[f'true_{label}'] = data[label].astype(int)
        oof[f'prob_{label}'] = np.nan
        oof[f'pred_{label}'] = 0
    audits = []
    for fold in sorted(data.fold_id.unique()):
        training, weights, audit = build_training_rows(prepared, fold_id=int(fold), include_synthetic=augmented)
        mask = data.fold_id.eq(fold)
        x_train, x_test = training[features].to_numpy(float), data.loc[mask, features].to_numpy(float)
        for label in prepared.labels:
            checkpoint = root / f'fit_fold{fold}_{label}.json' if root else None
            fp = digest({'contract': contract, 'fold': int(fold), 'label': label})
            if progress:
                progress.update(fold=int(fold), epoch=None, current_training_progress=f'label {label}',
                                current_training_percent=100 * ((int(fold)-1)*len(prepared.labels)+list(prepared.labels).index(label))/(5*len(prepared.labels)))
            if checkpoint and checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved['fingerprint'] != fp or saved['test_ids'] != data.loc[mask, 'perfume_id'].tolist():
                    raise ValueError(f'Incompatible classical checkpoint: {checkpoint}')
                probability = np.asarray(saved['probabilities'], dtype=float)
                if probability.shape != (int(mask.sum()),) or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
                    raise ValueError(f'Incomplete classical checkpoint: {checkpoint}')
                oof.loc[mask, f'prob_{label}'] = probability
                oof.loc[mask, f'pred_{label}'] = (probability >= 0.3).astype(int)
                continue
            if progress and progress.requested():
                raise Paused('Paused before next classical fit')
            y = training[label].to_numpy(int)
            if len(np.unique(y)) < 2:
                probability = np.full(mask.sum(), float(y[0]))
            else:
                kwargs = {}
                if job.model == 'xgboost' and augmented:
                    module = weighted_module(job.model)
                    _, _, ratio = module.weighted_scale_pos_weight(y, weights)
                    model = module.build_weighted_xgboost_model(prepared.config['model'], scale_pos_weight=ratio, model_seed=job.model_seed)
                elif job.model == 'random_forest' and augmented:
                    model = weighted_module(job.model).build_weighted_random_forest_model(prepared.config['model'], model_seed=job.model_seed)
                else:
                    model = model_module(job.model).MODEL_SPEC.build_model(job.model_seed, y)
                if augmented:
                    kwargs['clf__sample_weight'] = weights
                    if job.model == 'logistic_regression':
                        kwargs['scaler__sample_weight'] = weights
                model.fit(x_train, y, **kwargs)
                probability = positive_class_probability(model, x_test)
            oof.loc[mask, f'prob_{label}'] = probability
            oof.loc[mask, f'pred_{label}'] = (probability >= 0.3).astype(int)
            if checkpoint:
                atomic_json(checkpoint, {'fingerprint': fp, 'test_ids': data.loc[mask, 'perfume_id'].tolist(), 'probabilities': probability.tolist()})
            if progress:
                progress.update(current_training_progress=f'saved fold {fold}, label {label}',
                                current_training_percent=100*((int(fold)-1)*len(prepared.labels)+list(prepared.labels).index(label)+1)/(5*len(prepared.labels)))
                if progress.requested():
                    raise Paused('Paused after saved classical fold-label fit')
        audits.append({**audit, 'data_seed': job.data_seed, 'model_seed': job.model_seed})
    return oof, audits


def train_deep(job, prepared, root, contract, progress=None):
    module = weighted_module(job.model)
    frame = module.canonical_oof_frames(prepared)[job.condition].copy()
    # Historical control frames can contain secondary tuned predictions. Those
    # belong to seed 42 and must not be carried into a newly trained seed's OOF.
    frame = frame.drop(columns=[c for c in frame if c.startswith('pred_tuned_')])
    tokenized = None
    if job.model == 'bert':
        tokenized, _ = module.load_or_build_token_cache(prepared.master_chunk_df, prepared.settings, cache_path=root / 'tokenized.npz')
    audits = []
    for fold in sorted(frame.fold_id.unique()):
        task = module.FoldTask(condition=job.condition, fold_id=int(fold))
        membership = module.fold_membership(prepared, task)
        fold_cache = root / f'fold{fold}.json'
        fold_fp = digest({'contract': contract, 'fold': int(fold)})
        if progress:
            next_epoch = 1
            epoch_path = root / f'epoch_fold{fold}.pt'
            if epoch_path.exists():
                next_epoch = module._load_epoch_checkpoint(epoch_path, fold_fp)['next_epoch']
            progress.update(fold=int(fold), epoch=min(next_epoch, prepared.settings.max_epochs), current_training_progress='training epoch or finalizing saved fold',
                            current_training_percent=100 * (int(fold)-1)/5)
        def on_epoch(record):
            if progress:
                progress.update(epoch=min(int(record['epoch']) + 1, prepared.settings.max_epochs), last_completed_epoch=int(record['epoch']),
                                current_training_progress=f"saved epoch {record['epoch']}/{prepared.settings.max_epochs}",
                                current_training_percent=100*((int(fold)-1)+int(record['epoch'])/prepared.settings.max_epochs)/5)
        kwargs = dict(task=task, membership=membership, fold_fp=digest({'contract': contract, 'fold': int(fold)}),
                      epoch_path=root / f'epoch_fold{fold}.pt', epoch_callback=on_epoch, pause_requested=lambda: progress.requested() if progress else False,
                      model_seed=job.model_seed)
        if tokenized is not None:
            kwargs['tokenized_inputs'] = tokenized
        if fold_cache.exists():
            saved = json.loads(fold_cache.read_text())
            if saved['fingerprint'] != fold_fp or saved['test_ids'] != sorted(membership['test_ids']):
                raise ValueError(f'Incompatible fold checkpoint: {fold_cache}')
            probabilities, audit = np.asarray(saved['probabilities'], float), saved['audit']
        else:
            if progress and progress.requested():
                raise Paused('Paused before next deep fold')
            try:
                probabilities, audit = module.train_fold(prepared, **kwargs)
            except module.SafePause as exc:
                raise Paused(str(exc)) from exc
            atomic_json(fold_cache, {'fingerprint': fold_fp, 'test_ids': sorted(membership['test_ids']),
                                    'probabilities': probabilities.tolist(), 'audit': audit})
        if probabilities.shape != (len(membership['test_ids']), len(prepared.labels)) or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
            raise ValueError(f'Incomplete fold checkpoint: {fold_cache}')
        # Both trainers return probabilities in sorted test-perfume order.
        ids = sorted(membership['test_ids'])
        for index, perfume in enumerate(ids):
            mask = frame.perfume_id.eq(perfume)
            for li, label in enumerate(prepared.labels):
                frame.loc[mask, f'prob_{label}'] = probabilities[index, li]
                frame.loc[mask, f'pred_{label}'] = int(probabilities[index, li] >= 0.3)
        audits.append(audit)
    return frame, audits


def execute_job(job, root, contract, protocol, prepared, *, resume=False, reuse=None, progress=None):
    status = completion_status(root, contract, protocol['reference'], protocol['labels'])
    if status == 'complete':
        if not resume:
            raise ValueError(f'Run exists; use --resume: {root}')
        return 'skipped_validated_complete'
    if status == 'incomplete' and not resume:
        raise ValueError(f'Incomplete run detected; use --resume: {root}')
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive lock prevents two runner processes writing the same seed job.
    lock = root / '.running.lock'
    # Recover an abandoned lock only when its recorded process no longer exists.
    import os
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
        except ProcessLookupError:
            lock.unlink()
    try:
        handle = lock.open('x')
    except FileExistsError as exc:
        raise ValueError(f'Run locked: {lock}; remove only after confirming its process stopped') from exc
    try:
        import os
        handle.write(str(os.getpid())); handle.close()
        if (root / 'contract.json').exists():
            previous = json.loads((root / 'contract.json').read_text())
            if previous != contract:
                atomic_json(root / f'contract_before_repair_{digest(previous)}.json', previous)
        atomic_json(root / 'contract.json', contract)
        if reuse is not None:
            for name, expected_hash in {**reuse['evidence'], reuse['prediction_path']: reuse['prediction_sha256']}.items():
                if sha256_file(Path(name)) != expected_hash:
                    raise ValueError(f'Historical reuse source changed since audit: {name}')
            frame = pd.read_csv(reuse['prediction_path'])
            audits = [{'historical_reuse': reuse}]
        elif job.model in CLASSICAL:
            frame, audits = train_classical(job, prepared, contract['features'], root=root, contract=contract, progress=progress)
        else:
            frame, audits = train_deep(job, prepared, root, contract, progress=progress)
        frame = validate_oof(frame, protocol['reference'], protocol['labels'])
        frame.to_csv(root / 'oof.csv', index=False)
        # Read back exactly the persisted probabilities before marking complete.
        frame = validate_oof(pd.read_csv(root / 'oof.csv'), protocol['reference'], protocol['labels'])
        atomic_json(root / 'metrics.json', pooled_metrics(frame, protocol['labels']))
        atomic_json(root / 'training_audit.json', {'folds': audits, 'data_seed': job.data_seed, 'model_seed': job.model_seed})
        atomic_json(root / 'complete.json', {'status': 'complete', 'contract_sha256': digest(contract),
                    'artifacts': {name: sha256_file(root / name) for name in ['oof.csv', 'metrics.json', 'training_audit.json']}, 'historical_reuse': reuse})
    finally:
        lock.unlink(missing_ok=True)
    return 'completed'


def seed_summary(values, expected=(42, 123, 2026)):
    present = sorted(set(values) & set(expected))
    return {'completed_seeds': present, 'n_completed_seeds': len(present), 'incomplete': present != sorted(expected),
            'mean': float(np.mean([values[s] for s in present])) if present else None,
            'sample_std_ddof1': float(np.std([values[s] for s in present], ddof=1)) if len(present) > 1 else None}


def summarize(jobs, contracts, root, protocol):
    groups = {}
    for job in jobs:
        key = (job.experiment, job.model, job.condition, job.variant)
        groups.setdefault(key, {})
        path = root / job.run_id
        if completion_status(path, contracts[job.run_id], protocol['reference'], protocol['labels']) == 'complete':
            frame = validate_oof(pd.read_csv(path / 'oof.csv'), protocol['reference'], protocol['labels'])
            groups[key][job.model_seed] = pooled_metrics(frame, protocol['labels'])
    summaries = []
    for key, seeds in groups.items():
        for metric in METRICS:
            summaries.append(dict(zip(['experiment', 'model', 'condition', 'variant'], key), metric=metric,
                                  **seed_summary({s: v[metric] for s, v in seeds.items()})))
    comparisons = []
    for key, seeds in groups.items():
        experiment, model, condition, variant = key
        comparator = (experiment, model, CONTROL, variant) if condition == AUGMENTED else (experiment, model, condition, 'full_hybrid')
        if comparator == key or comparator not in groups:
            continue
        matched = sorted(set(seeds) & set(groups[comparator]))
        for metric in METRICS:
            differences = {s: seeds[s][metric] - groups[comparator][s][metric] for s in matched}
            comparisons.append({'comparison': list(key), 'minus': list(comparator), 'metric': metric,
                                'per_seed_differences': differences, **seed_summary(differences)})
    result = {'aggregation': 'Metrics pooled over all five OOF folds per seed, then mean and sample SD across seeds.',
              'uncertainty': 'Training seed variability only; historical bootstrap CIs are separate.',
              'summaries': summaries, 'paired_comparisons': comparisons}
    atomic_json(root / 'summary.json', result)
    pd.DataFrame(summaries).to_csv(root / 'summary.csv', index=False)
    return result


def checksum_entries(root):
    manifest = root / 'SHA256SUMS'
    result = {}
    for line in manifest.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        path = (root / name.strip().lstrip('*')).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Historical checksum path escapes its root')
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f'Historical artifact checksum mismatch: {path}')
        result[str(path)] = checksum
    return result


def audit_seed42(job, protocol, prepared):
    """Audit immutable evidence; uncertain provenance means retraining, never reuse."""
    entry = {'run_id': job.run_id, 'status': 'requires_retraining', 'checks': []}
    if job.model_seed != 42:
        entry['reason'] = 'No historical result for this training seed'
        return entry
    try:
        base = protocol['base']
        historical_root = project_path(base['benchmark']['experiment_root'])
        benchmark = project_path(base['artifacts']['root_dir'])
        metrics_path = benchmark / 'metrics' / f'{job.model}.json'
        accepted = json.loads(metrics_path.read_text())
        run_manifest_path = benchmark / 'logs' / f'{job.model}_run.json'
        run_manifest = json.loads(run_manifest_path.read_text())
        historical_software = run_manifest['package_versions']
        require(accepted['data_seed'] == accepted['model_seed'] == 42, 'Historical seeds differ')
        require(accepted['config_sha256'] == sha256_file(protocol['base_path']), 'Historical base config differs')
        require(accepted['threshold'] == 0.3, 'Threshold differs')
        for name, key in [('outer_manifest_sha256', 'outer_cv'), ('inner_manifest_sha256', 'inner_validation')]:
            require(accepted['split_provenance'][name] == sha256_file(project_path(base['splits'][key]['manifest_path'])), f'{key} differs')
        require(accepted['target_columns'] == protocol['labels'], 'Target definitions/order differ')
        require(accepted['dataset_sha256'] == sha256_file(project_path(accepted['dataset_path'])), 'Dataset differs')
        if job.model in CLASSICAL:
            require(accepted['model_parameters'] == dict(model_module(job.model).MODEL_SPEC.model_params), 'Model/preprocessing settings differ')
        else:
            for name, value in base['models'][job.model].items():
                require((accepted['chunk_aggregation']['strategy'] if name == 'aggregation_strategy' else accepted['model_parameters'][name]) == value, f'Deep setting differs: {name}')
            require(accepted['model_parameters']['training_mode'] == prepared.config['model']['training_mode'], 'Architecture differs')
            require(accepted['frozen_input_sha256']['labels_csv'] == base['provenance']['frozen_input_sha256']['labels_csv'], 'Labels differ')
            # The prepared loader checks transcript-only data, pinned checkpoint,
            # original model settings, synthetic corpus, and shared inner splits.
            if job.model == 'bert':
                require(prepared.control_replay_validation['max_absolute_probability_difference'] == 0, 'Pinned-checkpoint control replay differs')
        entry['checks'] += ['dataset and labels', 'outer and inner manifests', 'model/checkpoint/preprocessing', 'threshold and seeds']
        prediction = benchmark / 'predictions' / f'{job.model}_oof.csv'
        evidence = [metrics_path, run_manifest_path]
        if job.experiment == 'structured_ablation':
            name = 'source_grouped_feature_ablation_v1' if job.model == 'logistic_regression' else f'source_grouped_{job.model}_feature_combination_ablation_v1'
            historical_root = PROJECT_ROOT / 'outputs' / name
            report_path = historical_root / 'REPORT.json'
            report = json.loads(report_path.read_text())
            require(report['status'] in ('passed', 'completed'), 'Ablation report not complete')
            expected_features = variants_for(job.model, protocol['features'])[job.variant]
            if job.model == 'logistic_regression':
                design = report['design']
                require(design['seed'] == 42 and design['threshold'] == 0.3, 'Ablation seed/threshold differ')
                require(design['model'] == accepted['model_parameters'], 'Ablation model differs')
                actual_features = next(v['feature_columns'].split(';') for v in design['variants'] if v['variant'] == job.variant)
                require(actual_features == expected_features, 'Feature definitions differ')
                for key, value in [('config', protocol['base_path']), ('feature_csv', project_path(base['data']['canonical_input_csv'])), ('manifest_csv', project_path(base['splits']['outer_cv']['manifest_path']))]:
                    require(report['inputs']['sha256'][key] == sha256_file(value), f'Ablation input differs: {key}')
            else:
                source = PROJECT_ROOT / f'config/evaluation_config_source_grouped_{job.model}_feature_combination_ablation_v1.yaml'
                cfg = read_yaml(source)
                require(report['protocol_sha256'] == sha256_file(source), 'Ablation protocol differs')
                require(report['model'] == cfg['model'], 'Ablation model differs')
                require(report['threshold'] == 0.3, 'Ablation threshold differs')
                require(yaml.safe_load((historical_root / 'PROTOCOL.yaml').read_text()) == cfg, 'Saved ablation protocol differs')
            prediction = historical_root / 'oof_predictions' / f'{job.variant}_oof.csv'
            evidence.append(report_path)
            historical_software = report['software'].get('package_versions', report['software'])
        elif job.condition == AUGMENTED:
            cfg_path = weighted_config_path(job.model)
            cfg = read_yaml(cfg_path)
            historical_root = project_path(cfg['outputs']['root'])
            report_path = historical_root / 'REPORT.json'
            report = json.loads(report_path.read_text())
            require(report['status'] in ('passed', 'complete', 'completed'), 'Augmentation not complete')
            require(report['protocol_sha256'] == sha256_file(cfg_path), 'Augmentation protocol differs')
            require(report['model'] == cfg['model'], 'Augmentation model differs')
            require(report['threshold'] == 0.3, 'Augmentation threshold differs')
            require(yaml.safe_load((historical_root / 'PROTOCOL.yaml').read_text()) == cfg, 'Saved weighted protocol differs')
            validation = json.loads((historical_root / 'validation_report.json').read_text())
            require(validation['status'] == 'passed', 'Augmentation validation did not pass')
            prediction = historical_root / 'oof_predictions' / f'{AUGMENTED}_oof.csv'
            historical_software = report.get('software', historical_software)
            evidence.extend([report_path, historical_root / 'PROTOCOL.yaml', historical_root / 'validation_report.json'])
        for package, recorded_version in historical_software.items():
            package = package.replace('scikit_learn', 'scikit-learn')
            if package in ('numpy', 'pandas', 'scikit-learn', 'torch', 'transformers', 'xgboost'):
                require(version(package) == recorded_version, f'Training software version differs: {package}')
        entries = checksum_entries(historical_root)
        require(str(prediction.resolve()) in entries, 'Prediction absent from historical checksum manifest')
        for path in evidence:
            if path.is_relative_to(historical_root):
                require(str(path.resolve()) in entries, f'Evidence absent from checksum manifest: {path}')
        frame = validate_oof(pd.read_csv(prediction), protocol['reference'], protocol['labels'])
        values = pooled_metrics(frame, protocol['labels'])
        if job.experiment == 'main' and job.condition == CONTROL:
            for metric, value in values.items():
                field = 'macro_pr_auc' if metric == 'macro_average_precision' else metric
                require(np.isclose(value, accepted[field], atol=1e-12, rtol=0), f'Historical metric differs: {metric}')
        entry.update(status='reusable', reason='Frozen protocol and complete predictions validated',
                     prediction_path=str(prediction), prediction_sha256=sha256_file(prediction), historical_software=historical_software,
                     evidence={str(p): sha256_file(p) for p in [*evidence, historical_root / 'SHA256SUMS']}, metrics=values)
        entry['checks'] += ['condition/variant protocol', 'historical artifact checksums', 'complete matching OOF rows and targets', 'pooled metrics recomputed']
    except (AssertionError, ValueError, KeyError, FileNotFoundError, StopIteration) as exc:
        entry['reason'] = str(exc) or type(exc).__name__
    return entry


class Paused(RuntimeError):
    pass


class Progress:
    """30-second heartbeat, independent of slow estimator fits/epochs."""
    def __init__(self, root, jobs, completed=0):
        import threading
        import time
        self.root, self.jobs = root, jobs
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.mutex = threading.RLock()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.pause = False
        self.state = dict(status='running', completed_jobs=completed, total_jobs=len(jobs),
                          model=None, condition=None, seed=None, fold=None, epoch=None,
                          current_training_progress='preflight', current_training_percent=None)
        self.runtimes = json.loads((root / 'runtimes.json').read_text()) if (root / 'runtimes.json').exists() else {}
        self.previous = {}

    def requested(self):
        return self.pause or (self.root / 'PAUSE').exists()

    def _signal(self, signum, frame):
        self.pause = True

    def __enter__(self):
        import signal
        for sig in [signal.SIGINT, signal.SIGTERM]:
            self.previous[sig] = signal.signal(sig, self._signal)
        self.emit(); self.thread.start()
        return self

    def __exit__(self, *args):
        import signal
        self.stop.set(); self.thread.join(timeout=2)
        for sig, previous in self.previous.items():
            signal.signal(sig, previous)
        self.emit()

    def _heartbeat(self):
        while not self.stop.wait(30):
            self.emit()

    def update(self, **values):
        with self.mutex:
            self.state.update(values)
            self.emit()

    def observed(self, key, seconds):
        with self.mutex:
            self.runtimes.setdefault(key, []).append(seconds)
            atomic_json(self.root / 'runtimes.json', self.runtimes)

    def emit(self):
        import time
        with self.mutex:
            state = dict(self.state)
            remaining = self.jobs[state['completed_jobs']:]
            estimates = []
            for job in remaining:
                observations = self.runtimes.get(f'{job.model}/{job.condition}/{job.variant}', [])
                if len(observations) < 2:
                    estimates = None; break
                estimates.append(float(np.mean(observations)))
            state.update(elapsed_seconds=time.monotonic() - self.started,
                         remaining_seconds=sum(estimates) if estimates is not None else 'estimating',
                         overall_job_completion_percent=100 * state['completed_jobs'] / max(1, state['total_jobs']),
                         job_percentage_is_elapsed_time_percentage=False,
                         pause_requested=self.requested(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            atomic_json(self.root / 'status.json', state)
            print(json.dumps(state, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--models', nargs='+', choices=[*CLASSICAL, 'bert', 'bilstm_attention'])
    parser.add_argument('--conditions', nargs='+', choices=[CONTROL, AUGMENTED])
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 123, 2026])
    parser.add_argument('--experiments', nargs='+', choices=['main', 'structured_ablation'])
    parser.add_argument('--variants', nargs='+')
    parser.add_argument('--execute', action='store_true', help='Explicitly enable training')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reuse-seed42', action='store_true', help='Import only freshly audited reusable historical predictions')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args(argv)
    config = read_yaml(args.config)
    root = project_path(config['output_root'])
    # A configurable root must still remain under the new thesis namespace.
    if not root.is_relative_to(PROJECT_ROOT / 'outputs/thesis_three_seed_v1'):
        raise ValueError('Output root must remain under outputs/thesis_three_seed_v1')
    if args.status:
        path = root / 'status.json'
        if path.exists():
            print(path.read_text())
        elif (root / 'inventory.json').exists():
            rows = json.loads((root / 'inventory.json').read_text())
            print(json.dumps({'status': 'planned_only', 'total_jobs': len(rows), 'completed_jobs': sum(r['status'] == 'complete' for r in rows),
                              'inventory': str(root / 'inventory.csv'), 'training_launched': False}, indent=2))
        else:
            print('No execution status yet; use --dry-run for inventory.')
        return 0
    if args.pause:
        root.mkdir(parents=True, exist_ok=True)
        (root / 'PAUSE').touch()
        print('Pause requested: finishes and checkpoints the current epoch or classical fold-label fit.')
        return 0
    if args.execute and args.dry_run:
        parser.error('--execute and --dry-run are mutually exclusive')
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(config)
    jobs = build_jobs(config, protocol['features'], models=args.models, conditions=args.conditions,
                      seeds=args.seeds, experiments=args.experiments, variants=args.variants)
    prepared = {model: prepare_model(model, root) for model in dict.fromkeys(j.model for j in jobs)}
    contracts = {j.run_id: contract_for(j, protocol, prepared[j.model]) for j in jobs}
    audits = {j.run_id: audit_seed42(j, protocol, prepared[j.model]) for j in jobs if j.model_seed == 42}
    atomic_json(root / 'seed42_reuse_audit.json', list(audits.values()))
    audit_lines = ['# Historical seed-42 reuse audit', '',
                   'All entries were checked against the fixed dataset, labels, source groups, outer/inner splits, model settings, preprocessing, threshold, and complete predictions. Deep loaders also validate text and checkpoint provenance. Historical files remain unchanged.', '',
                   '| Job | Decision | Reason |', '| --- | --- | --- |']
    audit_lines += [f"| {a['run_id']} | {a['status']} | {a['reason']} |" for a in audits.values()]
    audit_lines += ['', 'All selected seed-123 and seed-2026 jobs require new training. Reuse is opt-in with `--reuse-seed42`; the JSON audit retains source paths and hashes.']
    (root / 'seed42_reuse_audit.md').write_text('\n'.join(audit_lines) + '\n')
    inventory = []
    for job in jobs:
        status = completion_status(root / job.run_id, contracts[job.run_id], protocol['reference'], protocol['labels'])
        command = [sys.executable, '-m', 'src.training.run_thesis_three_seed', '--config', str(args.config),
                   '--models', job.model, '--conditions', job.condition, '--experiments', job.experiment,
                   '--variants', job.variant, '--seeds', str(job.model_seed), '--execute', '--resume']
        if args.reuse_seed42:
            command.append('--reuse-seed42')
        row = {**asdict(job), 'command': command, 'contract_sha256': digest(contracts[job.run_id]), 'run_id': job.run_id, 'output': str(root / job.run_id), 'status': status,
               'historical_seed42': audits.get(job.run_id, {}).get('status', 'requires_retraining')}
        inventory.append(row)
        print(json.dumps(row), flush=True)
    atomic_json(root / 'inventory.json', inventory)
    pd.DataFrame(inventory).to_csv(root / 'inventory.csv', index=False)
    atomic_json(root / 'split_validation.json', {'status': 'passed', 'data_seed': 42, 'perfumes': len(protocol['reference']),
                'source_groups': int(protocol['outer'].source_group_id.nunique()), 'folds': 5,
                'outer_and_inner_fixed_for_all_jobs': True, 'frozen_files': protocol['files']})
    if args.summarize or not args.execute:
        summarize(jobs, contracts, root, protocol)
        print(f'{len(jobs)} jobs; {sum(a["status"] == "reusable" for a in audits.values())} seed-42 runs reusable. No training launched.')
        return 0
    import time
    if args.resume:
        (root / 'PAUSE').unlink(missing_ok=True)
    with Progress(root, jobs) as progress:
        try:
            for job in jobs:
                if progress.requested():
                    raise Paused('Paused between jobs')
                progress.update(model=job.model, condition=job.condition, seed=job.model_seed, fold=None, epoch=None,
                                current_training_progress='starting or validating saved job', current_training_percent=0)
                started = time.monotonic()
                reuse = audits.get(job.run_id) if args.reuse_seed42 else None
                if reuse and reuse['status'] != 'reusable':
                    reuse = None
                result = execute_job(job, root / job.run_id, contracts[job.run_id], protocol, prepared[job.model],
                                     resume=args.resume, reuse=reuse, progress=progress)
                if result == 'completed' and reuse is None and next(row['status'] for row in inventory if row['run_id'] == job.run_id) == 'pending':
                    progress.observed(f'{job.model}/{job.condition}/{job.variant}', time.monotonic() - started)
                progress.update(completed_jobs=progress.state['completed_jobs'] + 1, current_training_percent=100)
            progress.update(status='complete')
        except Paused as exc:
            progress.update(status='paused', current_training_progress=str(exc))
            return 0
        except BaseException:
            progress.update(status='failed')
            raise
    summarize(jobs, contracts, root, protocol)
    return 0


if __name__ == '__main__':
    sys.exit(main())
