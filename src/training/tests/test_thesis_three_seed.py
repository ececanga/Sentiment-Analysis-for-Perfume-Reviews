"""Small, executable checks; no full-data model training."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
import yaml

from src.training.experiment_seeds import data_seed, model_seed
from src.training import run_thesis_three_seed as runner
from src.training.tests.test_run_classical_model_benchmarks import (
    _make_dataset, _write_config, _seed_existing_comparison_table, _build_dummy_xgb_model,
)
from src.training.model_benchmark_runner import BenchmarkModelSpec, run_classical_model_benchmark


class SeedSupportTests(unittest.TestCase):
    def test_backward_compatibility_and_independent_seed_fields(self):
        self.assertEqual(model_seed({'seeds': {'data_seed': 7}}), 7)
        self.assertEqual(model_seed({'model': {'random_seed': 13}}), 13)
        cfg = {'seeds': {'data_seed': 42, 'model_seed': 2026}, 'bootstrap': {'seed': 8}}
        self.assertEqual((data_seed(cfg), model_seed(cfg)), (42, 2026))
        self.assertEqual(cfg['bootstrap']['seed'], 8)

    def test_seed_reaches_both_classical_fits_without_changing_splits(self):
        results, seen = [], []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (42, 123, 2026, 7, 99):
                cfg_path = root / f'config{seed}.yaml'
                _write_config(cfg_path)
                cfg = yaml.safe_load(cfg_path.read_text())
                cfg['seeds']['model_seed'] = seed
                cfg['evaluation']['threshold_tuning'] = {'enabled': True}
                cfg_path.write_text(yaml.safe_dump(cfg))
                csv = root / 'data.csv'
                _make_dataset().to_csv(csv, index=False)
                out = root / str(seed)
                _seed_existing_comparison_table(out)
                def build(actual_seed, y):
                    seen.append(actual_seed)
                    return _build_dummy_xgb_model(actual_seed, y)
                spec = BenchmarkModelSpec('xgboost', 'XGBoost', build, {}, [])
                result = run_classical_model_benchmark(input_csv=csv, out_root=out, config_path=cfg_path, model_spec=spec)
                self.assertEqual(set(seen), {seed}); seen.clear()
                payload = json.loads(result['metrics_path'].read_text())
                self.assertEqual((payload['data_seed'], payload['model_seed']), (42, seed))
                results.append((pd.read_csv(result['fold_manifest_path']),
                                pd.read_csv(result['inner_validation_manifest_path']), payload['run_id']))
            for result in results[1:]:
                pd.testing.assert_frame_equal(results[0][0], result[0])
                pd.testing.assert_frame_equal(results[0][1], result[1])
            self.assertEqual(len({r[2] for r in results}), 5)

    def test_seed_cohort_validation_and_subset_selection(self):
        config = runner.read_yaml(runner.DEFAULT_CONFIG)
        self.assertEqual(runner.configured_seeds(config), (42, 123, 2026, 7, 99))
        self.assertEqual(runner.configured_seeds({**config, 'model_seeds': [42, 123, 2026]}), (42, 123, 2026))
        for seeds in [[42, 7], [42, 7, 7], [42, True, 99], [42, -1, 99], [42, 2**32, 99]]:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                runner.configured_seeds({**config, 'model_seeds': seeds})
        with self.assertRaisesRegex(ValueError, 'split seed'):
            runner.configured_seeds({**config, 'data_seed': 7})
        for seeds in [[8], [7, 7], []]:
            with self.subTest(selection=seeds), self.assertRaises(ValueError):
                runner.build_jobs(config, [], seeds=seeds)
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit) as error:
            runner.main(['--dry-run', '--seeds', '8'])
        self.assertEqual(error.exception.code, 2)

    def test_estimator_overrides_and_shuffling(self):
        from src.training.run_source_grouped_weighted_synthetic_random_forest import build_weighted_random_forest_model
        from src.training.run_source_grouped_weighted_synthetic_xgboost import build_weighted_xgboost_model
        for model, build in [('random_forest', build_weighted_random_forest_model), ('xgboost', build_weighted_xgboost_model)]:
            cfg = runner.read_yaml(runner.weighted_config_path(model))['model']
            kw = {'scale_pos_weight': 2.0} if model == 'xgboost' else {}
            self.assertEqual(build(cfg, model_seed=2026, **kw).named_steps['clf'].random_state, 2026)
            self.assertEqual(build(cfg, **kw).named_steps['clf'].random_state, 42)
        from src.training.run_bert_benchmark import set_global_seed
        samples = []
        for seed in [42, 123, 42]:
            set_global_seed(seed)
            samples.append(torch.rand(4))
        self.assertTrue(torch.equal(samples[0], samples[2]))
        self.assertFalse(torch.equal(samples[0], samples[1]))

    def test_real_split_inventory_and_synthetic_exclusion_for_every_seed(self):
        config = runner.read_yaml(runner.DEFAULT_CONFIG)
        protocol = runner.load_protocol(config)
        jobs = runner.build_jobs(config, protocol['features'])
        self.assertEqual(len(jobs), 195)
        self.assertEqual(len({j.run_id for j in jobs}), 195)
        added = runner.build_jobs(config, protocol['features'], seeds=[7, 99])
        self.assertEqual(len(added), 78)
        old = runner.build_jobs({**config, 'model_seeds': [42, 123, 2026]}, protocol['features'])
        self.assertEqual(len(old), 117)
        self.assertEqual(set(old) | set(added), set(jobs))
        with tempfile.TemporaryDirectory() as tmp:
            for model in ['bert', 'bilstm_attention']:
                prepared = runner.prepare_model(model, Path(tmp))
                module = runner.weighted_module(model)
                baseline = None
                for seed in config['model_seeds']:
                    for condition in [runner.CONTROL, runner.AUGMENTED]:
                        for fold in range(1, 6):
                            member = module.fold_membership(prepared, module.FoldTask(condition, fold))
                            selected = prepared.master_chunk_df.loc[member['train_mask']]
                            synthetic = selected.loc[selected.is_synthetic]
                            self.assertFalse(set(synthetic.perfume_id) & (member['val_ids'] | member['test_ids']))
                            if condition == runner.CONTROL:
                                self.assertEqual(len(synthetic), 0)
                                from src.training.run_bert_benchmark import compute_pos_weight
                                weights, _ = module.weighted_pos_weight(prepared, membership=member, condition=condition)
                                natural = prepared.perfume_df.loc[prepared.perfume_df.perfume_id.isin(member['train_ids'])]
                                np.testing.assert_array_equal(weights, compute_pos_weight(natural, prepared.labels))
                    current = prepared.inner_manifest.copy()
                    if baseline is not None: pd.testing.assert_frame_equal(baseline, current)
                    baseline = current


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.labels = ['joy']
        self.reference = pd.DataFrame({'perfume_id': ['a', 'b', 'c', 'd'], 'source_group_id': ['a', 'b', 'c', 'd'],
                                       'fold_id': [1, 1, 2, 2], 'true_joy': [0, 1, 0, 1]})
        self.frame = self.reference.assign(prob_joy=[.1, .9, .8, .2], pred_joy=[0, 1, 1, 0])
        self.contract = {'job': 'test', 'seed': 42}

    def complete(self, path, frame=None, contract=None):
        path.mkdir(parents=True, exist_ok=True)
        contract = contract or self.contract
        runner.atomic_json(path / 'contract.json', contract)
        (self.frame if frame is None else frame).to_csv(path / 'oof.csv', index=False)
        runner.atomic_json(path / 'metrics.json', runner.pooled_metrics(pd.read_csv(path / 'oof.csv'), self.labels))
        runner.atomic_json(path / 'complete.json', {'status': 'complete', 'contract_sha256': runner.digest(contract),
                           'artifacts': {n: runner.sha256_file(path/n) for n in ['oof.csv', 'metrics.json']}})

    def test_resume_rejects_changed_config_corruption_and_incomplete_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'job'
            self.assertEqual(runner.completion_status(path, self.contract, self.reference, self.labels), 'pending')
            path.mkdir(); runner.atomic_json(path/'contract.json', self.contract)
            self.assertEqual(runner.completion_status(path, self.contract, self.reference, self.labels), 'incomplete')
            self.complete(path)
            self.assertEqual(runner.completion_status(path, self.contract, self.reference, self.labels), 'complete')
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                runner.completion_status(path, {'seed': 123}, self.reference, self.labels)
            (path/'oof.csv').write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                runner.completion_status(path, self.contract, self.reference, self.labels)
            self.complete(path, self.frame.iloc[:2])
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                runner.completion_status(path, self.contract, self.reference, self.labels)

    def test_scoped_code_repair_preserves_contracts_and_rejects_other_changes(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(runner, 'PROJECT_ROOT', Path(tmp)):
            root = Path(tmp)
            path = root / 'job'
            self.complete(path)
            previous_bytes = (path / 'contract.json').read_bytes()
            current = {**self.contract, 'code': {'runner.py': 'new-code'}}
            registry = root / 'config/thesis_three_seed_resume_compatibility.json'
            entries = [{'previous_sha256': runner.digest(self.contract), 'current_sha256': runner.digest(current), 'state': state}
                       for state in ['complete', 'incomplete']]
            runner.atomic_json(registry, {'contracts': entries})
            self.assertEqual(runner.completion_status(path, current, self.reference, self.labels), 'complete')
            self.assertEqual((path / 'contract.json').read_bytes(), previous_bytes)
            with self.assertRaises(ValueError):
                runner.completion_status(path, {**current, 'seed': 123}, self.reference, self.labels)
            with self.assertRaises(ValueError):
                runner.completion_status(path, {**current, 'code': {'runner.py': 'unknown'}}, self.reference, self.labels)
            incomplete = root / 'unstarted'
            incomplete.mkdir()
            runner.atomic_json(incomplete / 'contract.json', self.contract)
            (incomplete / 'tokenized.npz').touch()
            self.assertEqual(runner.completion_status(incomplete, current, self.reference, self.labels), 'incomplete')
            (incomplete / 'epoch_fold1.pt').touch()
            with self.assertRaises(ValueError):
                runner.completion_status(incomplete, current, self.reference, self.labels)

    def test_oof_rejects_wrong_targets_groups_threshold_and_nan(self):
        for column, value in [('true_joy', 1), ('source_group_id', 'wrong'), ('pred_joy', 1), ('prob_joy', float('nan'))]:
            frame = self.frame.copy(); frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                runner.validate_oof(frame, self.reference, self.labels)

    def test_summary_pools_per_seed_and_pairs_matching_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs, contracts, expected = [], {}, {}
            probabilities = {42: [.1,.9,.8,.9], 123: [.8,.2,.8,.2], 2026: [.1,.9,.1,.9]}
            for condition in [runner.CONTROL, runner.AUGMENTED]:
                for seed, prob in probabilities.items():
                    job = runner.Job('main', 'logistic_regression', condition, 'full_hybrid', seed)
                    jobs.append(job); contract = {'job': runner.asdict(job)}; contracts[job.run_id] = contract
                    frame = self.reference.assign(prob_joy=prob, pred_joy=(np.array(prob)>=.3).astype(int))
                    self.complete(root/job.run_id, frame, contract)
                    expected[seed] = runner.pooled_metrics(frame, self.labels)['macro_f1']
            summary = runner.summarize(jobs, contracts, root, {'reference': self.reference, 'labels': self.labels,
                                                             'model_seeds': (42, 123, 2026)})
            row = next(r for r in summary['summaries'] if r['condition']==runner.CONTROL and r['metric']=='macro_f1')
            self.assertAlmostEqual(row['mean'], np.mean(list(expected.values())))
            self.assertAlmostEqual(row['sample_std_ddof1'], np.std(list(expected.values()), ddof=1))
            self.assertFalse(row['incomplete'])
            from sklearn.metrics import f1_score
            ensemble = (np.mean(list(probabilities.values()), axis=0) >= .3).astype(int)
            self.assertNotAlmostEqual(row['mean'], f1_score(self.reference.true_joy, ensemble))
            fold_average = np.mean([f1_score(self.reference.true_joy.iloc[indices],
                (np.array(probabilities[42])[indices] >= .3).astype(int)) for indices in [[0,1],[2,3]]])
            self.assertNotAlmostEqual(expected[42], fold_average)
            for difference in summary['paired_comparisons']:
                self.assertEqual(difference['mean'], 0)
            self.assertTrue(runner.seed_summary({42: .5}, expected=(42, 123, 2026))['incomplete'])
            self.assertIsNone(runner.seed_summary({42: .5}, expected=(42, 123, 2026))['sample_std_ddof1'])

    def test_five_seed_summary_includes_new_seeds_and_requires_matching_pairs(self):
        seeds = (42, 123, 2026, 7, 99)
        protocol = {'reference': self.reference, 'labels': self.labels, 'model_seeds': seeds}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs, contracts, values = [], {}, {}
            for condition in [runner.CONTROL, runner.AUGMENTED]:
                values[condition] = {}
                for seed in seeds:
                    job = runner.Job('main', 'logistic_regression', condition, 'full_hybrid', seed)
                    jobs.append(job)
                    contract = {'job': runner.asdict(job)}
                    contracts[job.run_id] = contract
                    if condition == runner.AUGMENTED and seed == 99:
                        continue
                    # New seeds have deliberately different metrics, including
                    # a paired difference that a hardcoded three-seed summary loses.
                    prob = ([.1, .9, .1, .9] if seed == 7 and condition == runner.AUGMENTED
                            else [.9, .1, .9, .1] if seed in (7, 99)
                            else [.1, .9, .8, .9])
                    frame = self.reference.assign(prob_joy=prob, pred_joy=(np.array(prob) >= .3).astype(int))
                    self.complete(root/job.run_id, frame, contract)
                    values[condition][seed] = runner.pooled_metrics(frame, self.labels)['macro_f1']
            summary = runner.summarize(jobs, contracts, root, protocol)
            rows = {r['condition']: r for r in summary['summaries'] if r['metric'] == 'macro_f1'}
            control = rows[runner.CONTROL]
            self.assertEqual(control['n_expected_seeds'], 5)
            self.assertEqual(control['completed_seeds'], list(seeds))
            self.assertFalse(control['incomplete'])
            self.assertAlmostEqual(control['mean'], np.mean(list(values[runner.CONTROL].values())))
            self.assertAlmostEqual(control['sample_std_ddof1'], np.std(list(values[runner.CONTROL].values()), ddof=1))
            self.assertEqual(rows[runner.AUGMENTED]['missing_seeds'], [99])
            pair = next(r for r in summary['paired_comparisons'] if r['metric'] == 'macro_f1')
            differences = [values[runner.AUGMENTED][s] - values[runner.CONTROL][s] for s in seeds[:-1]]
            self.assertTrue(pair['incomplete'])
            self.assertEqual(pair['n_completed_seeds'], 4)
            self.assertAlmostEqual(pair['mean'], np.mean(differences))
            self.assertAlmostEqual(pair['sample_std_ddof1'], np.std(differences, ddof=1))
            three = runner.seed_summary({42: .2, 123: .4, 2026: .6}, expected=seeds)
            self.assertTrue(three['incomplete'])
            self.assertEqual(three['missing_seeds'], [7, 99])
            with self.assertRaises(ValueError):
                runner.seed_summary({888: .5}, expected=seeds)

    def test_extension_dry_run_and_resume_preserve_completed_jobs(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            project = Path(tmp).resolve()
            root = project / 'outputs/thesis_three_seed_v1'
            config = {'output_root': 'outputs/thesis_three_seed_v1', 'data_seed': 42,
                      'model_seeds': [42, 123, 2026, 7, 99], 'models': ['logistic_regression'],
                      'conditions': [runner.CONTROL], 'experiments': ['main']}
            path = project / 'config.yaml'
            path.write_text(yaml.safe_dump(config))
            protocol = {'features': ['NRC_joy'], 'reference': self.reference, 'labels': self.labels,
                        'outer': self.reference, 'files': {}, 'model_seeds': tuple(config['model_seeds'])}
            jobs = runner.build_jobs(config, protocol['features'])
            original_bytes = {}
            for job in jobs[:3]:
                self.complete(root/job.run_id, contract={'job': runner.asdict(job)})
                for artifact in (root/job.run_id).iterdir():
                    original_bytes[artifact] = artifact.read_bytes()
            stack.enter_context(mock.patch.object(runner, 'PROJECT_ROOT', project))
            stack.enter_context(mock.patch.object(runner, 'load_protocol', return_value=protocol))
            stack.enter_context(mock.patch.object(runner, 'prepare_model', return_value=None))
            stack.enter_context(mock.patch.object(runner, 'contract_for', side_effect=lambda job, *a: {'job': runner.asdict(job)}))
            stack.enter_context(mock.patch.object(runner, 'audit_seed42', side_effect=lambda job, *a: {
                'run_id': job.run_id, 'status': 'reusable', 'reason': 'test fixture'}))
            stack.enter_context(mock.patch('builtins.print'))
            with mock.patch.object(runner, 'execute_job') as execute:
                runner.main(['--config', str(path), '--dry-run'])
                execute.assert_not_called()
            status = json.loads((root/'status.json').read_text())
            self.assertEqual((status['completed_jobs'], status['total_jobs']), (3, 5))
            self.assertEqual(status['overall_job_completion_percent'], 60)
            self.assertEqual(status['status'], 'ready')
            self.assertFalse(status['training_launched'])
            self.assertTrue(all(r['incomplete'] for r in json.loads((root/'summary.json').read_text())['summaries']))
            with self.assertRaisesRegex(ValueError, '--resume'):
                runner.main(['--config', str(path), '--execute'])
            before_training = []
            def train(job, output, contract, *args, **kwargs):
                before_training.append(json.loads((root/'status.json').read_text())['completed_jobs'])
                self.complete(output, contract=contract)
                return 'completed'
            with mock.patch.object(runner, 'execute_job', side_effect=train) as execute:
                runner.main(['--config', str(path), '--execute', '--resume'])
                self.assertEqual([call.args[0].model_seed for call in execute.call_args_list], [7, 99])
            self.assertEqual(before_training, [3, 4])
            self.assertTrue(all(p.read_bytes() == b for p, b in original_bytes.items()))
            inventory = json.loads((root/'inventory.json').read_text())
            self.assertTrue(all(r['status'] == 'complete' for r in inventory))
            status = json.loads((root/'status.json').read_text())
            self.assertEqual(status['overall_job_completion_percent'], 100)
            self.assertEqual(status['remaining_seconds'], 0)
            self.assertFalse(any(r['incomplete'] for r in json.loads((root/'summary.json').read_text())['summaries']))

    def test_small_classical_interruption_and_resume_does_not_refit_saved_task(self):
        # Tiny real LR fits, independent of thesis data and output tree.
        rows = []
        for i in range(20):
            rows.append({'perfume_id': str(i), 'Perfume': str(i), 'joy': i%2, 'NRC_joy': float(i%3)})
        youtube = pd.DataFrame(rows)
        manifest = youtube[['perfume_id','Perfume']].assign(fold_id=[i%5+1 for i in range(20)], source_group_id=[str(i) for i in range(20)], source_group_size=1)
        prepared = SimpleNamespace(youtube=youtube, synthetic=pd.DataFrame(columns=["perfume_id"]), manifest=manifest, labels=('joy',), feature_columns=('NRC_joy',))
        job = runner.Job('main','logistic_regression',runner.CONTROL,'full_hybrid',123)
        class Pause:
            calls = 0
            def update(self, **kw): pass
            def requested(self):
                self.calls += 1
                return self.calls >= 2
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            expected, _ = runner.train_classical(job, prepared, ['NRC_joy'])
            with self.assertRaises(runner.Paused):
                runner.train_classical(job, prepared, ['NRC_joy'], root=root, contract={'seed':123}, progress=Pause())
            checkpoint=root/'fit_fold1_joy.json'; before=checkpoint.read_bytes()
            module=runner.model_module('logistic_regression')
            original=module.MODEL_SPEC.build_model
            with mock.patch.object(module, 'MODEL_SPEC', replace(module.MODEL_SPEC, build_model=mock.Mock(side_effect=original))) as spec:
                actual, _ = runner.train_classical(job, prepared, ['NRC_joy'], root=root, contract={'seed':123})
                self.assertEqual(spec.build_model.call_count,4)
            pd.testing.assert_frame_equal(expected,actual)
            self.assertEqual(checkpoint.read_bytes(),before)

    def test_heartbeat_percentage_eta_and_pause_file(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('builtins.print'):
            root=Path(tmp);jobs=[runner.Job('main','bert',runner.CONTROL,'full_hybrid',s) for s in [42,123,2026]]
            p=runner.Progress(root,jobs)
            p.update(completed_jobs=1, model='bert',condition=runner.CONTROL,seed=123,fold=2,epoch=1)
            status=json.loads((root/'status.json').read_text())
            self.assertEqual(status['remaining_seconds'],'estimating')
            self.assertAlmostEqual(status['overall_job_completion_percent'],100/3)
            self.assertFalse(status['job_percentage_is_elapsed_time_percentage'])
            p.observed('bert/youtube_only/full_hybrid',10);p.observed('bert/youtube_only/full_hybrid',20)
            p.emit();status=json.loads((root/'status.json').read_text());self.assertEqual(status['remaining_seconds'],30)
            (root/'PAUSE').touch();self.assertTrue(p.requested())


class TinyDeepResumeTests(unittest.TestCase):
    def test_both_deep_adapters_pause_at_epoch_and_resume_exactly_with_new_seed(self):
        from contextlib import ExitStack
        from src.training.tests.test_run_source_grouped_weighted_synthetic_bert import (
            _tiny_prepared, TinySequenceClassifier, _assert_nested_equal,
        )
        from src.training.run_bilstm_attention_benchmark import load_bilstm_settings
        for name, condition in [(m, c) for m in ['bert', 'bilstm_attention'] for c in [runner.CONTROL, runner.AUGMENTED]]:
            with self.subTest(model=name, condition=condition), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                prepared, member, tokenized = _tiny_prepared()
                # Use the real condition/split helpers: mocking membership hid
                # the original augmented-only BERT adapter failure.
                perfumes = prepared.perfume_df.copy()
                perfumes['fold_id'] = np.where(perfumes.perfume_id.isin(member['test_ids']), 1, 2)
                inner = perfumes.loc[~perfumes.perfume_id.isin(member['test_ids']),
                                     ['perfume_id', 'source_group_id']].copy()
                inner['outer_fold_id'] = 1
                inner['inner_role'] = np.where(inner.perfume_id.isin(member['train_ids']), 'train', 'val')
                prepared = replace(prepared, perfume_df=perfumes, inner_manifest=inner)
                module = runner.weighted_module(name)
                if name == 'bilstm_attention':
                    settings = replace(load_bilstm_settings({}), min_token_freq=1, max_vocab_size=30,
                                       max_tokens=8, embedding_dim=4, hidden_size=4, attention_dim=4,
                                       train_batch_size=2, eval_batch_size=2, max_epochs=3, patience=99)
                    prepared = replace(prepared, settings=settings)
                else:
                    stack.enter_context(mock.patch.object(module, '_load_model', side_effect=lambda s,l,d: TinySequenceClassifier(len(l)).to(d)))
                    stack.enter_context(mock.patch.object(module, 'load_or_build_token_cache', return_value=(tokenized,'tiny')))
                frame = prepared.perfume_df.loc[prepared.perfume_df.perfume_id.isin(member['test_ids']),
                    ['perfume_id','Perfume','source_group_id','source_group_size','fold_id']].copy()
                for label in prepared.labels:
                    frame[f'true_{label}'] = prepared.perfume_df.set_index('perfume_id').loc[frame.perfume_id,label].to_numpy()
                    frame[f'prob_{label}'] = np.nan;frame[f'pred_{label}'] = 0
                stack.enter_context(mock.patch.object(module,'canonical_oof_frames',return_value={condition:frame}))
                job=runner.Job('main',name,condition,'standard',123)
                contract={'seed':123,'tiny':True};root=Path(tmp)
                full=root/'full';full.mkdir();paused=root/'paused';paused.mkdir()
                expected,_=runner.train_deep(job,prepared,full,contract)
                class Pause:
                    pause=False
                    def update(self,**values):
                        if values.get('last_completed_epoch')==1:self.pause=True
                    def requested(self):return self.pause
                with self.assertRaises(runner.Paused):
                    runner.train_deep(job,prepared,paused,contract,progress=Pause())
                checkpoint=module._load_epoch_checkpoint(paused/'epoch_fold1.pt',runner.digest({'contract':contract,'fold':1}))
                self.assertEqual(checkpoint['next_epoch'],2)
                actual,_=runner.train_deep(job,prepared,paused,contract)
                pd.testing.assert_frame_equal(actual,expected,check_exact=True)
                full_state=module._load_epoch_checkpoint(full/'epoch_fold1.pt',runner.digest({'contract':contract,'fold':1}))
                resumed_state=module._load_epoch_checkpoint(paused/'epoch_fold1.pt',runner.digest({'contract':contract,'fold':1}))
                for key in ['model_state','optimizer_state','best_state','best_epoch','stale_epochs','epoch_history',
                            'loader_generator_state','torch_rng_state','python_random_state','numpy_random_state']:
                    _assert_nested_equal(self,full_state[key],resumed_state[key])
                if name=='bert':_assert_nested_equal(self,full_state['scheduler_state'],resumed_state['scheduler_state'])
                # The explicit seed reaches initialization and train shuffling.
                other=root/'other';other.mkdir()
                changed,_=runner.train_deep(replace(job,model_seed=2026),prepared,other,{'seed':2026})
                columns=[f'prob_{l}' for l in prepared.labels]
                self.assertFalse(np.array_equal(actual[columns].to_numpy(),changed[columns].to_numpy()))

    def test_unfinished_epoch_repeats_from_last_saved_epoch(self):
        from src.training.tests.test_run_source_grouped_weighted_synthetic_bert import _tiny_prepared, TinySequenceClassifier
        module=runner.weighted_module('bert')
        prepared,membership,tokenized=_tiny_prepared()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(module,'_load_model',side_effect=lambda s,l,d: TinySequenceClassifier(len(l)).to(d)):
            root=Path(tmp)
            args=dict(task=module.FoldTask(runner.AUGMENTED,1),membership=membership,tokenized_inputs=tokenized,
                      fold_fp='unfinished-epoch-test',epoch_callback=None,pause_requested=lambda:False,model_seed=123)
            expected,_=module.train_fold(prepared,epoch_path=root/'full.pt',**args)
            original=torch.optim.AdamW.step
            steps={'count':0}
            def interrupted_step(optimizer,*a,**kw):
                result=original(optimizer,*a,**kw)
                steps['count']+=1
                if steps['count']==5:
                    raise RuntimeError('simulated loss of process during epoch two')
                return result
            with mock.patch.object(torch.optim.AdamW,'step',interrupted_step):
                with self.assertRaisesRegex(RuntimeError,'epoch two'):
                    module.train_fold(prepared,epoch_path=root/'interrupted.pt',**args)
            saved=module._load_epoch_checkpoint(root/'interrupted.pt','unfinished-epoch-test')
            self.assertEqual(saved['next_epoch'],2)
            self.assertEqual(len(saved['epoch_history']),1)
            actual,summary=module.train_fold(prepared,epoch_path=root/'interrupted.pt',**args)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual([r['epoch'] for r in summary['epoch_history']],[1,2,3])


class NativeStartupTests(unittest.TestCase):
    def test_fresh_process_load_order_and_real_bert_training_batch(self):
        """Do not inherit the test process's already-loaded PyTorch runtime."""
        import subprocess
        import sys
        code = '''
import tempfile
from pathlib import Path
from src.training import run_thesis_three_seed as r
assert 'torch' in __import__('sys').modules
import torch
with tempfile.TemporaryDirectory() as tmp:
    cfg = r.read_yaml(r.DEFAULT_CONFIG)
    prepared = {name: r.prepare_model(name, Path(tmp)) for name in cfg['models']}
    bert = prepared['bert']
    module = r.weighted_module('bert')
    module.set_global_seed(123)
    model = module._load_model(bert.settings, bert.labels, torch.device('cpu'))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=bert.settings.learning_rate)
    ids = torch.tensor([[101, 2023, 2003, 102], [101, 2023, 2003, 102]])
    labels = torch.tensor([[0.,1.,0.,1.,0.,1.,0.], [1.,0.,1.,0.,1.,0.,1.]])
    loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    print('native_bert_batch_passed')
'''
        result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', code],
                                cwd=runner.PROJECT_ROOT, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('native_bert_batch_passed', result.stdout)
