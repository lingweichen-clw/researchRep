from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.run_metrla_simple_fusion_retrieval_queue import (
    build_run_specs,
    materialize_config,
    remove_temporary_bank,
)


class SimpleFusionRetrievalQueueTest(unittest.TestCase):
    def test_queue_crosses_three_backbones_with_two_missing_selectors(self) -> None:
        repo_root = Path('D:/repo')
        specs = build_run_specs(repo_root)

        self.assertEqual(len(specs), 6)
        self.assertEqual({spec.backbone for spec in specs}, {'gwn', 'stgcn', 'argcn'})
        self.assertEqual({spec.selector for spec in specs}, {'raw_l1', 'random'})
        self.assertEqual(len({spec.run_name for spec in specs}), 6)
        for spec in specs:
            self.assertIn('/thesis_only/simple_fusion_retrieval_comparison/', spec.run_name)
            if spec.selector == 'random':
                self.assertEqual(spec.candidate_ranking, 'learned_key')
                self.assertEqual(spec.pretrained_checkpoint.name, 'random_encoder_seed42.pt')
                self.assertEqual(spec.bank.name, 'metrla_random_simple_fusion')
            else:
                self.assertEqual(spec.candidate_ranking, 'raw_l1')
                self.assertEqual(spec.pretrained_checkpoint.name, 'pretrain_best.pt')
                self.assertEqual(spec.bank.name, 'metrla_source_simple_fusion')

    def test_materialized_config_changes_only_selector_and_artifact_routes(self) -> None:
        repo_root = Path.cwd()
        spec = next(
            value
            for value in build_run_specs(repo_root)
            if value.backbone == 'argcn' and value.selector == 'raw_l1'
        )
        source = yaml.safe_load(spec.source_config.read_text(encoding='utf-8'))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'resolved.yaml'
            materialize_config(spec, output)
            resolved = yaml.safe_load(output.read_text(encoding='utf-8'))

        self.assertEqual(resolved['target']['candidate_ranking'], 'raw_l1')
        self.assertEqual(resolved['target']['candidate_payload'], 'offset_only')
        self.assertEqual(resolved['target']['downstream_mode'], 'learned_topk_offset_only_horizon')
        self.assertEqual(resolved['target']['epochs'], 10)
        self.assertEqual(resolved['runtime']['run_name'], spec.run_name)
        self.assertEqual(resolved['bank']['output_dir'], str(spec.bank))
        for section in ('data', 'model', 'pretrain'):
            self.assertEqual(resolved[section], source[section])
        preserved_target = dict(resolved['target'])
        preserved_target['candidate_ranking'] = source['target']['candidate_ranking']
        preserved_target['epochs'] = source['target']['epochs']
        self.assertEqual(preserved_target, source['target'])

    def test_bank_cleanup_rejects_paths_outside_temporary_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary_root = root / 'temporary_banks'
            temporary_root.mkdir()
            outside = root / 'outside'
            outside.mkdir()
            with self.assertRaises(ValueError):
                remove_temporary_bank(outside, temporary_root)


if __name__ == '__main__':
    unittest.main()
