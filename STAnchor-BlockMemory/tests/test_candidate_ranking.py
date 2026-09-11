from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from stanchor.config import load_config, validate_candidate_ranking
from stanchor.engine.target import retrieve_for_downstream_mode
from stanchor.modes import LEARNED_TOPK_CONFIDENCE, LEARNED_TOPK_ERROR_AWARE


class CandidateRankingTest(unittest.TestCase):
    def test_validate_candidate_ranking(self) -> None:
        self.assertEqual(validate_candidate_ranking('learned_key'), 'learned_key')
        self.assertEqual(validate_candidate_ranking('raw_l1'), 'raw_l1')
        with self.assertRaises(ValueError):
            validate_candidate_ranking('event_key')

    def test_raw_l1_requires_error_aware_mode(self) -> None:
        with self.assertRaises(ValueError):
            retrieve_for_downstream_mode(
                LEARNED_TOPK_CONFIDENCE,
                pretrained=object(),
                retriever=object(),
                bank=object(),
                data=object(),
                graph=object(),
                batch={},
                x=None,
                observed_x=None,
                device=None,
                candidate_ranking='raw_l1',
            )

    def test_raw_l1_router_does_not_apply_offset_decay(self) -> None:
        pretrained = MagicMock()
        query_keys = torch.randn(1, 2, 4)
        pretrained.encode_clean.return_value = SimpleNamespace(
            retrieval=SimpleNamespace(node_keys=query_keys),
            statistics=SimpleNamespace(level_features=torch.zeros(1, 2, 1)),
        )
        retriever = MagicMock(event_top_r=3, node_top_k=2)
        retriever.context_window_cache = None
        candidates = MagicMock(name='raw_l1_candidates')
        direct_aggregation = MagicMock(name='direct_raw_future_aggregation')
        retriever.aggregate.return_value = direct_aggregation
        data = SimpleNamespace(
            train=SimpleNamespace(retrieval_context_length=288, context_length=12),
            series=object(),
            scaler=object(),
        )
        batch = {
            'retrieval_x': torch.zeros(1, 288, 2, 1),
            'retrieval_observed': torch.ones(1, 288, 2, 1, dtype=torch.bool),
            'retrieval_weekday': torch.zeros(1, 288, dtype=torch.long),
            'retrieval_slot': torch.zeros(1, 288, dtype=torch.long),
            'query_weekday': torch.zeros(1, dtype=torch.long),
            'query_slot': torch.zeros(1, dtype=torch.long),
            'context_start': torch.ones(1, dtype=torch.long),
        }
        with (
            patch('stanchor.engine.target.calendar_event_candidates', return_value=MagicMock()),
            patch('stanchor.engine.target.raw_l1_node_candidates', return_value=(candidates, None, None)),
            patch('stanchor.engine.target.offset_decay_aggregation') as offset_decay,
        ):
            result_candidates, aggregation, result_keys = retrieve_for_downstream_mode(
                LEARNED_TOPK_ERROR_AWARE,
                pretrained=pretrained,
                retriever=retriever,
                bank=SimpleNamespace(manifest=SimpleNamespace(retrieval_dim=4)),
                data=data,
                graph=object(),
                batch=batch,
                x=torch.zeros(1, 12, 2, 1),
                observed_x=torch.ones(1, 12, 2, 1, dtype=torch.bool),
                device=torch.device('cpu'),
                candidate_protocol='weekday_radius1_overlap',
                include_query_keys=True,
                candidate_ranking='raw_l1',
            )
        self.assertIs(result_candidates, candidates)
        self.assertIs(aggregation, direct_aggregation)
        self.assertTrue(torch.equal(result_keys, torch.zeros_like(query_keys)))
        pretrained.encode_clean.assert_not_called()
        retriever.aggregate.assert_called_once_with(candidates)
        offset_decay.assert_not_called()

    def test_baseonly_rejects_raw_l1_ranking(self) -> None:
        config = load_config('configs/formal_baseonly_st_norm.yaml')
        with self.assertRaises(ValueError):
            replace(config, target=replace(config.target, candidate_ranking='raw_l1')).validate()

    def test_rawl1_router_configs_keep_same_router_and_calendar_pool(self) -> None:
        pairs = (
            ('graph_wavenet', 'formal_base_as_candidate_gwn.yaml', 'ablation_rawl1_router_gwn.yaml'),
            ('argcn', 'formal_base_as_candidate_argcn.yaml', 'ablation_rawl1_router_argcn.yaml'),
        )
        for backbone, src_name, ablation_name in pairs:
            src = load_config('configs/' + src_name)
            ablation = load_config('configs/' + ablation_name)
            src.validate()
            ablation.validate()
            self.assertEqual(ablation.target.backbone_name, backbone)
            self.assertEqual(ablation.target.downstream_mode, LEARNED_TOPK_ERROR_AWARE)
            self.assertEqual(ablation.target.candidate_protocol, 'weekday_radius1_overlap')
            self.assertEqual(ablation.target.candidate_ranking, 'raw_l1')
            self.assertEqual(src.target.candidate_ranking, 'learned_key')
            self.assertEqual(ablation.bank.node_top_k, 12)
            self.assertEqual(ablation.bank.event_top_r, src.bank.event_top_r)
            self.assertEqual(ablation.target.calibrator_arch, 'retrieval_aware_mha_router')
            self.assertEqual(ablation.target.forecast_loss_space, 'physical')
            self.assertAlmostEqual(ablation.target.learning_rate, 0.0005)
            self.assertTrue(ablation.target.frozen_path_cache)
            self.assertNotIn('random_seed42', ablation.bank.output_dir)

    def test_local_retrieval_ablation_queue_uses_argcn_anchor(self) -> None:
        script = Path('scripts/run_metrla_retrieval_ablation_queue.ps1').read_text(
            encoding='utf-8'
        )
        self.assertIn("Label = 'random_router_argcn'", script)
        self.assertIn("Label = 'rawl1_router_argcn'", script)
        self.assertIn("Label = 'rawl1_router_gwn'", script)
        self.assertNotIn("Label = 'random_router_staeformer'", script)
        self.assertNotIn("Label = 'rawl1_router_staeformer'", script)

    def test_random_ablation_configs_keep_router_protocol(self) -> None:
        for backbone in ('staeformer', 'argcn'):
            config = load_config('configs/ablation_random_bank_router_' + backbone + '.yaml')
            config.validate()
            self.assertEqual(config.target.backbone_name, backbone)
            self.assertEqual(config.target.candidate_ranking, 'learned_key')
            self.assertIn('random_seed42', config.bank.output_dir)
            self.assertEqual(config.target.calibrator_arch, 'retrieval_aware_mha_router')


if __name__ == '__main__':
    unittest.main()
