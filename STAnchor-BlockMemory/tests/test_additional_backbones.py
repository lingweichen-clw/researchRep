from __future__ import annotations

import unittest

import numpy as np
import torch

from stanchor.config import (
    POSTHOC_FROZEN_BASE,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TargetConfig,
    load_config,
)
from stanchor.data.dataset import TrafficSeries, build_normalized_weekday_slot_mean_table
from stanchor.data.graph import GraphData, graph_from_dense
from stanchor.data.normalization import NodeStandardScaler
from stanchor.engine.target import build_downstream_model, configure_error_aware_stage
from stanchor.losses.downstream import compute_downstream_loss, masked_mae, masked_mse
from stanchor.models.baseline import DCRNNForecastBackbone, DLinearForecastBackbone, STNormForecastBackbone, STSSDLForecastBackbone
from stanchor.models.downstream import DownstreamOutput
from stanchor.modes import BASE_ONLY, LEARNED_TOPK_ERROR_AWARE
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates


def _test_graph(num_nodes: int) -> GraphData:
    nodes = torch.arange(num_nodes, dtype=torch.long)
    return GraphData(
        edge_index=torch.stack((nodes, nodes), dim=0),
        edge_weight=torch.ones(num_nodes),
        num_nodes=num_nodes,
        fingerprint="test-graph",
    )


def _zeros_output(shape: tuple[int, ...]) -> DownstreamOutput:
    zeros = torch.zeros(*shape)
    return DownstreamOutput(
        base_prediction=zeros,
        memory_prediction=zeros,
        confidence_features=zeros,
        confidence=zeros,
        fusion_weight=zeros,
        final_prediction=zeros,
        memory_valid=torch.ones_like(zeros, dtype=torch.bool),
    )


class AdditionalBackboneContractTest(unittest.TestCase):
    def _config(self, backbone_name: str) -> ExperimentConfig:
        return ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            model=ModelConfig(input_channels=1, output_channels=1),
            target=TargetConfig(
                backbone_name=backbone_name,
                downstream_mode=BASE_ONLY,
            ),
        )

    def test_st_norm_forward_shape_and_gradient(self) -> None:
        config = self._config("st_norm")
        config.validate()
        model = build_downstream_model(config, _test_graph(5))
        self.assertIsInstance(model.backbone, STNormForecastBackbone)
        output = model.backbone(torch.randn(2, 12, 5, 1))
        self.assertEqual(tuple(output.shape), (2, 12, 5, 1))
        self.assertTrue(bool(torch.isfinite(output).all()))
        output.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.backbone.parameters()))

    def test_dlinear_forward_shape_and_gradient(self) -> None:
        config = self._config("dlinear")
        config.validate()
        model = build_downstream_model(config, _test_graph(5))
        self.assertIsInstance(model.backbone, DLinearForecastBackbone)
        output = model.backbone(torch.randn(2, 12, 5, 1))
        self.assertEqual(tuple(output.shape), (2, 12, 5, 1))
        self.assertTrue(bool(torch.isfinite(output).all()))
        output.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.backbone.parameters()))

    def test_dlinear_does_not_use_reciprocal_length_init(self) -> None:
        model = DLinearForecastBackbone(12, 12, 1, 1, moving_avg_kernel=25)
        expected = torch.full_like(model.seasonal.weight, 1.0 / 12.0)
        self.assertFalse(torch.allclose(model.seasonal.weight, expected))
        self.assertFalse(torch.allclose(model.trend.weight, expected))

    def test_st_ssdl_forward_shape_gradient_and_no_label_leakage(self) -> None:
        config = self._config("st_ssdl")
        config.validate()
        model = build_downstream_model(config, _test_graph(5))
        self.assertIsInstance(model.backbone, STSSDLForecastBackbone)
        x = torch.randn(2, 12, 5, 1)
        tod = torch.arange(12).repeat(2, 1)
        x_his = torch.randn(2, 12, 5, 1)
        y_tod = (tod[:, -1:] + torch.arange(1, 13)) % 288
        model.backbone.eval()
        with torch.no_grad():
            first = model.backbone(x, tod=tod, x_his=x_his, y_tod=y_tod, labels=None)
            second = model.backbone(x, tod=tod, x_his=x_his, y_tod=y_tod, labels=torch.randn(2, 12, 5, 1))
        self.assertEqual(tuple(first.shape), (2, 12, 5, 1))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(torch.equal(first, second))
        model.backbone.train()
        output = model.backbone(x, tod=tod, x_his=x_his, y_tod=y_tod, labels=None)
        output.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.backbone.parameters()))

    def test_st_ssdl_symadj_matches_official_and_preserves_self_loops(self) -> None:
        config = self._config("st_ssdl")
        graph = graph_from_dense(
            np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
            add_self_loops=False,
        )
        model = build_downstream_model(config, graph)
        expected = torch.tensor([[1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0 / 3.0]])
        self.assertTrue(torch.allclose(model.backbone.adj_supports[0], expected, atol=1.0e-6))

    def test_st_ssdl_dynamic_support_uses_official_sigmoid_activation(self) -> None:
        model = build_downstream_model(self._config("st_ssdl"), _test_graph(2))
        embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        similarity = torch.einsum("bnc,bmc->bnm", embeddings, embeddings)
        expected = torch.softmax(torch.sigmoid(similarity), dim=-1)
        actual = model.backbone._dynamic_support(embeddings)
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-7))

    def test_dcrnn_forward_shape_gradient_and_no_label_leak(self) -> None:
        config = self._config("dcrnn")
        config.validate()
        model = build_downstream_model(config, _test_graph(5))
        self.assertIsInstance(model.backbone, DCRNNForecastBackbone)
        x = torch.randn(2, 12, 5, 1)
        tod = torch.arange(12).repeat(2, 1)
        model.backbone.eval()
        with torch.no_grad():
            first = model.backbone(x, tod=tod, labels=None)
            second = model.backbone(x, tod=tod, labels=torch.randn(2, 12, 5, 1))
        self.assertEqual(tuple(first.shape), (2, 12, 5, 1))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertTrue(torch.equal(first, second))
        model.backbone.train()
        output = model.backbone(x, tod=tod, labels=None)
        output.mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.backbone.parameters()))

    def test_each_new_base_completes_an_optimizer_step(self) -> None:
        torch.manual_seed(17)
        for backbone_name in ("st_norm", "dlinear", "st_ssdl", "dcrnn"):
            model = build_downstream_model(self._config(backbone_name), _test_graph(3))
            model.backbone.train()
            optimizer = torch.optim.Adam(model.backbone.parameters(), lr=1.0e-3)
            x = torch.randn(2, 12, 3, 1)
            labels = torch.randn(2, 12, 3, 1)
            before = {
                name: parameter.detach().clone()
                for name, parameter in model.backbone.named_parameters()
            }
            optimizer.zero_grad(set_to_none=True)
            if backbone_name == "st_ssdl":
                tod = torch.arange(12).repeat(2, 1)
                output = model.backbone(
                    x,
                    tod=tod,
                    x_his=torch.randn_like(x),
                    y_tod=(tod[:, -1:] + torch.arange(1, 13)) % 288,
                    labels=labels,
                )
                auxiliary = model.backbone.pop_auxiliary_losses()
                self.assertIsNotNone(auxiliary)
                loss = output.square().mean() + auxiliary["contrastive"] + auxiliary["deviation"]
            elif backbone_name == "dcrnn":
                output = model.backbone(x, tod=torch.arange(12).repeat(2, 1), labels=labels)
                loss = output.square().mean()
            else:
                output = model.backbone(x)
                loss = output.square().mean()
            loss.backward()
            optimizer.step()
            self.assertTrue(
                any(
                    not torch.equal(parameter.detach(), before[name])
                    for name, parameter in model.backbone.named_parameters()
                ),
                backbone_name,
            )

    def test_weekday_slot_mean_table_ignores_future_and_uses_train_only(self) -> None:
        values = torch.zeros(20, 3, 1)
        values[:10, 0, 0] = 10.0
        values[10:, 0, 0] = 99.0
        observed = torch.ones(20, 3, 1, dtype=torch.bool)
        weekday = torch.zeros(20, dtype=torch.long)
        slot = torch.zeros(20, dtype=torch.long)
        series = TrafficSeries(
            values=values.numpy(),
            observed=observed.numpy(),
            timestamps_ns=torch.arange(20).numpy(),
            weekday=weekday.numpy(),
            slot=slot.numpy(),
            slots_per_day=288,
        )
        scaler = NodeStandardScaler.fit(series.values[:10], series.observed[:10])
        table = build_normalized_weekday_slot_mean_table(series, train_end=10, scaler=scaler)
        self.assertEqual(table.shape, (3, 7 * 288))
        physical = scaler.mean[0, 0]
        self.assertAlmostEqual(float(table[0, 0] * (scaler.std[0, 0] + scaler.eps) + scaler.mean[0, 0]), float(physical), places=4)

    def test_router_factory_keeps_error_corrector_for_new_backbones(self) -> None:
        for backbone_name in ("st_norm", "dlinear", "st_ssdl", "dcrnn"):
            config = ExperimentConfig(
                data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
                model=ModelConfig(input_channels=1, output_channels=1),
                target=TargetConfig(
                    backbone_name=backbone_name,
                    downstream_mode=LEARNED_TOPK_ERROR_AWARE,
                    training_protocol=POSTHOC_FROZEN_BASE,
                    validation_correction_variant="base_as_candidate",
                    calibrator_arch="retrieval_aware_mha_router",
                    base_warmup_epochs=0,
                    calibrator_warmup_epochs=0,
                ),
            )
            config.validate()
            model = build_downstream_model(config, _test_graph(5))
            self.assertIsNotNone(model.error_corrector)

    def test_posthoc_st_norm_backbone_stays_eval_and_deterministic(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            model=ModelConfig(input_channels=1, output_channels=1),
            target=TargetConfig(
                backbone_name="st_norm",
                downstream_mode=LEARNED_TOPK_ERROR_AWARE,
                training_protocol=POSTHOC_FROZEN_BASE,
                validation_correction_variant="base_as_candidate",
                calibrator_arch="retrieval_aware_mha_router",
                base_warmup_epochs=0,
                calibrator_warmup_epochs=0,
            ),
        )
        model = build_downstream_model(config, _test_graph(5))
        configure_error_aware_stage(model, "posthoc_calibrator")
        model.train(True)
        self.assertFalse(model.backbone.training)
        before = {
            name: value.detach().clone()
            for name, value in model.backbone.named_buffers()
            if "running_" in name
        }
        x = torch.randn(2, 12, 5, 1)
        with torch.no_grad():
            first = model.backbone(x)
            second = model.backbone(x)
        self.assertTrue(torch.equal(first, second))
        for name, value in model.backbone.named_buffers():
            if name in before:
                self.assertTrue(torch.equal(value, before[name]))

    def test_posthoc_router_step_does_not_change_any_new_backbone_state(self) -> None:
        torch.manual_seed(23)
        batch, nodes, top_k = 2, 3, 2
        for backbone_name in ("st_norm", "dlinear", "st_ssdl", "dcrnn"):
            config = ExperimentConfig(
                data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
                model=ModelConfig(input_channels=1, output_channels=1),
                target=TargetConfig(
                    backbone_name=backbone_name,
                    downstream_mode=LEARNED_TOPK_ERROR_AWARE,
                    training_protocol=POSTHOC_FROZEN_BASE,
                    validation_correction_variant="base_as_candidate",
                    calibrator_arch="retrieval_aware_mha_router",
                    base_warmup_epochs=0,
                    calibrator_warmup_epochs=0,
                ),
            )
            model = build_downstream_model(config, _test_graph(nodes))
            groups = configure_error_aware_stage(model, "posthoc_calibrator")
            optimizer = torch.optim.Adam(groups, lr=1.0e-3)
            model.train(True)
            before = {
                name: value.detach().clone()
                for name, value in model.backbone.state_dict().items()
            }
            history = torch.randn(batch, 12, nodes, 1)
            base = torch.randn(batch, 12, nodes, 1)
            futures = torch.randn(batch, 12, nodes, top_k, 1)
            masks = torch.ones_like(futures, dtype=torch.bool)
            candidates = NodeCandidates(
                event_ids=torch.arange(batch * nodes * top_k).reshape(batch, nodes, top_k),
                total_scores=torch.randn(batch, nodes, top_k),
                shape_scores=torch.randn(batch, nodes, top_k),
                level_distances=torch.rand(batch, nodes, top_k),
                weights=torch.full((batch, nodes, top_k), 1.0 / top_k),
                valid=torch.ones(batch, nodes, top_k, dtype=torch.bool),
            )
            aggregation = AggregationOutput(
                prediction=futures.mean(dim=3),
                variance=futures.var(dim=3, unbiased=False),
                valid=torch.ones(batch, 12, nodes, 1, dtype=torch.bool),
                candidate_futures=futures,
                candidate_masks=masks,
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(
                history,
                candidates,
                aggregation,
                base_override=base,
                retrieval_node_keys=torch.randn(batch, nodes, 64),
            )
            output.final_prediction.square().mean().backward()
            optimizer.step()
            self.assertFalse(model.backbone.training)
            for name, value in model.backbone.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]), f"{backbone_name}.{name}")

    def test_backbone_state_dict_round_trip_is_deterministic(self) -> None:
        for backbone_name in ("st_norm", "dlinear"):
            config = self._config(backbone_name)
            model_a = build_downstream_model(config, _test_graph(5)).eval()
            model_b = build_downstream_model(config, _test_graph(5)).eval()
            model_b.backbone.load_state_dict(model_a.backbone.state_dict(), strict=True)
            x = torch.randn(2, 12, 5, 1)
            with torch.no_grad():
                first = model_a.backbone(x)
                second = model_b.backbone(x)
            self.assertTrue(torch.equal(first, second))

    def test_st_ssdl_state_dict_round_trip_is_deterministic(self) -> None:
        config = self._config("st_ssdl")
        model_a = build_downstream_model(config, _test_graph(5)).eval()
        model_b = build_downstream_model(config, _test_graph(5)).eval()
        model_b.backbone.load_state_dict(model_a.backbone.state_dict(), strict=True)
        x = torch.randn(2, 12, 5, 1)
        tod = torch.arange(12).repeat(2, 1)
        x_his = torch.randn(2, 12, 5, 1)
        y_tod = (tod[:, -1:] + torch.arange(1, 13)) % 288
        with torch.no_grad():
            first = model_a.backbone(x, tod=tod, x_his=x_his, y_tod=y_tod)
            second = model_b.backbone(x, tod=tod, x_his=x_his, y_tod=y_tod)
        self.assertTrue(torch.equal(first, second))

    def test_dcrnn_state_dict_round_trip_is_deterministic(self) -> None:
        config = self._config("dcrnn")
        model_a = build_downstream_model(config, _test_graph(5)).eval()
        model_b = build_downstream_model(config, _test_graph(5)).eval()
        model_b.backbone.load_state_dict(model_a.backbone.state_dict(), strict=True)
        x = torch.randn(2, 12, 5, 1)
        tod = torch.arange(12).repeat(2, 1)
        with torch.no_grad():
            first = model_a.backbone(x, tod=tod)
            second = model_b.backbone(x, tod=tod)
        self.assertTrue(torch.equal(first, second))

    def test_forecast_loss_name_switches_mae_and_mse(self) -> None:
        prediction = torch.tensor([[[[2.0], [4.0]]]])
        target = torch.zeros_like(prediction)
        observed = torch.ones_like(prediction, dtype=torch.bool)
        self.assertAlmostEqual(float(masked_mae(prediction, target, observed)), 3.0, places=6)
        self.assertAlmostEqual(float(masked_mse(prediction, target, observed)), 10.0, places=6)
        output = _zeros_output((1, 1, 2, 1))
        mae = compute_downstream_loss(
            output,
            target=target,
            observed=observed,
            confidence_weight=0.0,
            help_margin=0.0,
            help_temperature=0.1,
            use_confidence=False,
            forecast_prediction=prediction,
            forecast_target=target,
            forecast_loss_name="mae",
        )
        mse = compute_downstream_loss(
            output,
            target=target,
            observed=observed,
            confidence_weight=0.0,
            help_margin=0.0,
            help_temperature=0.1,
            use_confidence=False,
            forecast_prediction=prediction,
            forecast_target=target,
            forecast_loss_name="mse",
        )
        self.assertAlmostEqual(float(mae.forecast), 3.0, places=6)
        self.assertAlmostEqual(float(mse.forecast), 10.0, places=6)

    def test_formal_configs_keep_protocol_split(self) -> None:
        dlinear = load_config("configs/formal_baseonly_dlinear.yaml")
        self.assertEqual(dlinear.target.forecast_loss_space, "normalized")
        self.assertEqual(dlinear.target.forecast_loss_name, "mse")
        self.assertEqual(dlinear.target.epochs, 100)
        self.assertAlmostEqual(dlinear.target.learning_rate, 0.0001)
        self.assertTrue(dlinear.target.early_stopping_enabled)
        self.assertEqual(dlinear.target.patience, 3)

        st_norm = load_config("configs/formal_baseonly_st_norm.yaml")
        self.assertEqual(st_norm.target.forecast_loss_space, "normalized")
        self.assertEqual(st_norm.target.forecast_loss_name, "mse")
        self.assertEqual(st_norm.target.epochs, 100)
        self.assertEqual(st_norm.target.batch_size, 8)
        self.assertAlmostEqual(st_norm.target.learning_rate, 0.0001)
        self.assertAlmostEqual(st_norm.target.weight_decay, 0.0)
        self.assertEqual(st_norm.target.scheduler_name, "none")
        self.assertTrue(st_norm.target.early_stopping_enabled)
        self.assertEqual(st_norm.target.patience, 50)

        ssdl = load_config("configs/formal_baseonly_st_ssdl.yaml")
        self.assertEqual(ssdl.target.backbone_name, "st_ssdl")
        self.assertEqual(ssdl.target.forecast_loss_space, "physical")
        self.assertEqual(ssdl.target.forecast_loss_name, "mae")
        self.assertEqual(ssdl.target.scheduler_name, "multi_step_lr")
        self.assertEqual(tuple(ssdl.target.scheduler_milestones), (50, 70))
        self.assertAlmostEqual(ssdl.target.optimizer_eps, 0.001)
        self.assertEqual(ssdl.target.batch_size, 128)
        self.assertEqual(ssdl.target.epochs, 100)
        self.assertEqual(ssdl.target.patience, 30)

        dcrnn = load_config("configs/formal_baseonly_dcrnn.yaml")
        self.assertEqual(dcrnn.target.backbone_name, "dcrnn")
        self.assertEqual(dcrnn.target.forecast_loss_space, "physical")
        self.assertEqual(dcrnn.target.forecast_loss_name, "mae")
        self.assertEqual(dcrnn.target.scheduler_name, "multi_step_lr")
        self.assertEqual(tuple(dcrnn.target.scheduler_milestones), (20, 30, 40, 50))
        self.assertAlmostEqual(dcrnn.target.optimizer_eps, 0.001)
        self.assertAlmostEqual(dcrnn.target.learning_rate, 0.01)
        self.assertEqual(dcrnn.target.patience, 50)
        self.assertEqual(dcrnn.target.batch_size, 64)
        self.assertEqual(dcrnn.target.dcrnn_rnn_units, 64)
        self.assertEqual(dcrnn.target.dcrnn_rnn_layers, 2)
        self.assertEqual(dcrnn.target.dcrnn_max_diffusion_step, 2)
        self.assertEqual(dcrnn.target.dcrnn_filter_type, "dual_random_walk")
        self.assertTrue(dcrnn.target.dcrnn_use_curriculum_learning)
        self.assertEqual(dcrnn.target.dcrnn_cl_decay_steps, 2000)
        self.assertTrue(dcrnn.target.dcrnn_use_time_in_day)

        for name in (
            "formal_base_as_candidate_st_norm.yaml",
            "formal_base_as_candidate_dlinear.yaml",
            "formal_base_as_candidate_st_ssdl.yaml",
            "formal_base_as_candidate_dcrnn.yaml",
        ):
            config = load_config(f"configs/{name}")
            config.validate()
            self.assertEqual(config.target.forecast_loss_space, "physical")
            self.assertEqual(config.target.forecast_loss_name, "mae")
            self.assertEqual(config.target.epochs, 50)
            self.assertAlmostEqual(config.target.learning_rate, 0.0005)

    def test_formal_configs_load_with_paper_backbone_settings(self) -> None:
        for name, expected in (
            ("formal_baseonly_st_norm.yaml", (16, 1, 4)),
            ("formal_base_as_candidate_st_norm.yaml", (16, 1, 4)),
            ("formal_baseonly_dlinear.yaml", (25, False)),
            ("formal_base_as_candidate_dlinear.yaml", (25, False)),
        ):
            config = load_config(f"configs/{name}")
            config.validate()
            if config.target.backbone_name == "st_norm":
                self.assertEqual(
                    (config.target.st_norm_channels, config.target.st_norm_blocks, config.target.st_norm_layers),
                    expected,
                )
            else:
                self.assertEqual(
                    (config.target.dlinear_moving_avg_kernel, config.target.dlinear_individual),
                    expected,
                )

    def test_queue_configs_keep_dataset_paths_and_router_protocol(self) -> None:
        for backbone in ("st_norm", "dlinear", "st_ssdl", "dcrnn"):
            pems_base = load_config(f"configs/cross_dataset_pemsbay_baseonly_{backbone}.yaml")
            pems_base.validate()
            self.assertEqual(pems_base.data.raw_path.replace("\\", "/"), "../data/pemsBay_data/pems-bay.h5")
            self.assertEqual(pems_base.target.backbone_name, backbone)
            self.assertEqual(pems_base.target.downstream_mode, BASE_ONLY)

            random_cfg = load_config(f"configs/ablation_random_bank_router_{backbone}.yaml")
            random_cfg.validate()
            self.assertIn("random_seed42", random_cfg.bank.output_dir)
            self.assertEqual(random_cfg.bank.event_top_r, 32)
            self.assertEqual(random_cfg.target.forecast_loss_space, "physical")
            self.assertEqual(random_cfg.target.forecast_loss_name, "mae")
            self.assertAlmostEqual(random_cfg.target.learning_rate, 0.0005)

            pems_router = load_config(f"configs/cross_dataset_pemsbay_router_{backbone}.yaml")
            pems_router.validate()
            self.assertEqual(pems_router.data.raw_path.replace("\\", "/"), "../data/pemsBay_data/pems-bay.h5")
            self.assertEqual(pems_router.bank.event_top_r, 96)
            self.assertEqual(pems_router.target.forecast_loss_space, "physical")
            self.assertEqual(pems_router.target.forecast_loss_name, "mae")
            self.assertAlmostEqual(pems_router.target.learning_rate, 0.0005)
            self.assertEqual(pems_router.target.calibrator_arch, "retrieval_aware_mha_router")

        st_norm_base = load_config("configs/cross_dataset_pemsbay_baseonly_st_norm.yaml")
        self.assertEqual(st_norm_base.target.forecast_loss_space, "normalized")
        self.assertEqual(st_norm_base.target.forecast_loss_name, "mse")
        self.assertEqual(st_norm_base.target.batch_size, 8)
        self.assertEqual(st_norm_base.target.epochs, 100)
        self.assertAlmostEqual(st_norm_base.target.learning_rate, 0.0001)
        self.assertEqual(st_norm_base.target.scheduler_name, "none")
        self.assertTrue(st_norm_base.target.early_stopping_enabled)

        dlinear_base = load_config("configs/cross_dataset_pemsbay_baseonly_dlinear.yaml")
        self.assertEqual(dlinear_base.target.forecast_loss_space, "normalized")
        self.assertEqual(dlinear_base.target.forecast_loss_name, "mse")
        self.assertTrue(dlinear_base.target.early_stopping_enabled)

        ssdl_base = load_config("configs/cross_dataset_pemsbay_baseonly_st_ssdl.yaml")
        self.assertEqual(ssdl_base.target.scheduler_name, "multi_step_lr")
        self.assertAlmostEqual(ssdl_base.target.optimizer_eps, 0.001)
        self.assertEqual(ssdl_base.target.batch_size, 64)
        self.assertEqual(ssdl_base.target.epochs, 100)
        self.assertEqual(ssdl_base.target.patience, 30)
        self.assertEqual(tuple(ssdl_base.target.scheduler_milestones), (10, 70))
        self.assertEqual(ssdl_base.target.st_ssdl_cl_decay_steps, 8000)
        self.assertEqual(ssdl_base.target.st_ssdl_input_embedding_dim, 10)
        self.assertEqual(ssdl_base.target.st_ssdl_node_embedding_dim, 20)

        dcrnn_base = load_config("configs/cross_dataset_pemsbay_baseonly_dcrnn.yaml")
        self.assertEqual(dcrnn_base.target.backbone_name, "dcrnn")
        self.assertEqual(dcrnn_base.target.forecast_loss_space, "physical")
        self.assertEqual(dcrnn_base.target.forecast_loss_name, "mae")
        self.assertEqual(dcrnn_base.target.scheduler_name, "multi_step_lr")
        self.assertEqual(tuple(dcrnn_base.target.scheduler_milestones), (20, 30, 40, 50))
        self.assertAlmostEqual(dcrnn_base.target.optimizer_eps, 0.001)
        self.assertEqual(dcrnn_base.target.patience, 50)
        self.assertEqual(dcrnn_base.target.batch_size, 64)

    def test_router_configs_match_their_frozen_base_architecture(self) -> None:
        fields = {
            "st_norm": (
                "st_norm_channels", "st_norm_kernel_size", "st_norm_blocks",
                "st_norm_layers", "st_norm_use_snorm", "st_norm_use_tnorm",
                "st_norm_dropout",
            ),
            "dlinear": ("dlinear_moving_avg_kernel", "dlinear_individual"),
            "st_ssdl": (
                "st_ssdl_rnn_units", "st_ssdl_rnn_layers", "st_ssdl_cheb_k",
                "st_ssdl_prototype_num", "st_ssdl_prototype_dim",
                "st_ssdl_tod_embed_dim", "st_ssdl_node_embedding_dim",
                "st_ssdl_input_embedding_dim", "st_ssdl_adaptive_embedding_dim",
                "st_ssdl_use_ste", "st_ssdl_use_curriculum_learning",
                "st_ssdl_cl_decay_steps", "st_ssdl_triplet_margin",
            ),
            "dcrnn": (
                "dcrnn_rnn_units", "dcrnn_rnn_layers", "dcrnn_max_diffusion_step",
                "dcrnn_filter_type", "dcrnn_use_curriculum_learning",
                "dcrnn_cl_decay_steps", "dcrnn_use_time_in_day",
            ),
        }
        for backbone_name, names in fields.items():
            metrla_base = load_config(f"configs/formal_baseonly_{backbone_name}.yaml")
            metrla_router = load_config(f"configs/formal_base_as_candidate_{backbone_name}.yaml")
            metrla_random = load_config(f"configs/ablation_random_bank_router_{backbone_name}.yaml")
            pems_base = load_config(f"configs/cross_dataset_pemsbay_baseonly_{backbone_name}.yaml")
            pems_router = load_config(f"configs/cross_dataset_pemsbay_router_{backbone_name}.yaml")
            for field in names:
                self.assertEqual(
                    getattr(metrla_router.target, field),
                    getattr(metrla_base.target, field),
                    f"METR-LA {backbone_name}.{field}",
                )
                self.assertEqual(
                    getattr(metrla_random.target, field),
                    getattr(metrla_base.target, field),
                    f"METR-LA random {backbone_name}.{field}",
                )
                self.assertEqual(
                    getattr(pems_router.target, field),
                    getattr(pems_base.target, field),
                    f"PEMS-BAY {backbone_name}.{field}",
                )
