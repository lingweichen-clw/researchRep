# Graph WaveNet Source Audit

## Source

The supplied source is `D:/projects/researchProjects/Graph-WaveNet`, corresponding to *Graph WaveNet for Deep Spatial-Temporal Graph Modeling* (IJCAI 2019). The project uses the source for architectural reference only; its original data loader and checkpoint layout are not part of the STAnchor training protocol.

## Interface mapping

STAnchor uses traffic tensors with layout `[B, T, N, C]`, where `B` is batch size, `T=12` is the observed context, `N=207` is the sensor count, and `C=1` is the traffic channel. Graph WaveNet expects `[B, C, N, T]`. The adapter must therefore perform:

```text
[B, T, N, C] -> permute(0, 3, 2, 1) -> [B, C, N, T]
```

The official model returns `[B, H, N, 1]` when `out_dim=H`; this already matches the STAnchor downstream layout `[B, H, N, C]` for `C=1`.

## Architectural components

- gated temporal convolutions with exponentially increasing dilation;
- graph convolution over fixed supports;
- learned adaptive adjacency from `nodevec1 @ nodevec2`;
- residual, skip, and batch-normalization paths;
- a final `1x1` projection to the forecast horizon.

## Source compatibility issue

The supplied `Graph-WaveNet/model.py` declares `gate_convs`, `residual_convs`, and `skip_convs` as `nn.Conv1d` while passing four-dimensional tensors `[B, C, N, T]`. A direct smoke forward fails with:

```text
RuntimeError: Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size [B, C, N, T]
```

The adapter must use `nn.Conv2d` with kernels `(1, kernel_size)` and `(1, 1)` for these paths. This is an interface correction required for execution; it does not change the Graph WaveNet computation intended by the paper.

## Training protocol boundary

The STAnchor adapter should preserve the existing protocol:

- use the project data split and train-only scaler;
- keep the v3 retrieval encoder frozen;
- use the same Bank and candidate protocol;
- train Graph WaveNet as the downstream backbone;
- compare base-only, retrieval fusion, and error-aware/PostHoc variants under the same seed and horizon metrics.

The original `Graph-WaveNet/train.py` data loader, padding convention, and scalar normalization must not silently replace the project pipeline.
