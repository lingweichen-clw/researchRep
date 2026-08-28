from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def quant(x, p):
    return float(np.nanquantile(x, p))


def robust_pair(a, b):
    scale = np.nanstd(np.concatenate([a, b])) + 1e-4
    return float(np.linalg.norm((a - b) / scale) / np.sqrt(a.size))


def cluster_plot(out, kind, rows, X, Y, keys, sid, max_rows=24):
    rows = rows[:max_rows]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rows)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    context_curves, future_curves = [], []
    key_points = []
    for idx, row in enumerate(rows):
        n, i, j = row['node'], row['i'], row['j']
        color = colors[idx]
        axes[0].plot(np.arange(X.shape[-1]), X[i, n], color=color, alpha=0.24, linewidth=1.0)
        axes[0].plot(np.arange(X.shape[-1]), X[j, n], color=color, alpha=0.24, linewidth=1.0, linestyle='--')
        axes[1].plot(np.arange(Y.shape[-1]), Y[i, n], color=color, alpha=0.24, linewidth=1.0)
        axes[1].plot(np.arange(Y.shape[-1]), Y[j, n], color=color, alpha=0.24, linewidth=1.0, linestyle='--')
        context_curves.extend([X[i, n], X[j, n]])
        future_curves.extend([Y[i, n], Y[j, n]])
        key_points.extend([keys[i, n], keys[j, n]])
    context_curves = np.asarray(context_curves)
    future_curves = np.asarray(future_curves)
    axes[0].plot(np.arange(context_curves.shape[1]), np.median(context_curves, axis=0), color='#111111', linewidth=2.8, label='cluster median')
    axes[1].plot(np.arange(future_curves.shape[1]), np.median(future_curves, axis=0), color='#111111', linewidth=2.8, label='cluster median')
    axes[0].set_title(f'{kind}: context ({len(rows)} pairs)')
    axes[1].set_title(f'{kind}: future ({len(rows)} pairs)')
    axes[0].set_xlabel('Context step (5 min)'); axes[1].set_xlabel('Future step (5 min)')
    axes[0].set_ylabel('Robust-normalized value'); axes[1].set_ylabel('Robust-normalized value')
    for ax, length in ((axes[0], X.shape[-1]), (axes[1], Y.shape[-1])):
        ax.set_xticks([0, length // 2, length - 1])
        ax.grid(alpha=0.22)
    points = np.asarray(key_points)
    # Plot only target cluster points; no full-bank background cloud.
    axes[2].scatter(points[:, 0], points[:, 1], c=np.repeat(colors, 2, axis=0), s=22, alpha=0.68, edgecolors='none')
    center = points.mean(axis=0)
    axes[2].scatter([center[0]], [center[1]], marker='X', s=100, color='#d62728', edgecolor='black', linewidth=0.7, label='cluster center')
    axes[2].set_title(f'{kind}: target key cluster (PCA)')
    axes[2].set_xlabel('Key PC1'); axes[2].set_ylabel('Key PC2'); axes[2].grid(alpha=0.22); axes[2].legend(frameon=False)
    fig.savefig(out / f'{kind}_cluster.png', dpi=220)
    plt.close(fig)
    return {'pairs_plotted': len(rows), 'context_curve_count': int(context_curves.shape[0]), 'future_curve_count': int(future_curves.shape[0]), 'key_points_plotted': int(points.shape[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--bank', required=True); ap.add_argument('--output-dir', required=True)
    ap.add_argument('--num-events', type=int, default=5000); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pairs-per-cluster', type=int, default=24)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    with pd.HDFStore(a.data, 'r') as store: values = store.get('/df').to_numpy(dtype=np.float32)
    bank = Path(a.bank); sid0 = np.load(bank / 'sample_id.npy').astype(np.int64)
    future = np.load(bank / 'future_values.npy', mmap_mode='r').astype(np.float32)
    keys = np.load(bank / 'node_keys.npy', mmap_mode='r').astype(np.float32)
    E, N, D = keys.shape; H = future.shape[1]
    take = np.arange(E) if E <= a.num_events else np.sort(rng.choice(E, a.num_events, replace=False))
    sid = sid0[take]; ok = (sid >= 12) & (sid < len(values)); take = take[ok]; sid = sid0[take]
    X = np.stack([values[x-11:x+1].T for x in sid], axis=0)
    med = np.nanmedian(values, axis=0); X = np.where(np.isfinite(X), X, med[None, :, None])
    Y = np.asarray(future[take, :, :, 0]).transpose(0, 2, 1)
    records = []
    for n in range(N):
        x, y, k = X[:, n, :], Y[:, n, :], np.asarray(keys[take, n, :])
        xz = (x - np.nanmedian(x, 0)) / (np.nanstd(x, 0) + 1e-4)
        yz = (y - np.nanmedian(y, 0)) / (np.nanstd(y, 0) + 1e-4)
        nn = NearestNeighbors(n_neighbors=min(25, len(xz))).fit(xz); _, inds = nn.kneighbors(xz)
        for i in range(len(xz)):
            for rnk in range(1, inds.shape[1]):
                j = int(inds[i, rnk])
                if i >= j: continue
                records.append({'node': n, 'i': i, 'j': j, 'context_distance': float(np.linalg.norm(xz[i]-xz[j]) / np.sqrt(12)), 'future_distance': float(np.linalg.norm(yz[i]-yz[j]) / np.sqrt(H)), 'key_distance': float(np.linalg.norm(k[i]-k[j]) / np.sqrt(D))})
    ctx = np.asarray([r['context_distance'] for r in records]); fut = np.asarray([r['future_distance'] for r in records]); kd = np.asarray([r['key_distance'] for r in records])
    alo, ahi, flo, fhi, klo, khi = quant(ctx,.08), quant(ctx,.92), quant(fut,.08), quant(fut,.92), quant(kd,.08), quant(kd,.92)
    A = [r for r in records if r['context_distance'] <= alo and r['future_distance'] >= fhi and r['key_distance'] >= khi]
    B = [r for r in records if r['context_distance'] >= ahi and r['future_distance'] <= flo and r['key_distance'] <= klo]
    A.sort(key=lambda r: (-r['future_distance'], -r['key_distance'], r['node'], r['i'], r['j']))
    B.sort(key=lambda r: (-r['context_distance'], -r['key_distance'], r['node'], r['i'], r['j']))
    chosen = {'context_similar_future_different': A[:a.pairs_per_cluster], 'context_different_future_similar': B[:a.pairs_per_cluster]}
    for typ, rows in chosen.items():
        for r in rows: r.update({'sample_i': int(sid[r['i']]), 'sample_j': int(sid[r['j']]), 'node_id': int(r['node'])})
    plots = {}
    for typ, rows in chosen.items():
        plots[typ] = cluster_plot(out, typ, rows, X, Y, np.asarray(keys[take]), sid, a.pairs_per_cluster)
    payload = {'schema_version': 2, 'selection': {'context_similar_future_different': 'context <= P8, future >= P92, key >= P92', 'context_different_future_similar': 'context >= P92, future <= P8, key <= P8', 'manual_selection': False, 'pairs_per_cluster': a.pairs_per_cluster}, 'num_events_used': int(len(take)), 'nodes': N, 'retrieval_dim': D, 'history_steps': 12, 'horizon': H, 'thresholds': {'context_low': alo, 'context_high': ahi, 'future_low': flo, 'future_high': fhi, 'key_low': klo, 'key_high': khi}, 'cluster_plot_summary': plots, 'cases': chosen}
    (out / 'mirage_cases.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame([{'type': t, **r} for t, rows in chosen.items() for r in rows]).to_csv(out / 'mirage_cases.csv', index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
***
