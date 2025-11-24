#!/usr/bin/env python3
"""
Usage: python3 run.py <dataset-name>
Example: python3 run.py sift-128-euclidean
"""

import sys
import time
import numpy as np
import faiss
import lorann
import matplotlib.pyplot as plt
import urllib.request
import h5py
import os

DATASETS = {
    "sift-128-euclidean":         {"url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5", "metric": "euclidean"},
    "gist-960-euclidean":         {"url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5", "metric": "euclidean"},
    "glove-25-angular":           {"url": "http://ann-benchmarks.com/glove-25-angular.hdf5", "metric": "angular"},
    "glove-50-angular":           {"url": "http://ann-benchmarks.com/glove-50-angular.hdf5", "metric": "angular"},
    "glove-100-angular":          {"url": "http://ann-benchmarks.com/glove-100-angular.hdf5", "metric": "angular"},
    "glove-200-angular":          {"url": "http://ann-benchmarks.com/glove-200-angular.hdf5", "metric": "angular"},
    "kosarak-jaccard":            {"url": "http://ann-benchmarks.com/kosarak-jaccard.hdf5", "metric": "jaccard"},
    "movielens10m-jaccard":       {"url": "http://ann-benchmarks.com/movielens10m-jaccard.hdf5", "metric": "jaccard"},
    "nytimes-256-angular":        {"url": "http://ann-benchmarks.com/nytimes-256-angular.hdf5", "metric": "angular"},
    "lastfm-64-dot":              {"url": "http://ann-benchmarks.com/lastfm-64-dot.hdf5", "metric": "angular"},
    "deep-image-96-angular":      {"url": "http://ann-benchmarks.com/deep-96-angular.hdf5", "metric": "angular"},
    "fashion-mnist-784-euclidean":{"url": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5", "metric": "euclidean"},
    "mnist-784-euclidean":        {"url": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5", "metric": "euclidean"},
    "coco-i2i-512-angular":       {"url": "https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-i2i-512-angular.hdf5", "metric": "angular"},
    "coco-t2i-512-angular":       {"url": "https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3/coco-t2i-512-angular.hdf5", "metric": "angular"},
}

if len(sys.argv) != 2 or sys.argv[1] not in DATASETS:
    print("Usage: python3 bench.py <dataset>")
    print("Available datasets:")
    for name in DATASETS:
        print(f"  {name}")
    sys.exit(1)

DATASET = sys.argv[1]
URL = DATASETS[DATASET]["url"]
METRIC = DATASETS[DATASET]["metric"]
FILENAME = f"{DATASET}.hdf5"

if not os.path.exists(FILENAME):
    print(f"Downloading {DATASET} ...")
    urllib.request.urlretrieve(URL, FILENAME)

f = h5py.File(FILENAME, 'r')
X_train = f['train'][:].astype('float32')
X_query = f['test'][:].astype('float32')
f.close()

print(f"{DATASET} loaded: {X_train.shape[0]:,} base, {X_query.shape[0]:,} queries, d={X_train.shape[1]}, metric={METRIC}")

def preprocess_data(data, metric):
    if metric == "angular":
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return data / norms
    elif metric == "jaccard":
        return (data > 0).astype('float32')
    else:
        return data

X_train_processed = preprocess_data(X_train, METRIC)
X_query_processed = preprocess_data(X_query, METRIC)

def get_faiss_metric(metric):
    if metric == "euclidean":
        return faiss.METRIC_L2
    elif metric == "angular":
        return faiss.METRIC_INNER_PRODUCT
    else:
        return faiss.METRIC_Jaccard

def get_lorann_distance(metric):
    if metric == "euclidean":
        return lorann.L2
    elif metric == "angular":
        return lorann.IP
    else:
        return lorann.L2

def get_scann_distance(metric):
    if metric == "euclidean":
        return "squared_l2"
    elif metric == "angular":
        return "dot_product"
    else:
        return "squared_l2"

FAISS_METRIC = get_faiss_metric(METRIC)
LORANN_DISTANCE = get_lorann_distance(METRIC)
SCANN_DISTANCE = get_scann_distance(METRIC)

N_QUERY = X_query.shape[0]
D = X_train.shape[1]
K = 100

try:
    import scann
    SCANN_AVAILABLE = True
except ImportError:
    print("ScaNN can't be imported")
    SCANN_AVAILABLE = False

try:
    import glass
    GLASS_AVAILABLE = True
except ImportError:
    print("GLASS can't be imported")
    GLASS_AVAILABLE = False

print("\nComputing ground truth......")
if METRIC == "euclidean":
    exact = faiss.IndexFlatL2(D)
elif METRIC == "angular":
    exact = faiss.IndexFlatIP(D)
else:
    exact = faiss.IndexFlat(D, FAISS_METRIC)

exact.add(X_train_processed)
_, exact_idx = exact.search(X_query_processed, K)

print("\nBuilding FAISS-IVFPQ.......")
nlist = 4096
possible_m = [32, 25, 20, 16, 10, 8, 5, 4, 2, 1]
m = 1
for i in possible_m:
    if D % i == 0:
        m = i
        break 
nbits = 8

if METRIC == "euclidean":
    quantizer_ivfpq = faiss.IndexFlatL2(D)
elif METRIC == "angular":
    quantizer_ivfpq = faiss.IndexFlatIP(D)
else:
    quantizer_ivfpq = faiss.IndexFlat(D, FAISS_METRIC)

faiss_ivfpq = faiss.IndexIVFPQ(quantizer_ivfpq, D, nlist, m, nbits, FAISS_METRIC)
faiss_ivfpq.cp.min_points_per_centroid = 5
faiss_ivfpq.verbose = False
t0 = time.time()
faiss_ivfpq.train(X_train_processed)
faiss_ivfpq.add(X_train_processed)
build_time_ivfpq = time.time() - t0
print(f"FAISS-IVFPQ built in {build_time_ivfpq:.2f}s")

print("\nBuilding FAISS-IVFFlat......")
nlist_flat = 4096
if METRIC == "euclidean":
    quantizer_flat = faiss.IndexFlatL2(D)
elif METRIC == "angular":
    quantizer_flat = faiss.IndexFlatIP(D)
else:
    quantizer_flat = faiss.IndexFlat(D, FAISS_METRIC)

faiss_ivfflat = faiss.IndexIVFFlat(quantizer_flat, D, nlist_flat, FAISS_METRIC)
faiss_ivfflat.cp.min_points_per_centroid = 5
faiss_ivfflat.verbose = False
t0 = time.time()
faiss_ivfflat.train(X_train_processed)
faiss_ivfflat.add(X_train_processed)
build_time_ivfflat = time.time() - t0
print(f"FAISS-IVFFlat built in {build_time_ivfflat:.2f}s")

print("\nBuilding FAISS-HNSW......")
M = 32
if METRIC == "euclidean":
    faiss_hnsw = faiss.IndexHNSWFlat(D, M)
elif METRIC == "angular":
    faiss_hnsw = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
else:
    faiss_hnsw = faiss.IndexHNSWFlat(D, M, FAISS_METRIC)

faiss_hnsw.hnsw.efConstruction = 200
faiss_hnsw.verbose = False
t0 = time.time()
faiss_hnsw.add(X_train_processed)
build_time_hnsw = time.time() - t0
print(f"FAISS-HNSW built in {build_time_hnsw:.2f}s")

print("\nBuilding FAISS-IVFPQFS......")
nbits_fs = 4
possible_M = [32, 28, 25, 24, 20, 16, 14, 12, 10, 8, 7, 5, 4, 2, 1]
M_fs = 1
for m in possible_M:
    if D % m == 0:
        M_fs = m
        break
if M_fs == 1:
    M_fs = 32

if METRIC == "euclidean":
    quantizer_ivfpqfs = faiss.IndexFlatL2(D)
elif METRIC == "angular":
    quantizer_ivfpqfs = faiss.IndexFlatIP(D)
else:
    quantizer_ivfpqfs = faiss.IndexFlat(D, FAISS_METRIC)

faiss_ivfpqfs = faiss.IndexIVFPQFastScan(quantizer_ivfpqfs, D, nlist, M_fs, nbits_fs, FAISS_METRIC)
faiss_ivfpqfs.cp.min_points_per_centroid = 5
faiss_ivfpqfs.verbose = False
t0 = time.time()
faiss_ivfpqfs.train(X_train_processed)
faiss_ivfpqfs.add(X_train_processed)
build_time_ivfpqfs = time.time() - t0
print(f"FAISS-IVFPQFS built in {build_time_ivfpqfs:.2f}s")

if SCANN_AVAILABLE:
    print("\nBuilding ScaNN.......")
    t0 = time.time()
    scann_searcher = scann.scann_ops_pybind.builder(X_train_processed, 10, SCANN_DISTANCE)
    scann_searcher = scann_searcher.tree(
        num_leaves=2000, 
        num_leaves_to_search=100, 
        training_sample_size=250000
    )
    scann_searcher = scann_searcher.score_ah(
        2, 
        anisotropic_quantization_threshold=0.2
    )
    scann_searcher = scann_searcher.reorder(100).build()
    build_time_scann = time.time() - t0
    print(f"ScaNN built in {build_time_scann:.2f}s")
else:
    scann_searcher = None
    build_time_scann = 0

if GLASS_AVAILABLE:
    print("\nBuilding GLASS.......")
    t0 = time.time()
    glass_metric = "L2" if METRIC == "euclidean" else "COSINE"
    index = glass.Index(index_type="HNSW", metric=glass_metric, R=32, L=50, quant="SQ8U")
    graph = index.build(X_train_processed)
    glass_searcher = glass.Searcher(
        graph=graph, 
        data=X_train_processed, 
        metric=glass_metric,
        quantizer="SQ4U"
    )
    glass_searcher.set_ef(32)
    build_time_glass = time.time() - t0
    print(f"GLASS built in {build_time_glass:.2f}s")
else:
    glass_searcher = None
    build_time_glass = 0

def pad_to_multiple_of_32(data):
    current_dim = data.shape[1]
    target_dim = ((current_dim + 31) // 32) * 32
    pad_width = target_dim - current_dim
    if pad_width > 0:
        padded_data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
        return padded_data, target_dim
    return data, current_dim

print("\nBuilding LoRANN.......")
X_train_padded, D_padded = pad_to_multiple_of_32(X_train_processed)
X_query_padded, _ = pad_to_multiple_of_32(X_query_processed)

global_d = None

lorann_index = lorann.LorannIndex(
    data=X_train_padded,
    n_clusters=1024,
    global_dim=global_d, #D_padded
    quantization_bits=4,
    distance=LORANN_DISTANCE
)

t0 = time.time()
lorann_index.build()
build_time_lorann = time.time() - t0
print(f"LoRANN built in {build_time_lorann:.2f}s")

def calculate_recall(approx_indices, exact_indices, k):
    recalls = []
    for approx, exact in zip(approx_indices, exact_indices):
        if len(exact) > 0:
            intersection = len(np.intersect1d(approx[:k], exact[:k]))
            recalls.append(intersection / min(k, len(exact)))
    return np.mean(recalls)

ivfpq_probes = [8, 16, 32, 64, 128, 256, 512]
ivfflat_probes = [8, 16, 32, 64, 128, 256]  
hnsw_ef_search = [16, 32, 64, 128, 256, 512, 1024]
ivfpqfs_probes = [8, 16, 32, 64, 128, 256, 512]
scann_leaves_to_search = [10, 20, 50, 100, 200, 500]
glass_ef_search = [16, 32, 64, 128, 256, 512]
lorann_probes = [8, 16, 32, 64, 128, 256]
rerank_points = [100, 200, 400, 800, 1200, 1600, 2400, 3200]

print("Testing Phase:")
print("="*30)

print("\nFAISS-IVFPQ: Testing Seaching......")
ivfpq_results = []
for nprobe in ivfpq_probes:
    faiss_ivfpq.nprobe = nprobe
    t0 = time.time()
    D_ivfpq, I_ivfpq = faiss_ivfpq.search(X_query_processed, K)
    elapsed = time.time() - t0
    recall = calculate_recall(I_ivfpq, exact_idx, K)
    qps = N_QUERY / elapsed
    ivfpq_results.append((nprobe, recall, qps))
    print(f"  nprobe={nprobe:3d} → Recall@100={recall:.4f}, QPS={qps:6.0f}")

print("\nFAISS-IVFFlat: Testing Seaching......")
ivfflat_results = []
for nprobe in ivfflat_probes:
    faiss_ivfflat.nprobe = nprobe
    t0 = time.time()
    D_flat, I_flat = faiss_ivfflat.search(X_query_processed, K)
    elapsed = time.time() - t0
    recall = calculate_recall(I_flat, exact_idx, K)
    qps = N_QUERY / elapsed
    ivfflat_results.append((nprobe, recall, qps))
    print(f"  nprobe={nprobe:3d} → Recall@100={recall:.4f}, QPS={qps:6.0f}")

print("\nFAISS-HNSW: Testing Seaching......")
hnsw_results = []
for ef_search in hnsw_ef_search:
    faiss_hnsw.hnsw.efSearch = ef_search
    t0 = time.time()
    D_hnsw, I_hnsw = faiss_hnsw.search(X_query_processed, K)
    elapsed = time.time() - t0
    recall = calculate_recall(I_hnsw, exact_idx, K)
    qps = N_QUERY / elapsed
    hnsw_results.append((ef_search, recall, qps))
    print(f"  ef={ef_search:3d} → Recall@100={recall:.4f}, QPS={qps:6.0f}")

print("\nFAISS-IVFPQFS: Testing Seaching......")
ivfpqfs_results = []
for nprobe in ivfpqfs_probes:
    faiss_ivfpqfs.nprobe = nprobe
    t0 = time.time()
    D_ivfpqfs, I_ivfpqfs = faiss_ivfpqfs.search(X_query_processed, K)
    elapsed = time.time() - t0
    recall = calculate_recall(I_ivfpqfs, exact_idx, K)
    qps = N_QUERY / elapsed
    ivfpqfs_results.append((nprobe, recall, qps))
    print(f"  nprobe={nprobe:3d} → Recall@100={recall:.4f}, QPS={qps:6.0f}")

if SCANN_AVAILABLE:
    print("\nScaNN: Testing Seaching......")
    scann_results = []
    for leaves in scann_leaves_to_search:
        t0 = time.time()
        neighbors, distances = scann_searcher.search_batched(X_query_processed, leaves_to_search=leaves, final_num_neighbors=K)
        elapsed = time.time() - t0
        recall = calculate_recall(neighbors, exact_idx, K)
        qps = N_QUERY / elapsed
        scann_results.append((leaves, recall, qps))
        print(f"  leaves={leaves:3d} → Recall@100={recall:.4f}, QPS={qps:6.0f}")
else:
    scann_results = []

if GLASS_AVAILABLE:
    print("\nGLASS: Testing Seaching......")
    glass_results = []
    for ef_search in glass_ef_search:
        glass_searcher.set_ef(ef_search)
        t0 = time.time()
        all_indices = []
        for i in range(N_QUERY):
            ret = glass_searcher.search(query=X_query_processed[i], k=K)
            if ret and len(ret) > 0:
                if isinstance(ret[0], tuple):
                    indices = [idx for idx, dist in ret]
                else:
                    indices = ret
                all_indices.append(indices[:K])
            else:
                all_indices.append([])
        elapsed = time.time() - t0
        indices_array = np.array(all_indices)
        recall = calculate_recall(indices_array, exact_idx, K)
        qps = N_QUERY / elapsed
        glass_results.append((ef_search, recall, qps))
        print(f"  ef={ef_search:3d} → Recall@100={recall:.4f}, QPS={qps:6.0f}")
else:
    glass_results = []

print("\nLoRANN: Testing Seaching......")
lorann_results = []
for i, p in enumerate(lorann_probes):
    rerank = rerank_points[i] if i < len(rerank_points) else rerank_points[-1]
    
    t0 = time.time()
    approx_idx, _ = lorann_index.search(X_query_padded, K, p, rerank, return_distances=True)
    elapsed = time.time() - t0
    recall = calculate_recall(approx_idx, exact_idx, K)
    qps = N_QUERY / elapsed
    lorann_results.append((p, rerank, recall, qps))
    print(f"  probes={p:3d}, rerank={rerank:4d} → Recall@100={recall:.4f}, QPS={qps:6.0f}")

plt.figure(figsize=(14, 9))
ivfpq_r, ivfpq_qps = zip(*[(r, q) for _, r, q in ivfpq_results])
ivfflat_r, ivfflat_qps = zip(*[(r, q) for _, r, q in ivfflat_results])
hnsw_r, hnsw_qps = zip(*[(r, q) for _, r, q in hnsw_results])
ivfpqfs_r, ivfpqfs_qps = zip(*[(r, q) for _, r, q in ivfpqfs_results])
lorann_r, lorann_qps = zip(*[(r, q) for _, _, r, q in lorann_results])

if SCANN_AVAILABLE:
    scann_r, scann_qps = zip(*[(r, q) for _, r, q in scann_results])
if GLASS_AVAILABLE:
    glass_r, glass_qps = zip(*[(r, q) for _, r, q in glass_results])

plt.plot(ivfpq_r, ivfpq_qps, 'o-', label='FAISS-IVFPQ', color='#2ca02c', lw=2, ms=8)
plt.plot(ivfflat_r, ivfflat_qps, 's-', label='FAISS-IVFFlat', color='#ff7f0e', lw=2, ms=8)
plt.plot(hnsw_r, hnsw_qps, '^-', label='FAISS-HNSW', color='#d62728', lw=2, ms=8)
plt.plot(ivfpqfs_r, ivfpqfs_qps, 'v-', label='FAISS-IVFPQFS', color='#9467bd', lw=2, ms=8)
plt.plot(lorann_r, lorann_qps, 'D-', label='LoRANN', color='#1f77b4', lw=2, ms=8)
if SCANN_AVAILABLE:
    plt.plot(scann_r, scann_qps, '*-', label='ScaNN', color='#8c564b', lw=2, ms=10)
if GLASS_AVAILABLE:
    plt.plot(glass_r, glass_qps, 'X-', label='GLASS', color='#e377c2', lw=2, ms=10)

for r, q, p in zip(ivfpq_r, ivfpq_qps, ivfpq_probes):
    plt.text(r+0.002, q, str(p), fontsize=8, color='#2ca02c')
for r, q, p in zip(ivfflat_r, ivfflat_qps, ivfflat_probes):
    plt.text(r+0.002, q, str(p), fontsize=8, color='#ff7f0e')
for r, q, p in zip(hnsw_r, hnsw_qps, hnsw_ef_search):
    plt.text(r+0.002, q, f"ef{p}", fontsize=8, color='#d62728')
for r, q, p in zip(ivfpqfs_r, ivfpqfs_qps, ivfpqfs_probes):
    plt.text(r+0.002, q, str(p), fontsize=8, color='#9467bd')
for r, q, (p, rerank) in zip(lorann_r, lorann_qps, [(p, r) for p, r, _, _ in lorann_results]):
    plt.text(r+0.002, q, f"p{p}", fontsize=8, color='#1f77b4')
if SCANN_AVAILABLE:
    for r, q, p in zip(scann_r, scann_qps, scann_leaves_to_search):
        plt.text(r+0.002, q, f"l{p}", fontsize=8, color='#8c564b')
if GLASS_AVAILABLE:
    for r, q, p in zip(glass_r, glass_qps, glass_ef_search):
        plt.text(r+0.002, q, f"ef{p}", fontsize=8, color='#e377c2')

plt.xlabel('Recall@100', fontsize=14)
plt.ylabel('Queries per Second (QPS)', fontsize=14)
plt.title(f'ANN Algorithm Comparison on {DATASET} (metric: {METRIC})\n', fontsize=15)
plt.legend(fontsize=11, loc='upper right')
plt.grid(alpha=0.3)
plt.xlim(0.3, 1.02)
all_qps = list(ivfpq_qps) + list(ivfflat_qps) + list(hnsw_qps) + list(ivfpqfs_qps) + list(lorann_qps)
if SCANN_AVAILABLE:
    all_qps.extend(scann_qps)
if GLASS_AVAILABLE:
    all_qps.extend(glass_qps)

plt.ylim(0, max(all_qps) * 1.15)
plt.tight_layout()
plt.savefig(f"comparison_{DATASET}.png", dpi=200, bbox_inches='tight')
plt.savefig(f"comparison_{DATASET}.pdf", bbox_inches='tight')
print("\nPlot saved as:", f"comparison_{DATASET}.png")
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"{'Method':<15} {'Build Time':<12} {'Best QPS':<10} {'Max Recall':<12} {'Params'}")
print("-"*80)

def print_best_results(name, results, build_time):
    if name == "LoRANN":
        best_qps = max(q for _, _, _, q in results)
        max_recall = max(r for _, _, r, _ in results)
    else:
        best_qps = max(q for _, _, q in results)
        max_recall = max(r for _, r, _ in results)
    print(f"{name:<15} {build_time:6.1f}s      {best_qps:8.0f}   {max_recall:.4f}")

print_best_results("FAISS-IVFPQ", ivfpq_results, build_time_ivfpq)
print_best_results("FAISS-IVFFlat", ivfflat_results, build_time_ivfflat)
print_best_results("FAISS-HNSW", hnsw_results, build_time_hnsw)
print_best_results("FAISS-IVFPQFS", ivfpqfs_results, build_time_ivfpqfs)
if SCANN_AVAILABLE:
    print_best_results("ScaNN", scann_results, build_time_scann)
if GLASS_AVAILABLE:
    print_best_results("GLASS", glass_results, build_time_glass)
print_best_results("LoRANN", lorann_results, build_time_lorann)


print("="*80)
