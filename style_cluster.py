import os
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_ROOT = "./data/images"
OUT_CLUSTER_ROOT = "./clusters"
CACHE_DIR = "./cache"
EMBED_CACHE = os.path.join(CACHE_DIR, "embeddings.npy")
META_CACHE = os.path.join(CACHE_DIR, "meta.pkl")

# KEY PARAMETERS TO TUNE:
NUM_CLUSTERS = 6          
USE_PCA = True             
PCA_N = 256                
KMEANS_ITERATIONS = 500    
KMEANS_INIT = 20           

THUMB_SIZE = (512, 512)
BATCH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "patrickjohncyh/fashion-clip"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_CLUSTER_ROOT, exist_ok=True)

def gather_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() in exts:
                paths.append(os.path.join(r, f))
    return sorted(paths)

def normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / (n if n > 0 else 1)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n==0] = 1
    return x / n

def get_color_features(path):
    try:
        img = cv2.imread(path)
        if img is None:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_small = cv2.resize(img, (64, 64))
        hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
        
        return np.concatenate([h_hist, s_hist]).astype(np.float32)
    except:
        return np.zeros(26, dtype=np.float32)

def get_texture_features(path):
    try:
        img = cv2.imread(path)
        if img is None:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (64 * 64)
        contrast = gray.std() / 255.0
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = laplacian.var() / 10000.0
        
        return np.array([edge_density, contrast, texture_var], dtype=np.float32)
    except:
        return np.zeros(3, dtype=np.float32)

def get_avg_hsv(path):
    try:
        img = cv2.imread(path)
        if img is None:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        h = hsv[:,:,0].mean() * 2
        s = hsv[:,:,1].mean() / 255.0
        v = hsv[:,:,2].mean() / 255.0
        return np.array([h, s, v], dtype=np.float32)
    except:
        return np.array([0, 0, 0], dtype=np.float32)

print("="*60)
print("FASHION CLUSTERING")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Clusters: {NUM_CLUSTERS}")
print(f"PCA: {'Enabled (%d -> %d)' % (512, PCA_N) if USE_PCA else 'Disabled (full 512-dim)'}")
print("="*60)

image_paths = gather_images(DATA_ROOT)
print(f"\nFound {len(image_paths)} images")
if len(image_paths) == 0:
    raise SystemExit("No images found!")

recommended_clusters = max(4, min(12, len(image_paths) // 50))
if NUM_CLUSTERS > recommended_clusters:
    print(f" WARNING: {NUM_CLUSTERS} clusters may be too many for {len(image_paths)} images")
    print(f" Recommended: {recommended_clusters} clusters")

print("\nLoading FashionCLIP model...")
model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

# Compute or load embeddings
if os.path.exists(EMBED_CACHE) and os.path.exists(META_CACHE):
    print("\n Loading cached embeddings...")
    embs = np.load(EMBED_CACHE)
    with open(META_CACHE, "rb") as f:
        meta = pickle.load(f)
    
    # Check if we need to recompute with different settings
    need_recompute = False
    if USE_PCA and meta.get("pca") is None:
        print("  PCA enabled but not in cache - recomputing...")
        need_recompute = True
    elif USE_PCA and meta.get("pca") is not None and meta["pca"].n_components_ != PCA_N:
        print(f"  PCA dimension changed ({meta['pca'].n_components_} -> {PCA_N}) - recomputing...")
        need_recompute = True
    
    if not need_recompute:
        pca = meta.get("pca", None)
        kmeans = meta.get("kmeans", None)
        labels = meta.get("labels", None)
        hsv_feats = meta.get("hsv_feats", None)
        color_feats = meta.get("color_feats", None)
        texture_feats = meta.get("texture_feats", None)
        
        # Always recompute clustering if parameters changed
        print(f"Recomputing clustering with {NUM_CLUSTERS} clusters...")
        if pca is not None and USE_PCA:
            reduced = pca.transform(embs)
        else:
            reduced = embs
        
        kmeans = KMeans(
            n_clusters=NUM_CLUSTERS, 
            random_state=42, 
            n_init=KMEANS_INIT,
            max_iter=KMEANS_ITERATIONS
        )
        labels = kmeans.fit_predict(reduced)
        
        meta["kmeans"] = kmeans
        meta["labels"] = labels
        with open(META_CACHE, "wb") as f:
            pickle.dump(meta, f)
else:
    need_recompute = True

if need_recompute or not os.path.exists(EMBED_CACHE):
    print("\n Computing CLIP embeddings...")
    all_embs = []
    for i in tqdm(range(0, len(image_paths), BATCH), desc="CLIP"):
        batch_paths = image_paths[i:i+BATCH]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
        all_embs.append(emb.cpu().numpy())
    
    embs = np.vstack(all_embs)
    embs = normalize_np(embs)
    np.save(EMBED_CACHE, embs)
    print(f" Saved embeddings: {embs.shape}")

    # Extract features
    print("\n Extracting features...")
    hsv_feats = np.array([get_avg_hsv(p) for p in tqdm(image_paths, desc="HSV")])
    color_feats = np.array([get_color_features(p) for p in tqdm(image_paths, desc="Color")])
    color_feats = normalize_np(color_feats)
    texture_feats = np.array([get_texture_features(p) for p in tqdm(image_paths, desc="Texture")])
    texture_feats = normalize_np(texture_feats)

    # PCA (optional)
    pca = None
    if USE_PCA and embs.shape[1] > PCA_N:
        print(f"\n Running PCA: {embs.shape[1]} -> {PCA_N}...")
        pca = PCA(n_components=PCA_N, random_state=42)
        reduced = pca.fit_transform(embs)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f" Variance preserved: {var_explained:.2%}")
        if var_explained < 0.85:
            print(f" WARNING: Only {var_explained:.1%} variance preserved")
            print(f" Consider increasing PCA_N to {int(PCA_N * 1.5)} or disabling PCA")
    else:
        reduced = embs
        print("\n Using full 512-dim embeddings (PCA disabled)")

    # KMeans clustering
    print(f"\n Running KMeans (k={NUM_CLUSTERS})...")
    kmeans = KMeans(
        n_clusters=NUM_CLUSTERS,
        random_state=42,
        n_init=KMEANS_INIT,
        max_iter=KMEANS_ITERATIONS,
        verbose=0
    )
    labels = kmeans.fit_predict(reduced)

    # Save metadata
    meta = {
        "pca": pca,
        "kmeans": kmeans,
        "labels": labels,
        "hsv_feats": hsv_feats,
        "color_feats": color_feats,
        "texture_feats": texture_feats,
        "image_paths": image_paths
    }
    with open(META_CACHE, "wb") as f:
        pickle.dump(meta, f)
    print(f" Saved metadata")

# Clustering quality metrics
print("\n" + "="*60)
print("CLUSTERING QUALITY METRICS")
print("="*60)

if pca is not None and USE_PCA:
    reduced = pca.transform(embs)
else:
    reduced = embs

# Cluster distribution
print("\n" + "="*60)
print("CLUSTER DISTRIBUTION")
print("="*60)
cluster_to_indices = {}
for i, l in enumerate(labels):
    cluster_to_indices.setdefault(int(l), []).append(i)

sizes = [len(v) for v in cluster_to_indices.values()]
print(f"Total items: {len(image_paths)}")
print(f"Mean cluster size: {np.mean(sizes):.1f}")
print(f"Std deviation: {np.std(sizes):.1f}")
print(f"Min size: {min(sizes)} | Max size: {max(sizes)}")

if max(sizes) > 3 * np.mean(sizes):
    print(" WARNING: Very unbalanced clusters detected!")
    print(" Consider reducing NUM_CLUSTERS or increasing PCA_N")

print("\nPer-cluster breakdown:")
for cid in sorted(cluster_to_indices.keys()):
    count = len(cluster_to_indices[cid])
    bar = "█" * int(count / max(sizes) * 40)
    print(f" Cluster {cid:2d}: {count:4d} items {bar}")

# Save thumbnails
print("\n" + "="*60)
print("SAVING CLUSTER THUMBNAILS")
print("="*60)
for cid, idxs in cluster_to_indices.items():
    folder = os.path.join(OUT_CLUSTER_ROOT, f"cluster_{cid}")
    os.makedirs(folder, exist_ok=True)
    for idx in tqdm(idxs, desc=f"Cluster {cid}", leave=False):
        src = image_paths[idx]
        name = f"{idx}_{Path(src).name}"
        save_path = os.path.join(folder, name)
        if os.path.exists(save_path):
            continue
        try:
            img = Image.open(src).convert("RGB")
            img = img.resize(THUMB_SIZE, Image.Resampling.LANCZOS)
            img.save(save_path, "JPEG", quality=90)
        except Exception as e:
            print(f"Error: {e}")

# Create visualization
print("\n Creating cluster visualization...")
try:
    pca2 = PCA(n_components=2, random_state=42)
    points_2d = pca2.fit_transform(reduced)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        points_2d[:,0], points_2d[:,1], 
        c=labels, 
        cmap='tab10', 
        s=50, 
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"2D Cluster Visualization (k={NUM_CLUSTERS})", fontsize=16)
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(CACHE_DIR, "clusters_2d.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved visualization: {plot_path}")
except Exception as e:
    print(f"Visualization failed: {e}")

print("\n" + "="*60)
print(" CLUSTERING COMPLETE!")
print("="*60)
print(f"Clusters: {OUT_CLUSTER_ROOT}")
print(f"Cache: {CACHE_DIR}")
print(f"Visualization: {CACHE_DIR}/clusters_2d.png")
print("\nRun: streamlit run app.py")