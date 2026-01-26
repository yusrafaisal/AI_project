# app.py
import streamlit as st
from PIL import Image
import os
import numpy as np
import pickle
from pathlib import Path
import math
import cv2
from rembg import remove

CLUSTER_ROOT = "./clusters"
CACHE_DIR = "./cache"
META_CACHE = os.path.join(CACHE_DIR, "meta.pkl")
EMBED_CACHE = os.path.join(CACHE_DIR, "embeddings.npy")
TOP_K = 5
TOP_K_TEXT = 10
USE_HSV_REFINEMENT = True
HSV_WEIGHT = 0.5

st.set_page_config(layout="wide", page_title="Fashion Explorer")

st.markdown("""
    <style>
        /* Reduce default top padding in Streamlit main block */
        .css-18e3th9 {  /* main block container */
            padding-top: 0rem;  /* adjust this value (default is ~4rem) */
        }
    </style>
    <div style="text-align: center; margin-top: 0;">
        <h1 style="font-size: 50px; margin-bottom: 0;">OUTFITLY</h1>
        <h3 style="font-size: 28px; margin-top: 0px;">AI-based Pakistani Outfit Styling and Recommendation System</h3>
    </div>
""", unsafe_allow_html=True)

# Load metadata (lazy)
@st.cache_data(show_spinner=False)
def load_meta():
    if not os.path.exists(META_CACHE):
        raise FileNotFoundError("Meta cache not found. Run cluster.py first.")
    with open(META_CACHE, "rb") as f:
        meta = pickle.load(f)
    embeddings = np.load(EMBED_CACHE)
    return meta, embeddings

meta, embeddings = load_meta()
pca = meta.get("pca", None)
kmeans = meta["kmeans"]
labels = meta["labels"]
hsv_feats = meta["hsv_feats"]
image_paths = meta["image_paths"]

color_feats = meta.get("color_feats", None)
texture_feats = meta.get("texture_feats", None)

# cluster folders
cluster_dirs = sorted([d for d in os.listdir(CLUSTER_ROOT) if d.startswith("cluster_")])

def normalize_np(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / (n if n > 0 else 1)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n==0] = 1
    return x / n

from transformers import CLIPProcessor, CLIPModel
import torch
MODEL_NAME = "patrickjohncyh/fashion-clip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor

model, processor = load_model()

# HSV similarity
def hsv_sim(q, t):
    dh = min(abs(q[0]-t[0]), 360-abs(q[0]-t[0])) / 180
    ds = abs(q[1]-t[1])
    dv = abs(q[2]-t[2])
    return 1 - (0.6*dh + 0.2*ds + 0.2*dv)

# Texture similarity
def texture_sim(q, t):
    if q is None or t is None:
        return 0.5
    diff = np.linalg.norm(q - t)
    return 1.0 - (diff / 2.0)

# Color similarity
def color_sim(q, t):
    if q is None or t is None:
        return 0.5
    diff = np.linalg.norm(q - t)
    return 1.0 - (diff / 2.0)

# query embedders
def embed_query_image_pil(img_pil):
    img = img_pil.convert("RGB")
    inp = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_image_features(**inp).cpu().numpy().squeeze()
    return normalize_np(emb)

def embed_text(prompt):
    inp = processor(text=[prompt], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_text_features(**inp).cpu().numpy().squeeze()
    return normalize_np(emb)

def get_texture_features(img_pil):
    try:
        img = np.array(img_pil.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (64, 64))
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (64 * 64)
        contrast = gray.std() / 255.0
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = laplacian.var() / 10000.0
        
        return normalize_np(np.array([edge_density, contrast, texture_var], dtype=np.float32))
    except:
        return None

def get_color_features(img_pil):
    try:
        img = np.array(img_pil.convert("RGB"))
        img_small = cv2.resize(img, (64, 64))
        hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        h_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        
        h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
        s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
        
        return normalize_np(np.concatenate([h_hist, s_hist]).astype(np.float32))
    except:
        return None

def get_cluster(emb):
    r = pca.transform(emb.reshape(1,-1)) if embeddings.shape[1] > (pca.n_components_ if pca is not None else embeddings.shape[1]) else emb.reshape(1,-1)
    return int(kmeans.predict(r)[0])

def recommend_from_embedding(emb, query_hsv=None, query_color=None, query_texture=None, K=TOP_K):
   
    cid = get_cluster(emb)
    idxs = [i for i,l in enumerate(labels) if l == cid]
    
    # Base CLIP similarity
    sims = embeddings[idxs].dot(emb)
    order = np.argsort(-sims)
    idxs = np.array(idxs)[order]
    sims = sims[order]
    

    if query_hsv is not None and USE_HSV_REFINEMENT:
        hsv_scores = np.array([hsv_sim(query_hsv, hsv_feats[i]) for i in idxs])
        final = (1 - HSV_WEIGHT) * sims + HSV_WEIGHT * hsv_scores
        
        if query_color is not None and color_feats is not None:
            color_scores = np.array([color_sim(query_color, color_feats[i]) for i in idxs])
            final = 0.7 * final + 0.15 * color_scores
        
        if query_texture is not None and texture_feats is not None:
            texture_scores = np.array([texture_sim(query_texture, texture_feats[i]) for i in idxs])
            final = 0.85 * final + 0.15 * texture_scores
        
        order2 = np.argsort(-final)
        idxs = idxs[order2]
        sims = final[order2]
    
    return idxs[:K], sims[:K], cid

def recommend_from_text_embedding(emb, query_hsv=None, query_color=None, query_texture=None, K=TOP_K_TEXT):

    cid = get_cluster(emb)
    idxs = [i for i,l in enumerate(labels) if l == cid]
    
    # Base CLIP similarity
    sims = embeddings[idxs].dot(emb)
    order = np.argsort(-sims)
    idxs = np.array(idxs)[order]
    sims = sims[order]
    
    if query_hsv is not None and USE_HSV_REFINEMENT:
        hsv_scores = np.array([hsv_sim(query_hsv, hsv_feats[i]) for i in idxs])
        final = (1 - HSV_WEIGHT) * sims + HSV_WEIGHT * hsv_scores
        
        if query_color is not None and color_feats is not None:
            color_scores = np.array([color_sim(query_color, color_feats[i]) for i in idxs])
            final = 0.7 * final + 0.15 * color_scores
        
        if query_texture is not None and texture_feats is not None:
            texture_scores = np.array([texture_sim(query_texture, texture_feats[i]) for i in idxs])
            final = 0.85 * final + 0.15 * texture_scores
        
        order2 = np.argsort(-final)
        idxs = idxs[order2]
        sims = final[order2]
    
    return idxs[:K], sims[:K], cid


def get_avg_hsv_pil(img_pil):
    img = img_pil.convert("RGB").resize((64,64))
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[:,:,0].mean() * 2
    s = hsv[:,:,1].mean() / 255.0
    v = hsv[:,:,2].mean() / 255.0
    return np.array([h,s,v], dtype=np.float32)

def remove_bg_pil(img_pil):
    import io
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    output = remove(img_bytes)
    out_img = Image.open(io.BytesIO(output)).convert("RGBA")  
    # convert to RGB (white background) 
    bg = Image.new("RGB", out_img.size, (255, 255, 255))
    bg.paste(out_img, mask=out_img.split()[3])  
    return bg

# UI
mode = st.sidebar.radio("Mode", ["Browse Clusters", "Image Recommendation", "Text Search"])
if mode == "Browse Clusters":
    cid = st.selectbox("Cluster", cluster_dirs)
    folder = os.path.join(CLUSTER_ROOT, cid)
    files = sorted(os.listdir(folder))
    cols = st.columns(5)
    col_idx = 0
    for f in files:
        path = os.path.join(folder, f)
        img = Image.open(path)
        with cols[col_idx]:
            st.image(img, use_container_width=True)
        col_idx = (col_idx + 1) % 5

elif mode == "Image Recommendation":
    st.subheader("Upload an image to find similar outfits")
    uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png"])
    if uploaded:
        # remove background first
        qimg = Image.open(uploaded).convert("RGB")
        qimg_nobg = remove_bg_pil(qimg)
        st.image(qimg_nobg, caption="Query (background removed)", width=300)

        # compute embeddings & features on background removed image
        qhsv = get_avg_hsv_pil(qimg_nobg)
        qemb = embed_query_image_pil(qimg_nobg)
        qcolor = get_color_features(qimg_nobg)  # NEW
        qtexture = get_texture_features(qimg_nobg)  # NEW

        rec, score, cid = recommend_from_embedding(qemb, qhsv, qcolor, qtexture)
        st.markdown(f"### Cluster {cid} — Top {len(rec)} recommendations")
        cols = st.columns(5)
        col_idx = 0
        for idx in rec:
            # find thumbnail file saved for idx
            folder = os.path.join(CLUSTER_ROOT, f"cluster_{cid}")
            # search for file starting with idx_
            candidates = [x for x in os.listdir(folder) if x.startswith(f"{idx}_")]
            path = os.path.join(folder, candidates[0]) if candidates else image_paths[idx]
            with cols[col_idx]:
                st.image(Image.open(path), use_container_width=True)
            col_idx = (col_idx + 1) % 5

elif mode == "Text Search":
    st.subheader("Search by text")
    txt = st.text_input("Describe an outfit (e.g. 'black hoodie streetwear')")
    if txt:
        qemb = embed_text(txt)
        rec, score, cid = recommend_from_text_embedding(qemb, None)
        st.markdown(f"### Cluster {cid} — Matches")
        cols = st.columns(5)
        col_idx = 0
        for idx in rec:
            folder = os.path.join(CLUSTER_ROOT, f"cluster_{cid}")
            candidates = [x for x in os.listdir(folder) if x.startswith(f"{idx}_")]
            path = os.path.join(folder, candidates[0]) if candidates else image_paths[idx]
            with cols[col_idx]:
                st.image(Image.open(path), use_container_width=True)
            col_idx = (col_idx + 1) % 5