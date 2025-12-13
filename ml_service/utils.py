import base64
import io
import numpy as np
from PIL import Image
from numpy.linalg import norm
import os

EMBED_DIR = "embeddings"
THRESHOLD = 0.8

os.makedirs(EMBED_DIR, exist_ok=True)

def base64_to_image(b64):
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def save_embedding(student_id, embedding):
    path = f"{EMBED_DIR}/{student_id}.npy"
    if os.path.exists(path):
        old = np.load(path)
        embedding = np.vstack([old, embedding])
    np.save(path, embedding)

def load_database():
    db = {}
    for f in os.listdir(EMBED_DIR):
        sid = f.replace(".npy", "")
        db[sid] = np.load(f"{EMBED_DIR}/{f}")
    return db

def match_face(embedding, database):
    best_id, best_dist = None, float("inf")
    for sid, embs in database.items():
        d = np.mean([norm(embedding - e) for e in embs])
        if d < best_dist:
            best_dist, best_id = d, sid

    if best_dist < THRESHOLD:
        return best_id, best_dist
    return "Unknown", best_dist
