from fastapi import FastAPI
from pydantic import BaseModel
from face_model import get_embeddings
from utils import base64_to_image, save_embedding, load_database, match_face

app = FastAPI()

class EnrollRequest(BaseModel):
    student_id: str
    image_base64: str

class RecognizeRequest(BaseModel):
    image_base64: str

@app.post("/enroll")
def enroll(req: EnrollRequest):
    img = base64_to_image(req.image_base64)
    embeddings = get_embeddings(img)

    for emb in embeddings:
        save_embedding(req.student_id, emb)

    return {"status": "enrolled", "embeddings_added": len(embeddings)}

@app.post("/recognize")
def recognize(req: RecognizeRequest):
    img = base64_to_image(req.image_base64)
    embeddings = get_embeddings(img)

    if not embeddings:
        return {"status": "no_face_detected"}

    db = load_database()
    identity, dist = match_face(embeddings[0], db)

    return {
        "identity": identity,
        "distance": float(dist),
        "status": "recognized" if identity != "Unknown" else "unknown"
    }
