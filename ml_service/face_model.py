import torch 
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F
device ='cuda' if torch.cuda.is_available() else 'cpu'
# mtcnn =MTCNN(
#     image_size=160,
#     margin=20,
#     keep_all=True , 
#     device=device
# )
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    keep_all=False,   # ⭐ IMPORTANT
    device=device
)
facenet =InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)

# def get_embeddings(image):
#     faces = mtcnn(image)
#     if faces is None:
#         return []

#     embeddings = []
#     for face in faces:
#         with torch.no_grad():
#             emb = facenet(face.unsqueeze(0).to(device))
#             emb = F.normalize(emb, p=2, dim=1)  # ⭐ NORMALIZE HERE
#         embeddings.append(emb.cpu().numpy()[0])

#     return embeddings
# import torch.nn.functional as F

def get_embeddings(image):
    face = mtcnn(image)
    if face is None:
        return []

    with torch.no_grad():
        emb = facenet(face.unsqueeze(0).to(device))
        emb = F.normalize(emb, p=2, dim=1)

    return [emb.cpu().numpy()[0]]

