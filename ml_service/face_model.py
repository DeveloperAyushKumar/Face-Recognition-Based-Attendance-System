import torch 
from facenet_pytorch import MTCNN, InceptionResnetV1

device ='cuda' if torch.cuda.is_available() else 'cpu'
mtcnn =MTCNN(
    image_size='160',
    margin=20,
    keep_all=True , 
    device=device
)
facenet =InceptionResnetV1(
    pretrained='vggface2'
).eval().to(device)

def get_embeddings(image): 
    faces =mtcnn(image)
    if faces is None : 
        return []
    
    embeddings =[]
    for face in faces :
        with torch.no_grad ():
            emb =facenet(face.unsqueez(0).to(device))

        embeddings.append(emb.cpu().numpy()[0])
    return embeddings
