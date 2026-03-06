import io, torch, numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
db_vectors = np.load('data/clip_db.npy')
db_labels = np.load('data/db_labels.npy')

def search_vibe(image_bytes, top_k=3):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        client_vec = model.get_image_features(**inputs)
        client_vec = client_vec.image_embeds if not isinstance(client_vec, torch.Tensor) and hasattr(client_vec, 'image_embeds') else (client_vec[0] if not isinstance(client_vec, torch.Tensor) else client_vec)
        
    sim_scores = cosine_similarity(client_vec.cpu().numpy().reshape(1, -1), db_vectors)[0]
    
    unique_places = []
    for idx in np.argsort(sim_scores)[::-1]:
        if db_labels[idx] not in unique_places:
            unique_places.append(db_labels[idx])
        if len(unique_places) == top_k: break
    return unique_places

    # # Lấy thẳng 3 index cao nhất không cần lọc
    # top_3_idx = np.argsort(sim_scores)[::-1][:3]
    # return [db_labels[i] for i in top_3_idx]