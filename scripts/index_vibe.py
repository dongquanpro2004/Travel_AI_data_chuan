import os, torch, numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
db_dir, clip_db, db_labels = r'D:\Travel_AI_best\data\vibe_dataset', [], []

for folder in os.listdir(db_dir):
    path = os.path.join(db_dir, folder)
    if os.path.isdir(path) and folder != 'noise_others':
        for img in os.listdir(path):
            try:
                inputs = processor(images=Image.open(os.path.join(path, img)).convert("RGB"), return_tensors="pt")
                with torch.no_grad():
                    feat = model.get_image_features(**inputs)
                    feat = feat.image_embeds if not isinstance(feat, torch.Tensor) and hasattr(feat, 'image_embeds') else (feat[0] if not isinstance(feat, torch.Tensor) else feat)
                clip_db.append(feat.cpu().numpy().flatten())
                db_labels.append(folder)
            except: continue

np.save(r'D:\Travel_AI_best\data\clip_db.npy', np.array(clip_db))
np.save(r'D:\Travel_AI_best\data\db_labels.npy', np.array(db_labels))
print("Đã tạo xong 2 file .npy!")