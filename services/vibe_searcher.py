import io
import re
import torch
import requests
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# KHỞI TẠO CLIP VÀ LOAD VECTOR DATABASE
# ==========================================
try:
    print("Đang khởi tạo mô hình phân tích Vibe...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    db_vectors = np.load(r'D:\Travel_AI_best\data\clip_db.npy')
    db_labels = np.load(r'D:\Travel_AI_best\data\db_labels.npy')
except Exception as e:
    print(f"Lỗi load model hoặc data: {e}")
    model = None

import json
import html

def get_related_images_bing(query_name, num_images=3):
    """
    Cào cả URL ẢNH và URL TRANG WEB chứa ảnh từ Bing.
    """
    search_query = f"địa điểm {query_name} du lịch đẹp thực tế".replace(' ', '+')
    url = f"https://www.bing.com/images/search?q={search_query}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        
        # Tìm tất cả các khối data ẩn chứa thông tin ảnh của Bing
        blocks = re.findall(r'class="iusc".*?m="(.*?)"', response.text)
        
        results = []
        seen_urls = set()
        
        for block in blocks:
            # Giải mã các ký tự HTML (biến &quot; thành dấu ngoặc kép ")
            clean_json_str = html.unescape(block)
            try:
                # Ép kiểu nó về JSON Dictionary của Python
                data = json.loads(clean_json_str)
                img_url = data.get("murl") # Link ảnh
                page_url = data.get("purl") # Link trang web gốc
                
                # Tránh lấy ảnh trùng lặp
                if img_url and img_url not in seen_urls:
                    seen_urls.add(img_url)
                    results.append({
                        "image_url": img_url,
                        "source_page": page_url
                    })
                
                if len(results) >= num_images:
                    break
            except Exception:
                continue
                
        return results
    except Exception as e:
        print(f"Lỗi khi cào ảnh từ Bing cho {query_name}: {e}")
        return []

def search_vibe(image_bytes):
    if model is None:
        return {"error": "Model CLIP chưa sẵn sàng"}
        
    try:
        # 1. Xử lý ảnh đầu vào
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            client_vec = model.get_image_features(**inputs)
            
            # Ép kiểu Tensor an toàn
            if not isinstance(client_vec, torch.Tensor):
                if hasattr(client_vec, 'image_embeds'):
                    client_vec = client_vec.image_embeds
                elif hasattr(client_vec, 'pooler_output'):
                    client_vec = client_vec.pooler_output
                else:
                    client_vec = client_vec[0]
            
            # Chuẩn hóa Vector
            client_vec = client_vec / client_vec.norm(p=2, dim=-1, keepdim=True)
            
        # 2. So khớp với Database (Dùng Cosine Similarity)
        sim_scores = cosine_similarity(client_vec.cpu().numpy().reshape(1, -1), db_vectors)[0]
        
        # 3. Chỉ lấy đúng 1 nhãn có điểm cao nhất (Top 1)
        best_match_idx = np.argmax(sim_scores)
        best_place = db_labels[best_match_idx]

        # 4. Đem tên nhãn đó đi quét 8 ảnh + link bài viết từ Bing
        images = get_related_images_bing(best_place, num_images=8)
            
        # 5. Trả về kết quả (Đã loại bỏ hoàn toàn place_name)
        return {
            "status": "success",
            "recommendations": [
                {
                    "visual_matches": images
                }
            ]
        }

    except Exception as e:
        return {"error": f"Lỗi xử lý ảnh: {str(e)}"}