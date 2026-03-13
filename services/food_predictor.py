import os
import re  # Thêm thư viện re để quét link từ Bing
import torch
import numpy as np
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# KHỞI TẠO CLIP VÀ LOAD TEXT VECTOR DATABASE
# ==========================================
try:
    print("Đang khởi tạo AI Travel Lens (CLIP)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load vector của kho ảnh đồ ăn (Khởi tạo đường dẫn động)
    base_dir = os.path.dirname(os.path.dirname(__file__))
    food_db_vectors = np.load(os.path.join(base_dir, 'data', 'food_image_clip_db.npy'))
    food_db_labels = np.load(os.path.join(base_dir, 'data', 'food_image_labels.npy'))
except Exception as e:
    print(f"Lỗi khi load CLIP model hoặc vector DB món ăn: {e}")
    model = None

SIMILARITY_THRESHOLD = 0.22 

def get_related_images_bing(food_vietnamese_name, num_images=6):
    """
    Cào link ảnh trực tiếp từ Bing Images. 
    Không bị block 403, miễn phí vĩnh viễn và trả về URL xịn.
    """
    search_query = f"món ăn {food_vietnamese_name} Việt Nam ngon thực tế".replace(' ', '+')
    url = f"https://www.bing.com/images/search?q={search_query}"
    
    # Đóng giả làm trình duyệt Chrome
    headers = {
        "LeQuanDat": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        
        # Dùng Regex lôi đầu link ảnh gốc ra từ mã nguồn của Bing
        image_urls = re.findall(r'murl&quot;:&quot;(.*?)&quot;', response.text)
        
        # Xóa các link trùng lặp (nếu có) và cắt đúng số lượng
        clean_urls = list(dict.fromkeys(image_urls))
        return clean_urls[:num_images]
        
    except Exception as e:
        print(f"Lỗi khi cào ảnh từ Bing: {e}")
        return []

def predict_vietnamese_food(img_path):
    if model is None:
        return {"error": "Model CLIP chưa sẵn sàng"}
    
    try:
        # Xử lý ảnh đầu vào bằng CLIP
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            
            # Ép lấy Tensor để tương thích mọi version transformers
            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, 'image_embeds'):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, 'pooler_output'):
                    image_features = image_features.pooler_output
                else:
                    image_features = image_features[0]
                    
            # Chuẩn hóa Vector ảnh
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            client_vec = image_features.cpu().numpy()
            
        # Tính độ tương đồng
        sim_scores = cosine_similarity(client_vec, food_db_vectors)[0]
        best_match_idx = np.argmax(sim_scores)
        max_similarity = sim_scores[best_match_idx]
        
        if max_similarity < SIMILARITY_THRESHOLD:
            return {
                "status": "rejected", 
                "message": "Ảnh này không giống món ăn Việt Nam nào trong dữ liệu. Hãy thử chụp rõ hơn!",
                "confidence": float(max_similarity)
            }
            
        predicted_food_name = food_db_labels[best_match_idx]
        
    except Exception as e:
        return {"error": f"Lỗi xử lý ảnh: {str(e)}"}

    # ==========================================
    # LOGIC: TỪ ĐIỂN MÓN ĂN
    # ==========================================
    search_keywords = {
        'Banh beo': 'bánh bèo', 'Banh bot loc': 'bột lọc', 'Banh can': 'bánh căn', 
        'Banh canh': 'bánh canh', 'Banh chung': 'bánh chưng', 'Banh cuon': 'bánh cuốn', 
        'Banh duc': 'bánh đúc', 'Banh gio': 'bánh giò', 'Banh khot': 'khọt', 
        'Banh mi': 'bánh mì', 'Banh pia': 'bánh pía', 'Banh tet': 'bánh tét', 
        'Banh trang nuong': 'bánh tráng', 'Banh xeo': 'bánh xèo', 'Bun bo Hue': 'bún bò', 
        'Bun dau mam tom': 'bún đậu', 'Bun mam': 'bún mắm', 'Bun rieu': 'bún riêu', 
        'Bun thit nuong': 'bún thịt nướng', 'Ca kho to': 'cá kho', 'Canh chua': 'canh chua', 
        'Cao lau': 'cao lầu', 'Chao long': 'cháo lòng', 'Com tam': 'tấm', 
        'Goi cuon': 'gỏi cuốn', 'Hu tieu': 'hủ tiếu', 'Mi quang': 'mì quảng', 
        'Nem chua': 'nem chua', 'Pho': 'phở', 'Xoi xeo': 'xôi xéo'
    }
    
    vietnamese_name = search_keywords.get(predicted_food_name, predicted_food_name).title()

    food_descriptions = {
        'Banh beo': 'Món bánh bềnh bồng miền Trung, làm từ bột gạo, mỏng và nhỏ, phủ tôm chấy, mỡ hành ăn cùng nước mắm ngọt.',
        'Banh bot loc': 'Đặc sản cố đô Huế làm từ bột sắn với nhân tôm thịt đậm đà, vỏ bánh dai dai sần sật bùng vị mặn ngọt.',
        'Banh can': 'Món bánh nướng khuôn đất nung ở Nam Trung Bộ, vỏ giòn xốp, nhân trứng/hải sản, chấm nước mắm chua ngọt.',
        'Banh canh': 'Món nước sợi to làm từ bột mì/gạo/lọc, nước dùng sệt mặn mà, kèm các nấm, cua, giò heo hay cá lóc.',
        'Banh chung': 'Bánh truyền thống dịp Tết miền Bắc, hình vuông gói lá dong, nếp dẻo thơm nấu chín kỹ bọc nhân đậu xanh thịt mỡ.',
        'Banh cuon': 'Bánh tráng mỏng trên nồi hơi từ bột gạo, cuộn mộc nhĩ thịt băm, rắc hành phi thơm lừng, ăn kèm chả lụa.',
        'Banh duc': 'Món quà quê dân dã dẻo quánh, xắn ra từng mảng chấm tương bần hoặc chan nước mắm thịt băm mộc nhĩ nóng hổi.',
        'Banh gio': 'Bánh bột tẻ ngâm nước tro bọc lá chuối, nhân thịt băm mộc nhĩ nêm tiêu thơm lừng luộc chín mềm tan trong miệng.',
        'Banh khot': 'Bánh nướng khuôn nhỏ miền biển mặn, vỏ ngoài giòn rụm màu nghệ, nhân tôm tươi nguyên con, đẫm nước mắm chua ngọt.',
        'Banh mi': 'Món ăn đường phố nổi tiếng toàn cầu vỏ giòn ruột xốp, nhân kẹp pate, thịt nguội, chả lụa, bơ béo và dưa chua.',
        'Banh pia': 'Đặc sản Sóc Trăng với lớp vỏ ngàn lớp mỏng manh bọc lấy nhân sầu riêng, đậu xanh và lòng đỏ trứng muối bùi béo.',
        'Banh tet': 'Phiên bản phương Nam của bánh chưng có hình trụ dài, gói lá chuối cuốn chặt bằng lạt nếp dẻo thơm đậu xanh thịt mỡ.',
        'Banh trang nuong': 'Được mệnh danh là "Pizza Việt Nam" nướng xèo xèo trên than hồng cùng trứng cút, xúc xích, ruốc tôm và phô mai.',
        'Banh xeo': 'Món bánh chiên chảo lớn đổ xèo xèo màu vàng ươm, vỏ giòn dụm bọc lấy giá, tôm, thịt băm ăn cùng rau rừng.',
        'Bun bo Hue': 'Đặc sản vương giả với nước dùng cay nồng hương ruốc sả, vắt bún to đùng, kèm bắp bò, chả cua và móng heo.',
        'Bun dau mam tom': 'Món ăn gây nghiện miền Bắc với mẹt bún lá, đậu hũ chiên giòn, thịt luộc, chả cốm dùng mắm tôm đánh bọt.',
        'Bun mam': 'Đặc sản miền Tây Nam Bộ nước dùng đặc sánh nấu từ mắm cá linh, ngập ngụa hải sản tôm mực và heo quay.',
        'Bun rieu': 'Món bún mộc mạc thanh mát nước dùng chua dịu vị cà dom, nổi bật với tảng riêu cua đồng gạch xốp nở bung.',
        'Bun thit nuong': 'Bún lạnh miền Nam thanh mát trộn cùng thịt heo nướng than hoa sả ớt, chả giò giòn, rau thơm và đậu phộng nêm.',
        'Ca kho to': 'Món mặn đưa cơm kinh điển, cá tẩm ướp kho keo rắc tiêu quẹt sền sệt trong tộ đất, thấm đẫm mắm đường quyện ngọt.',
        'Canh chua': 'Món canh giải nhiệt trứ danh mát mẻ với vị chua từ me, ngọt từ dứa dưa và cá béo, cùng các loại rau ngò ôm.',
        'Cao lau': 'Đặc sản phố cổ Hội An sợi mì vàng ươm ngâm nước tro, xá xíu mềm mại, tóp mỡ chiên giòn rụm và ít nước sốt.',
        'Chao long': 'Món cháo bình dân nấu bằng nước luộc lòng sánh mịn, ăn kèm dồi trường, gan, phèo luộc chấm mắm tôm pha nêm.',
        'Com tam': 'Đặc sản sinh viên nức danh Sài Gòn rưới mỡ hành, sườn nướng than hoa cắn ngập răng, ăn kèm bì chả trứng ốp la.',
        'Goi cuon': 'Món ăn vặt thanh đạm tươi mát với tôm thịt, bún, rau thơm cuộn gọn trong bánh tráng chấm tương đậu phộng sệt sệt.',
        'Hu tieu': 'Món sợi xương túy Nam Bộ nước dùng hầm xương ống trong veo ngọt lịm, ngập tràn hải sản, lòng heo và hành phi.',
        'Mi quang': 'Món mì quê hương xứ Quảng nước dùng sâm sấp đậm đà vị nghệ tươi, tôm thịt đậm vị cắn kèm bánh tráng nướng.',
        'Nem chua': 'Món nhậu hấp dẫn lên men chua tự nhiên từ thịt mông sấn nhuyễn trộn bì heo, tỏi ớt lá ổi bọc lá chuối vuông vức.',
        'Pho': 'Quốc hồn quốc túy lừng lam thế giới, nước hầm xương bò thanh tao vị thảo quả rễ ngò, bánh phở mướt và bò tái.',
        'Xoi xeo': 'Gói xôi ăn sáng tuổi thơ Hà Nội màu nghệ tươi roi rói, phủ đậu xanh xéo nhuyễn tơi xốp, hành phi và mỡ gà thơm nức.'
    }

    # Dùng Bing để lấy mảng 6 link ảnh siêu xịn
    lens_images = get_related_images_bing(vietnamese_name, num_images=6)

    # ==========================================
    # TRẢ VỀ JSON
    # ==========================================
    response_data = {
        "status": "success",
        "food_name": predicted_food_name,
        "vietnamese_name": vietnamese_name,
        "description": food_descriptions.get(predicted_food_name, "Món ăn đặc sắc của ẩm thực Việt Nam."),
        "confidence": float(max_similarity),
        "visual_matches": lens_images 
    }
        
    return response_data