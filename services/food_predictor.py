import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH') 

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Lỗi khi load model: {e}")
    model = None

CONFIDENCE_THRESHOLD = 0.3

def predict_vietnamese_food(img_path):
    if model is None:
        return {"error": "Model chưa sẵn sàng"}
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    max_probability = np.max(predictions[0])
    
    if max_probability < CONFIDENCE_THRESHOLD:
        return {
            "status": "rejected", 
            "message": "Ảnh này có vẻ không phải là món ăn Việt Nam mà hệ thống biết. Vui lòng thử lại!",
            "confidence": float(max_probability)
        }
    
    predicted_class_index = np.argmax(predictions[0])
    class_names = [
        'Banh beo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung', 
        'Banh cuon', 'Banh duc', 'Banh gio', 'Banh khot', 'Banh mi', 
        'Banh pia', 'Banh tet', 'Banh trang nuong', 'Banh xeo', 'Bun bo Hue', 
        'Bun dau mam tom', 'Bun mam', 'Bun rieu', 'Bun thit nuong', 'Ca kho to', 
        'Canh chua', 'Cao lau', 'Chao long', 'Com tam', 'Goi cuon', 
        'Hu tieu', 'Mi quang', 'Nem chua', 'Pho', 'Xoi xeo'
    ]
    predicted_food_name = class_names[predicted_class_index]
    
    # ==========================================
    # LOGIC: TỪ ĐIỂN 30 MÓN ĂN VÀ TÌM KIẾM DATABASE
    # ==========================================
    suggested_places = []
    try:
        csv_path = os.path.join('data', 'places_db.csv')
        df = pd.read_csv(csv_path)
        
        # Ánh xạ tên tiếng Anh không dấu thành từ khóa tiếng Việt chuẩn nhất để dò tìm
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
        
        keyword = search_keywords.get(predicted_food_name, predicted_food_name)
        
        # Lọc database tìm quán có bán món này (tìm trong cột Name)
        filtered_df = df[df['Name'].str.contains(keyword, case=False, na=False)]
        
        for _, row in filtered_df.head(2).iterrows(): 
            suggested_places.append({
                "name": row['Name'],
                "destination": row['Destination'],
                "cost": row['Cost'],
                "address": row.get('Address', 'Đang cập nhật') # Lấy thêm địa chỉ nếu có
            })
            
    except Exception as e:
        print("Lỗi dò database:", e)

    # ==========================================
    # LOGIC: THÊM MÔ TẢ MÓN ĂN
    # ==========================================
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

    # ==========================================
    # TRẢ VỀ KẾT QUẢ JSON CHUẨN MỰC
    # ==========================================
    response_data = {
        "status": "success",
        "food_name": predicted_food_name,
        "vietnamese_name": search_keywords.get(predicted_food_name, predicted_food_name).title(),
        "description": food_descriptions.get(predicted_food_name, "Món ăn đặc sắc của ẩm thực Việt Nam."),
        "confidence": float(max_probability),
        "suggestions": suggested_places
    }
    
    # Nếu nhận diện được món nhưng database chưa có quán nào bán
    if not suggested_places:
        response_data["notice"] = "Hệ thống nhận diện thành công, nhưng hiện tại chưa có quán nào bán món này trong cẩm nang của chúng tôi."
        
    return response_data