import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def dynamic_replan(destination: str, current_budget: int, current_location: str, incident_type: str):
    csv_path = os.path.join('data', 'places_db.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {"error": "Không tìm thấy dữ liệu địa điểm."}

    # Lọc ngân sách và điểm đến
    filtered_df = df[
        (df['Destination'].str.contains(destination, case=False, na=False)) &
        (df['Cost'] <= current_budget)
    ]

    # Lọc không gian nếu là sự cố thời tiết (mưa)
    if "mưa" in incident_type.lower():
        filtered_df = filtered_df[filtered_df['Environment'].str.contains('Trong nhà', case=False, na=False)]

    if filtered_df.empty:
        return {"error": f"Không tìm ra phương án thay thế nào vừa với ngân sách {current_budget} VNĐ."}

    places_info = "\n".join([f"- {row['Name']} (Giá: {row['Cost']} VNĐ, Môi trường: {row['Environment']})" for _, row in filtered_df.iterrows()])

    # ==========================================
    # PROMPT ÉP KHUÔN JSON
    # ==========================================
    prompt = f"""
    Sự cố khẩn cấp: "{incident_type}" tại {destination}.
    Vị trí hiện tại: {current_location}. 
    Ngân sách cho phép: {current_budget} VNĐ.
    
    Kho địa điểm an toàn để thay thế:
    {places_info}
    
    BẮT BUỘC TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON SAU (Không lảm nhảm):
    {{
        "incident_type": "{incident_type}",
        "popup_message": "Câu thông báo an ủi và khuyên đổi lịch trình (cực kỳ ngắn gọn, ấm áp)",
        "suggested_place": {{
            "name": "Tên địa điểm được chọn",
            "cost": 0,
            "reason": "Lý do (VD: gần, trong nhà, rẻ...)"
        }}
    }}
    """
    
    try:
        # Lấy tên model từ file .env để bảo mật (Mặc định dùng flash cho tốc độ nhanh)
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name) 
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"Lỗi gọi AI: {str(e)}"}