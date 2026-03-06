import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def check_budget_pacing(destination: str, total_budget: int, spent_so_far: int, days_remaining: int):
    remaining_budget = total_budget - spent_so_far
    csv_path = os.path.join('data', 'places_db.csv')
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {"error": "Không tìm thấy dữ liệu địa điểm."}

    filtered_df = df[df['Destination'].str.contains(destination, case=False, na=False)]
    if filtered_df.empty:
        return {"error": "Chưa có dữ liệu địa điểm."}

    # Kế toán ưu tiên xếp từ RẺ đến MẮC, lấy top 8 rẻ nhất đưa cho AI
    cheap_places_df = filtered_df.sort_values(by='Cost', ascending=True).head(8)
    places_info = "\n".join([f"- {row['Name']} (Giá: {row['Cost']} VNĐ)" for _, row in cheap_places_df.iterrows()])

    # ==========================================
    # PROMPT ÉP KHUÔN JSON
    # ==========================================
    prompt = f"""
    Ngân sách ban đầu: {total_budget}. Đã tiêu: {spent_so_far}. Còn lại: {remaining_budget}. 
    Số ngày còn lại: {days_remaining}.
    
    Kho địa điểm cứu cánh (Rẻ hoặc Miễn phí):
    {places_info}
    
    YÊU CẦU: Chia tiền còn lại cho số ngày. Đánh giá tình hình tiêu tiền.
    BẮT BUỘC TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON SAU (Không lảm nhảm):
    {{
        "financial_status": "Báo động đỏ / Khá căng / An toàn",
        "daily_budget_left": 0,
        "ai_advice": "Lời khuyên từ Kế toán trưởng AI (ngắn gọn, thao túng tâm lý, có emoji)",
        "saved_suggestions": [
            {{
                "name": "Tên địa điểm 1",
                "cost": 0,
                "reason": "Lý do gợi ý"
            }},
            {{
                "name": "Tên địa điểm 2",
                "cost": 0,
                "reason": "Lý do gợi ý"
            }}
        ]
    }}
    """
    
    try:
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2 
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"Lỗi gọi AI: {str(e)}"}