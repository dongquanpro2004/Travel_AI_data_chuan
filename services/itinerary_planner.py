import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
from datetime import datetime
import json # Thêm thư viện xử lý JSON

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_smart_itinerary(destination: str, num_people: int, budget: int, days: int, preferences: str, start_date: str):
    csv_path = os.path.join('data', 'places_db.csv')
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return '{"error": "Không tìm thấy file dữ liệu."}'

    filtered_df = df[df['Destination'].str.contains(destination, case=False, na=False)]
    if filtered_df.empty:
        return f'{{"error": "Chưa có dữ liệu cho {destination}."}}'

    places_info = ""
    for index, row in filtered_df.iterrows():
        places_info += f"- {row['Name']} (Loại: {row['Category']}, Chi phí: {row['Cost']} VNĐ, Môi trường: {row['Environment']})\n"

    # Xử lý mùa mưa / đám đông (Giữ nguyên logic cực xịn của bạn)
    crowd_warning = ""
    weather_context = ""
    try:
        date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        month = date_obj.month
        day_of_week = date_obj.weekday()
        
        is_weekend = day_of_week >= 5
        is_holiday = (date_obj.day == 30 and month == 4) or (date_obj.day == 1 and month == 5) or (date_obj.day == 2 and month == 9)
                     
        if is_weekend or is_holiday:
            crowd_warning = "Ngày đi là Lễ/Cuối tuần. BẮT BUỘC ưu tiên các điểm 'Khám phá' vào Sáng sớm (7h-9h) để né đám đông."
        else:
            crowd_warning = "Mật độ khách bình thường."
            
        if 5 <= month <= 10:
            weather_context = "Đang là MÙA MƯA. BẮT BUỘC rải đều các địa điểm TRONG NHÀ (Thư giãn, Ẩm thực, Văn hóa) vào các buổi CHIỀU để dự phòng mưa rào."
        else:
            weather_context = "Mùa khô ráo."
    except ValueError:
        pass

    # ==========================================
    # PROMPT ÉP KHUÔN JSON
    # ==========================================
    prompt = f"""
    Dữ liệu: {destination} | Khởi hành: {start_date} | {days} ngày | {num_people} khách | Ngân sách tổng cho cả nhóm: {budget} VNĐ | Sở thích: {preferences}
    Kho địa điểm duyệt sẵn:
    {places_info}
    
    YÊU CẦU:
    1. Lên lịch trình hợp lý, tổng chi phí không vượt quá {budget} VNĐ.
    2. Constraint 1: {crowd_warning}
    3. Constraint 2: {weather_context}
    
    BẮT BUỘC TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON SAU (Không thêm bất kỳ chữ nào khác ở ngoài):
    {{
        "total_cost": 0,
        "general_advice": "1 câu ngắn gọn nhận xét về thời tiết/đám đông",
        "itinerary": [
            {{
                "day": 1,
                "session": "Sáng",
                "place_name": "Tên địa điểm",
                "cost": 150000,
                "reason": "Lý do chọn điểm này (ngắn gọn)"
            }}
        ],
        "luggage_checklist": [
            "Món 1 (Lý do)",
            "Món 2 (Lý do)"
        ]
    }}
    """

    try:
        # CẤU HÌNH ÉP TRẢ VỀ JSON: response_mime_type="application/json"
        model = genai.GenerativeModel('gemini-2.5-flash') 
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2 # Hạ nhiệt độ xuống để AI trả lời logic và cứng nhắc hơn, bớt sáng tạo bậy bạ
            )
        )
        # Load string trả về thành object JSON (Dict trong Python) để API FastAPI trả về cho đẹp
        return json.loads(response.text) 
    except Exception as e:
        return {"error": f"Lỗi gọi AI: {str(e)}"}