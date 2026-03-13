from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Any
import shutil
import os

from services.food_predictor import predict_vietnamese_food
from services.itinerary_planner import generate_smart_itinerary
from services.budget_pacer import check_budget_pacing
from services.vibe_searcher import search_vibe

app = FastAPI(title="Travel AI API", version="1.1.0")

# --- MODEL DATA ĐẦU VÀO ---
class ItineraryRequest(BaseModel):
    destination: str
    num_people: int
    budget: int
    days: int
    preferences: str
    start_date: str

class ReplanRequest(BaseModel):
    destination: str
    current_budget: int
    current_location: str
    incident_type: str  # THÊM MỚI: Truyền loại sự cố vào (Mưa, kẹt xe, đóng cửa...) 

class BudgetRequest(BaseModel):
    destination: str
    current_money: int
    days_remaining: int
    current_plan: Optional[dict] = None



# --- API 1: LÊN LỊCH TRÌNH THÔNG MINH (TRƯỚC CHUYẾN ĐI) ---
@app.post("/api/plan-itinerary/")
async def plan_itinerary_endpoint(request: ItineraryRequest):
    itinerary_result = generate_smart_itinerary(
        destination=request.destination,
        num_people=request.num_people,
        budget=request.budget,
        days=request.days,
        preferences=request.preferences,
        start_date=request.start_date
    )
    return {
        "status": "success",
        "destination": request.destination,
        "start_date": request.start_date,
        "itinerary": itinerary_result
    }



# API 2: CẢNH BÁO TỐC ĐỘ ĐỐT TIỀN (IDEA 6)
@app.post("/api/check-budget/")
async def check_budget_endpoint(request: BudgetRequest):
    budget_advice = check_budget_pacing(
        destination=request.destination,
        current_money=request.current_money,
        days_remaining=request.days_remaining,
        current_plan=request.current_plan
    )
    
    return {
        "status": "success",
        "action": "budget_pacing",
        "advice": budget_advice
    }



# --- API 4: NHẬN DIỆN MÓN ĂN ---
@app.post("/api/predict-food/")
async def predict_food_endpoint(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        prediction_result = predict_vietnamese_food(temp_file_path)
        return prediction_result
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# API 5 : gợi ý theo vibe
@app.post("/api/search_vibe")
async def api_search_vibe(file: UploadFile = File(...)):
    img_bytes = await file.read()
    places = search_vibe(img_bytes)
    return {"status": "success", "suggested_places": places}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)