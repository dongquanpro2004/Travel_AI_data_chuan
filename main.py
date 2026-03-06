from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os

from services.food_predictor import predict_vietnamese_food
from services.itinerary_planner import generate_smart_itinerary
from services.replanner import dynamic_replan
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
    total_budget: int
    spent_so_far: int
    days_remaining: int

# --- API 1: NHẬN DIỆN MÓN ĂN ---
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

# --- API 2: LÊN LỊCH TRÌNH THÔNG MINH (TRƯỚC CHUYẾN ĐI) ---
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

# API 3
@app.post("/api/dynamic-replan/")
async def dynamic_replan_endpoint(request: ReplanRequest):
    alternative_result = dynamic_replan(
        destination=request.destination,
        current_budget=request.current_budget,
        current_location=request.current_location,
        incident_type=request.incident_type
    )
    
    return {
        "status": "success",
        "incident": request.incident_type,
        "suggestion": alternative_result
    }

# API 4: CẢNH BÁO TỐC ĐỘ ĐỐT TIỀN (IDEA 6)
@app.post("/api/check-budget/")
async def check_budget_endpoint(request: BudgetRequest):
    budget_advice = check_budget_pacing(
        destination=request.destination,
        total_budget=request.total_budget,
        spent_so_far=request.spent_so_far,
        days_remaining=request.days_remaining
    )
    
    return {
        "status": "success",
        "action": "budget_pacing",
        "advice": budget_advice
    }

# API 5 : gợi ý theo vibe
@app.post("/api/search_vibe")
async def api_search_vibe(file: UploadFile = File(...)):
    img_bytes = await file.read()
    places = search_vibe(img_bytes)
    return {"status": "success", "suggested_places": places}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)