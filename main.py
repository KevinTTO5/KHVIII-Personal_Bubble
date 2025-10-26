# main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RiskEvent(BaseModel):
    ts: float
    track_id: int
    risk_type: str
    risk_score: float

@app.post("/events/risk")
def post_risk(event: RiskEvent):
    print(event)
    return {"status": "received", "risk": event.risk_type}

@app.get("/")
def root():
    return {"message": "SafetyBubble API is alive!"}
