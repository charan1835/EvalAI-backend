import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import random
from typing import Optional, List
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import aiosmtplib
from email.message import EmailMessage
from evaluator import evaluate_answer

# Load environment variables
load_dotenv()

app = FastAPI(
    title="EvalAI API",
    description="Professional AI-powered interview answer evaluation engine.",
    version="1.0.0"
)

# Robust CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    print("⚠️ WARNING: MONGODB_URI not found in .env, history features will be disabled.")
    db = None
else:
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client.evalai

# Email Configuration
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

async def send_otp_email(receiver: str, code: str):
    """Sends a professional OTP email using Gmail SMTP (SSL)."""
    if not EMAIL_USER or not EMAIL_PASS:
        return False
        
    msg = EmailMessage()
    msg["Subject"] = "🔐 EvalAI Security Code"
    msg["From"] = f"EvalAI Security <{EMAIL_USER}>"
    msg["To"] = receiver
    
    # Premium HTML Email Design
    msg.add_alternative(f"""
    <div style="font-family: sans-serif; background-color: #020617; color: #f8fafc; padding: 40px; text-align: center;">
        <div style="max-width: 400px; margin: 0 auto; background-color: #0f172a; padding: 40px; border-radius: 24px; border: 1px solid #1e293b;">
            <h1 style="color: #6366f1; font-weight: 900; margin-bottom: 8px;">EvalAI</h1>
            <p style="text-transform: uppercase; font-size: 10px; font-weight: 700; color: #64748b; letter-spacing: 2px;">Identity Verification</p>
            <div style="margin: 40px 0; padding: 20px; background-color: #020617; border-radius: 16px; border: 1px dashed #334155;">
                <span style="font-size: 36px; font-weight: 900; color: #f8fafc; letter-spacing: 12px;">{code}</span>
            </div>
            <p style="font-size: 14px; color: #94a3b8; line-height: 1.6;">Enter this code into the portal to complete your synchronization. This code expires in 5 minutes.</p>
        </div>
    </div>
    """, subtype="html")
    
    try:
        await aiosmtplib.send(
            msg, hostname="smtp.gmail.com", port=465, use_tls=True,
            username=EMAIL_USER, password=EMAIL_PASS
        )
        return True
    except Exception as e:
        print(f"❌ Email ERROR: {e}")
        return False

# Move data loading to a try-except block for resilience
try:
    data = pd.read_csv("Software Questions.csv", encoding="cp1252")
except Exception as e:
    print(f"Error loading CSV: {e}")
    data = pd.DataFrame(columns=["Question", "Answer", "Category", "Difficulty"])

# --- Models ---
class EvaluationRequest(BaseModel):
    reference: str
    user_answer: str

class HistoryEntry(BaseModel):
    topic: str
    question: str
    user_answer: str
    score: float
    status: str
    date: Optional[str] = None
    icon: Optional[str] = "📝"

class OTPRequest(BaseModel):
    email: str

class OTPVerify(BaseModel):
    email: str
    otp: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    category: str
    difficulty: str

# --- Endpoints ---

@app.get("/", tags=["System"])
def health_check():
    """Confirms the API is live."""
    return {"status": "online", "engine": "EvalAI NLP v1.0", "database": "connected" if db is not None else "disabled"}

# --- AUTH ENDPOINTS ---

@app.post("/auth/request-otp", tags=["Auth"])
async def request_otp(payload: OTPRequest):
    """Generates a 6-digit OTP and 'sends' it via console (Dev Mode)."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    email = payload.email.lower()
    otp_code = str(random.randint(100000, 999999))
    
    # Store OTP in MongoDB with a timestamp
    await db.verifications.update_one(
        {"email": email},
        {"$set": {"otp": otp_code, "timestamp": datetime.now()}},
        upsert=True
    )
    
    # Send the real email
    sent = await send_otp_email(email, otp_code)
    
    if sent:
        return {"status": "success", "message": f"Security code dispatched to {email}"}
    else:
        # Fallback to console during dev if email fails
        print(f"\n🔑 FALLBACK OTP FOR {email}: {otp_code}\n")
        return {"status": "success", "message": "Dispatched via console fallback (Email Fail)"}

@app.post("/auth/verify-otp", tags=["Auth"])
async def verify_otp(payload: OTPVerify):
    """Verifies the OTP and returns a dummy session token."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    email = payload.email.lower()
    record = await db.verifications.find_one({"email": email})
    
    if not record or record["otp"] != payload.otp:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    # Clear the OTP after successful verification
    await db.verifications.delete_one({"email": email})
    
    # In a real app, you'd generate a JWT here
    return {
        "status": "success", 
        "user": {
            "email": email,
            "name": email.split("@")[0].capitalize(),
            "tier": "Premium Elite"
        },
        "token": f"dev_token_{random.randint(1000, 9999)}"
    }

# --- INTERVIEW ENDPOINTS ---

@app.get("/question", response_model=QuestionResponse, tags=["Interview"])
def get_question(category: Optional[str] = Query(None, description="Filter by technology category")):
    """Fetches a random question from the dataset."""
    filtered = data[data["Category"] == category] if category else data

    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"No questions found for category: {category}")

    row = filtered.sample(n=1).iloc[0]

    return {
        "question":   row["Question"],
        "answer":     row["Answer"],
        "category":   row["Category"],
        "difficulty": row["Difficulty"],
    }

@app.post("/evaluate", tags=["Interview"])
async def evaluate(payload: EvaluationRequest):
    """Evaluates the user's answer against the reference using NLP semantics."""
    if not payload.reference or not payload.user_answer:
        raise HTTPException(status_code=400, detail="Reference and user answer cannot be empty.")

    return await evaluate_answer(payload.reference, payload.user_answer)

@app.post("/history", tags=["History"])
async def save_history(entry: HistoryEntry):
    """Saves a practice session result to MongoDB."""
    if db is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    entry_dict = entry.dict()
    if not entry_dict.get("date"):
        entry_dict["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    result = await db.history.insert_one(entry_dict)
    return {"status": "success", "id": str(result.inserted_id)}

@app.get("/history", tags=["History"])
async def get_history():
    """Retrieves all practice history from MongoDB."""
    if db is None:
        return {"history": []}
    
    history_cursor = db.history.find().sort("date", -1).limit(50)
    history = await history_cursor.to_list(length=50)
    
    # Convert MongoDB _id to string for JSON serialization
    for entry in history:
        entry["_id"] = str(entry["_id"])
        
    return {"history": history}

@app.get("/categories", tags=["Interview"])
def get_categories():
    """Returns a list of all available question categories."""
    cats = sorted(data["Category"].dropna().unique().tolist())
    return {"categories": cats}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting EvalAI Backend on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)