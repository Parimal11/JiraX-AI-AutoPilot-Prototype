
import json
import os
import sqlite3
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import requests
import google.generativeai as genai
from dotenv import load_dotenv


FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")




load_dotenv(dotenv_path=".env.local")

app = FastAPI(title="JiraX AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize DB
def init_db():
    conn = sqlite3.connect('jirax.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS repositories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    owner TEXT,
                    repo_name TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS sprint_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo_id INTEGER,
                    plan_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_db()

# Environment Variables
api_key = os.getenv("GEMINI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


if not api_key:
    print("❌ ERROR: GEMINI_API_KEY not found! Check your .env file.")
else:
    genai.configure(api_key=api_key)
    
    print("✅ Gemini API successfully configured.")

class AskRequest(BaseModel):
    prompt: str
    context: Optional[str] = ""

class Assignment(BaseModel):
    member: str
    issue_titles: List[str]

class SprintPlanResponse(BaseModel):
    sprint_name: str
    assignments: List[Assignment]
    health_score: float
    summary: str

@app.post("/ai/ask")
async def ask_ai(req: AskRequest):
    try:
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content(f"{req.context}\n\nUser: {req.prompt}")
        return {"answer": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ai/sprint-plan", response_model=SprintPlanResponse)
async def create_sprint_plan(data: dict):
    team = data.get("team", [])
    issues = data.get("issues", [])
    
    # 1. Be very explicit in the prompt about the REQUIRED fields
    prompt = f"""
    Create a sprint plan for this team: {team}.
    Issues: {issues}.
    
    You MUST return a JSON object with exactly these keys:
    1. "sprint_name": A creative name for the sprint.
    2. "assignments": A list of objects with "member" and "issue_titles".
    3. "health_score": A number from 0 to 100.
    4. "summary": A brief text summary of the plan.
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    
    # 2. Parse the string into a Python dict
    plan_data = json.loads(response.text)
    
    # 3. Validation fallback: Ensure the AI didn't just return a list
    if isinstance(plan_data, list):
        plan_data = {
            "sprint_name": "Dynamic Sprint",
            "assignments": plan_data, # Put the list into the assignments key
            "health_score": 85.0,
            "summary": "Automatically adjusted plan."
        }
    
    return plan_data 

@app.post("/ai/github/analyze")
async def analyze_repo(req: dict):
    repo_url = req.get("repo_url")
    if not repo_url:
        raise HTTPException(status_code=400, detail="Repo URL required")
    
    # Simple extraction logic
    try:
        parts = repo_url.replace("https://github.com/", "").split("/")
        owner, repo = parts[0], parts[1]
    except:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL")

    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    
    # Fetch issues
    resp = requests.get(f"https://api.github.com/repos/{owner}/{repo}/issues?state=open", headers=headers)
    issues = resp.json()
    
    # Mock Risk Analysis (In real use, pass issues to Gemini)
    return {
        "open_issues": issues,
        "risk_summary": "Low risk. Most issues are UI tweaks.",
        "suggested_sprint": []
    }

@app.post("/ai/vision")
async def analyze_vision(file: UploadFile = File(...)):
    try:
        # Read the file bytes
        image_bytes = await file.read()
        
        # Initialize Gemini 1.5 Flash (supports Vision)
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        
        prompt = (
            "You are a Senior UI/UX QA Engineer. Analyze this screenshot for: "
            "1. Alignment issues. 2. Color contrast problems. 3. Broken UI elements. "
            "4. General UX improvements. Provide a detailed report."
        )

        # Generate content using the image
        response = model.generate_content([
            prompt,
            {"mime_type": file.content_type, "data": image_bytes}
        ])
        
        return {"analysis": response.text}
    except Exception as e:
        print(f"Vision Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
