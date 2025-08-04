from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .database import engine
from sqlalchemy import text

app = FastAPI(title="AVIRD Data Analyzer")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/data")
async def get_data(limit: int = 10):
    try:
        with engine.connect() as conn:
            # Get all columns dynamically
            columns_result = conn.execute(text("PRAGMA table_info(incident_reports)"))
            columns = [row[1] for row in columns_result.fetchall()]
            
            # Query the data
            query = text(f"SELECT * FROM incident_reports LIMIT :limit")
            result = conn.execute(query, {"limit": limit})
            
            # Convert to list of dicts
            data = []
            for row in result:
                row_dict = dict(zip(columns, row))
                # Truncate long text fields for display
                for key, value in row_dict.items():
                    if isinstance(value, str) and len(value) > 200:
                        row_dict[key] = value[:200] + "..."
                data.append(row_dict)
            
            return {
                "total_columns": len(columns),
                "columns": columns,
                "data": data
            }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/test")
async def test_endpoint():
    return {"message": "FastAPI is working!"}