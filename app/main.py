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
            # Get all columns dynamically (works for both SQLite and PostgreSQL)
            if "sqlite" in str(engine.url):
                columns_result = conn.execute(text("PRAGMA table_info(incident_reports)"))
                columns = [row[1] for row in columns_result.fetchall()]
            else:
                # PostgreSQL
                columns_result = conn.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'incident_reports'
                    ORDER BY ordinal_position
                """))
                columns = [row[0] for row in columns_result.fetchall()]
            
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

@app.get("/entities", response_class=HTMLResponse)
async def entities_page(request: Request):
    return templates.TemplateResponse("entities.html", {"request": request})

@app.get("/api/entity-stats")
async def get_entity_stats():
    try:
        with engine.connect() as conn:
            # Query to get entity statistics similar to the old repo
            query = text("""
                SELECT 
                    operating_entity as entity,
                    COUNT(*) as total_unique_reports,
                    SUM(CASE WHEN highest_injury_severity_alleged = 'Fatality' THEN 1 ELSE 0 END) as fatality_count,
                    SUM(CASE WHEN highest_injury_severity_alleged = 'Serious' THEN 1 ELSE 0 END) as serious_count,
                    SUM(CASE WHEN highest_injury_severity_alleged = 'Moderate' THEN 1 ELSE 0 END) as moderate_count,
                    SUM(CASE WHEN highest_injury_severity_alleged = 'Minor' THEN 1 ELSE 0 END) as minor_count,
                    SUM(CASE WHEN highest_injury_severity_alleged = 'No injuries' THEN 1 ELSE 0 END) as no_injuries_count,
                    SUM(CASE WHEN highest_injury_severity_alleged NOT IN ('Fatality', 'Serious', 'Moderate', 'Minor', 'No injuries') OR highest_injury_severity_alleged IS NULL THEN 1 ELSE 0 END) as other_count
                FROM incident_reports 
                WHERE operating_entity IS NOT NULL 
                GROUP BY operating_entity
                ORDER BY total_unique_reports DESC
            """)
            
            result = conn.execute(query)
            entities = []
            
            for row in result:
                total = row.total_unique_reports
                entity_data = {
                    "entity": row.entity,
                    "total_unique_reports": total,
                    "injuries": {
                        "fatality": {
                            "count": row.fatality_count,
                            "percentage": round((row.fatality_count / total * 100), 1) if total > 0 else 0
                        },
                        "serious": {
                            "count": row.serious_count,
                            "percentage": round((row.serious_count / total * 100), 1) if total > 0 else 0
                        },
                        "moderate": {
                            "count": row.moderate_count,
                            "percentage": round((row.moderate_count / total * 100), 1) if total > 0 else 0
                        },
                        "minor": {
                            "count": row.minor_count,
                            "percentage": round((row.minor_count / total * 100), 1) if total > 0 else 0
                        },
                        "no_injuries": {
                            "count": row.no_injuries_count,
                            "percentage": round((row.no_injuries_count / total * 100), 1) if total > 0 else 0
                        },
                        "other": {
                            "count": row.other_count,
                            "percentage": round((row.other_count / total * 100), 1) if total > 0 else 0
                        }
                    }
                }
                entities.append(entity_data)
            
            return {"entities": entities}
            
    except Exception as e:
        return {"error": str(e)}

@app.get("/test")
async def test_endpoint():
    return {"message": "FastAPI is working!"}