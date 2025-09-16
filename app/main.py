from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from .database import engine
from .fault_analysis import create_session, process_user_argument, get_session_state, end_session
from sqlalchemy import text
import os

app = FastAPI(title="AVIRD Data Analyzer")

# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Environment configuration
DEBUG = os.getenv("ENVIRONMENT", "development") == "development"

# Only mount static files if directory exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class StartSessionRequest(BaseModel):
    user_position: str  # "prosecution" or "defense"

class ArgumentRequest(BaseModel):
    argument: str

# Serve robots.txt
@app.get("/robots.txt")
async def robots_txt():
    return HTMLResponse(content="""User-agent: *
Disallow: /

# This is a research prototype
# No crawling or indexing allowed""", media_type="text/plain")

# 404 Error handler
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

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

@app.get("/incident/{report_id}", response_class=HTMLResponse)
async def incident_page(request: Request, report_id: str):
    return templates.TemplateResponse("incident.html", {"request": request, "report_id": report_id})

@app.get("/api/incident/{report_id}")
async def get_incident_detail(report_id: str):
    try:
        with engine.connect() as conn:
            # Get all columns dynamically
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
            
            # Query for specific incident
            query = text(f"SELECT * FROM incident_reports WHERE report_id = :report_id")
            result = conn.execute(query, {"report_id": report_id})
            
            row = result.fetchone()
            if not row:
                return {"error": f"Incident with report ID {report_id} not found"}
            
            # Convert to dict
            incident_data = dict(zip(columns, row))
            
            # Get fault analysis data if available
            fault_data = None
            try:
                fault_query = text("""
                    SELECT fault_version, is_av_at_fault, av_fault_percentage, short_explanation_of_decision, created_at
                    FROM fault_analysis 
                    WHERE report_id = :report_id 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                fault_result = conn.execute(fault_query, {"report_id": report_id})
                fault_row = fault_result.fetchone()
                
                if fault_row:
                    fault_data = {
                        "fault_version": fault_row[0],
                        "is_av_at_fault": fault_row[1],
                        "av_fault_percentage": fault_row[2],
                        "short_explanation_of_decision": fault_row[3],
                        "created_at": fault_row[4]
                    }
            except Exception as fault_error:
                # If fault table doesn't exist or other error, continue without fault data
                print(f"Could not fetch fault data: {fault_error}")
            
            return {
                "incident": incident_data,
                "columns": columns,
                "fault_analysis": fault_data
            }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/incidents/same/{same_incident_id}")
async def get_same_incidents(same_incident_id: str):
    try:
        with engine.connect() as conn:
            query = text("SELECT report_id, same_incident_id FROM incident_reports WHERE same_incident_id = :same_incident_id")
            result = conn.execute(query, {"same_incident_id": same_incident_id})
            
            incidents = []
            for row in result:
                incidents.append({
                    "report_id": row[0],
                    "same_incident_id": row[1]
                })
            
            return {"incidents": incidents}
    except Exception as e:
        return {"error": str(e)}

@app.get("/test")
async def test_endpoint():
    return {"message": "FastAPI is working!"}

# Interactive Fault Analysis Endpoints

@app.post("/api/incident/{report_id}/interactive-fault")
async def start_interactive_fault_session(report_id: str, request: StartSessionRequest):
    """Start a new interactive fault analysis session"""
    try:
        # Validate position
        if request.user_position.lower() not in ["prosecution", "defense"]:
            raise HTTPException(status_code=400, detail="Position must be 'prosecution' or 'defense'")
        
        # Get incident data
        with engine.connect() as conn:
            # Get all columns dynamically
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
            
            # Query for specific incident
            query = text(f"SELECT * FROM incident_reports WHERE report_id = :report_id")
            result = conn.execute(query, {"report_id": report_id})
            
            row = result.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Incident with report ID {report_id} not found")
            
            # Convert to dict
            incident_data = dict(zip(columns, row))
        
        # Create session
        session_id = create_session(report_id, incident_data, request.user_position)
        
        return {
            "success": True,
            "session_id": session_id,
            "user_position": request.user_position,
            "ai_position": "defense" if request.user_position.lower() == "prosecution" else "prosecution"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/incident/{report_id}/interactive-fault/{session_id}/argue")
async def submit_argument(report_id: str, session_id: str, request: ArgumentRequest):
    """Submit user argument and get AI response"""
    try:
        # Validate argument length
        if not request.argument or len(request.argument.strip()) == 0:
            raise HTTPException(status_code=400, detail="Argument cannot be empty")
        
        if len(request.argument) > 1000:
            raise HTTPException(status_code=400, detail="Argument too long (max 1000 characters)")
        
        # Process argument
        result = process_user_argument(session_id, request.argument)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to process argument"))
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/incident/{report_id}/interactive-fault/{session_id}")
async def get_session_status(report_id: str, session_id: str):
    """Get current session state"""
    try:
        session_state = get_session_state(session_id)
        return session_state
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/incident/{report_id}/interactive-fault/{session_id}/end")
async def end_fault_session(report_id: str, session_id: str):
    """End session and get judge decision"""
    try:
        result = end_session(session_id)
        
        if not result.get("success", True):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to end session"))
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))