# Local Development Setup

## Initial Setup (One-time only)
1. Activate virtual environment:
   ```
   avird_env\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Load data into database:
   ```
   python load_data.py
   ```

## Running Locally (Every time)
1. Activate environment: `avird_env\Scripts\activate`
2. Start the server: `uvicorn app.main:app --reload`
3. Open browser to: `http://localhost:8000`
4. Click "Load Data" button to view incident reports

## Database
- Uses SQLite for local development (avird_data.db)
- Production uses PostgreSQL via DATABASE_URL environment variable