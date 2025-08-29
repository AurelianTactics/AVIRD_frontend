import pandas as pd
from sqlalchemy import create_engine, text
from .database import engine
import os

def clean_column_name(col_name):
    """Clean column names for SQL compatibility"""
    return (col_name.lower()
            .replace(' ', '_')
            .replace('/', '_')
            .replace('-', '_')
            .replace('?', '')
            .replace('.', '')
            .replace('(', '')
            .replace(')', '')
            .replace('#', 'num')
            .replace('%', 'pct')
            .replace('&', 'and')
            .replace(':', '')
            .replace(';', '')
            .replace(',', '')
            .replace("'", '')
            .replace('"', '')
            .replace('__', '_')
            .strip('_'))

def load_data_flexible():
    """Load all CSV columns dynamically into database"""
    # Use public URL if available (for railway run), otherwise use the configured engine
    database_url = os.getenv("DATABASE_PUBLIC_URL")
    if database_url:
        print(f"Using PUBLIC database URL: {database_url[:50]}...")
        local_engine = create_engine(database_url)
    else:
        print("Using default database engine...")
        local_engine = engine
    
    print("Creating database tables...")
    from .database import Base
    Base.metadata.create_all(bind=local_engine)
    
    print("Loading CSV data...")
    csv_path = "data/nhtsa_sgo/SGO-2021-01_Incident_Reports_ADS.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Load the full CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Clean column names for SQL and handle duplicates
    original_columns = df.columns.tolist()
    cleaned_columns = []
    seen_columns = set()
    
    for col in original_columns:
        clean_col = clean_column_name(col)
        if clean_col in seen_columns:
            # Add suffix for duplicates
            counter = 2
            while f"{clean_col}_{counter}" in seen_columns:
                counter += 1
            clean_col = f"{clean_col}_{counter}"
        
        cleaned_columns.append(clean_col)
        seen_columns.add(clean_col)
    
    df.columns = cleaned_columns
    
    print("Column mapping:")
    for orig, clean in zip(original_columns, cleaned_columns):
        print(f"  '{orig}' -> '{clean}'")
    
    # Replace empty strings and 'nan' with None for proper NULL handling
    df = df.replace(['', 'nan', 'NaN'], None)
    
    # Insert everything directly using pandas to_sql
    # Use smaller chunks to avoid SQLite variable limit
    print(f"Creating table and inserting {len(df)} rows...")
    df.to_sql('incident_reports', local_engine, if_exists='replace', index=False, 
              chunksize=50)
    
    print("Data loading complete!")
    print(f"Table created with columns: {list(df.columns)}")
    
    # Show a sample of what was inserted
    with local_engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) as count FROM incident_reports"))
        count = result.fetchone()[0]
        print(f"Verification: {count} rows inserted successfully")

if __name__ == "__main__":
    load_data_flexible()