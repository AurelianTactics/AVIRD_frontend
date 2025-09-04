#!/usr/bin/env python3
"""
Create fault_analysis table for both SQLite (local) and PostgreSQL (Railway)
Uses the same database configuration as the main app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import create_engine, text

def create_fault_analysis_table():
    """Create the fault_analysis table compatible with both SQLite and PostgreSQL"""
    
    # Get database URL - try public URL first for Railway external access
    database_url = os.getenv("DATABASE_PUBLIC_URL") or os.getenv("DATABASE_URL", "sqlite:///./avird_data.db")
    engine = create_engine(database_url)
    
    # Check if we're using SQLite or PostgreSQL
    is_sqlite = "sqlite" in str(engine.url)
    
    if is_sqlite:
        # SQLite version
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS fault_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT NOT NULL,
            fault_version TEXT NOT NULL,
            is_av_at_fault BOOLEAN,
            av_fault_percentage REAL CHECK(av_fault_percentage >= 0 AND av_fault_percentage <= 1),
            short_explanation_of_decision TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_id, fault_version)
        );
        """
    else:
        # PostgreSQL version  
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS fault_analysis (
            id SERIAL PRIMARY KEY,
            report_id VARCHAR(50) NOT NULL,
            fault_version VARCHAR(50) NOT NULL,
            is_av_at_fault BOOLEAN,
            av_fault_percentage DECIMAL(5,4) CHECK(av_fault_percentage >= 0 AND av_fault_percentage <= 1),
            short_explanation_of_decision TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(report_id, fault_version)
        );
        """
    
    try:
        with engine.connect() as conn:
            # Create the table
            conn.execute(text(create_table_sql))
            conn.commit()
            
            # Verify table was created by checking if we can query it
            result = conn.execute(text("SELECT COUNT(*) FROM fault_analysis"))
            count = result.fetchone()[0]
            
            db_type = "SQLite" if is_sqlite else "PostgreSQL"
            print(f"[SUCCESS] Successfully created fault_analysis table in {db_type}")
            print(f"   Current records: {count}")
            print(f"   Database URL: {str(engine.url)}")
            
    except Exception as e:
        print(f"[ERROR] Error creating fault_analysis table: {e}")
        raise

def show_table_info():
    """Display information about the fault_analysis table"""
    # Get database URL and create engine
    database_url = os.getenv("DATABASE_PUBLIC_URL") or os.getenv("DATABASE_URL", "sqlite:///./avird_data.db")
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as conn:
            # Get table schema info
            if "sqlite" in str(engine.url):
                schema_result = conn.execute(text("PRAGMA table_info(fault_analysis)"))
                print("\nTable Schema (SQLite):")
                for row in schema_result:
                    print(f"   {row[1]} {row[2]} {'NOT NULL' if row[3] else ''} {'PRIMARY KEY' if row[5] else ''}")
            else:
                schema_result = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = 'fault_analysis'
                    ORDER BY ordinal_position
                """))
                print("\nTable Schema (PostgreSQL):")
                for row in schema_result:
                    nullable = "NULL" if row[2] == "YES" else "NOT NULL"
                    default = f" DEFAULT {row[3]}" if row[3] else ""
                    print(f"   {row[0]} {row[1]} {nullable}{default}")
                    
    except Exception as e:
        print(f"[WARNING] Could not retrieve table info: {e}")

if __name__ == "__main__":
    # Get database URL for display
    database_url = os.getenv("DATABASE_PUBLIC_URL") or os.getenv("DATABASE_URL", "sqlite:///./avird_data.db")
    print("Setting up fault_analysis table...")
    print(f"Using database: {database_url}")
    
    create_fault_analysis_table()
    show_table_info()
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("   1. Run your get_basic_fault.py script to populate data")
    print("   2. Test with: python -c \"from app.database import engine; from sqlalchemy import text; conn = engine.connect(); print(conn.execute(text('SELECT COUNT(*) FROM fault_analysis')).fetchone()[0])\"")