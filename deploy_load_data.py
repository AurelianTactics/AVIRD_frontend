"""
Data loading script for Railway deployment
Run this after deploying to Railway to load your CSV data
"""
import os
from app.data_loader_flexible import load_data_flexible

def deploy_load_data():
    """Load data in production environment"""
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not found")
        print("Make sure Railway PostgreSQL service is connected")
        return
    
    if "sqlite" in database_url.lower():
        print("WARNING: Using SQLite database. For production, connect PostgreSQL service in Railway.")
    else:
        print(f"Using PostgreSQL database: {database_url[:50]}...")
    
    print("Loading data for production deployment...")
    load_data_flexible()
    print("Production data loading complete!")

if __name__ == "__main__":
    deploy_load_data()