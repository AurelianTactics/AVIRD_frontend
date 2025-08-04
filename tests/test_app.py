from app.database import get_db_session
from app.models import IncidentReport

def test_database():
    print("Testing database connection...")
    session = get_db_session()
    try:
        count = session.query(IncidentReport).count()
        print(f"Database has {count} incident reports")
        
        # Get first 3 records to verify data
        samples = session.query(IncidentReport).limit(3).all()
        for i, report in enumerate(samples, 1):
            print(f"  Report {i}: {report.report_id} - {report.reporting_entity}")
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    test_database()
    print("\nYour app is ready! Run: uvicorn app.main:app --reload")