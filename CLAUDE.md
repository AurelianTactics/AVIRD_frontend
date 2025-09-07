# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AVIRD Frontend is a FastAPI web application for displaying and analyzing NHTSA (National Highway Traffic Safety Administration) incident data related to Autonomous Vehicle Incident Reports Database. The application provides a web interface to explore incident reports with dynamic column handling and entity statistics.

## Development Commands

### Local Development Setup (One-time)
```bash
# Activate virtual environment
avird_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Load data into database
python load_data_flexible.py
```

### Running the Application
```bash
# Local development server
uvicorn app.main:app --reload

# Production server (as configured for deployment)
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Data Loading
```bash
# Load data locally
python load_data_flexible.py

# Load data on deployment (Railway)
python deploy_load_data.py
```

## Architecture

### Core Application Structure
- **`app/main.py`**: Main FastAPI application with routes for data display, entity statistics, and health checks
- **`app/database.py`**: Database configuration supporting both SQLite (local) and PostgreSQL (production)
- **`app/data_loader_flexible.py`**: Dynamic CSV data loader that handles column name cleaning and database insertion

### Database Design
- **Primary Table**: `incident_reports` - contains all NHTSA incident data with 137+ dynamically loaded columns
- **Database Strategy**: SQLite for local development (`avird_data.db`), PostgreSQL for production via `DATABASE_URL` environment variable
- **Column Handling**: Dynamic column discovery supports both SQLite and PostgreSQL with automatic SQL-safe column name cleaning

### Key Features
- **Dynamic Data Loading**: Handles CSV files with any number of columns, automatically cleaning names for SQL compatibility
- **Cross-Database Compatibility**: Seamless switching between SQLite (local) and PostgreSQL (production)
- **Entity Statistics**: Aggregated injury severity statistics by operating entity
- **Responsive Web UI**: HTML templates with dynamic data rendering

### API Endpoints
- **`GET /`**: Main dashboard with data table
- **`GET /api/data`**: JSON API for incident data with configurable limit
- **`GET /entities`**: Entity statistics page
- **`GET /api/entity-stats`**: JSON API for entity injury statistics
- **`GET /health`**: Health check endpoint

### Data Source
The application loads data from `data/nhtsa_sgo/SGO-2021-01_Incident_Reports_ADS.csv` - a comprehensive NHTSA dataset with incident reports for autonomous vehicle systems.

## Deployment

### Local Development
Uses SQLite database (`avird_data.db`) for simple local development without external dependencies.

### Production Deployment (Railway)
- Configured for Railway deployment with PostgreSQL database
- Uses `Procfile`, `railway.json`, and `start.sh` for deployment configuration
- Environment variables: `DATABASE_URL` (automatic), `PORT` (automatic)
- Deploy data loading: Use `python deploy_load_data.py` after deployment

### Key Files for Deployment
- **`requirements.txt`**: Python dependencies (FastAPI, SQLAlchemy, PostgreSQL driver, etc.)
- **`Procfile`**: Heroku-style process definition
- **`railway.json`**: Railway-specific deployment configuration
- **`start.sh`**: Production startup script
- do not be a sycophant
- solutions should work locally and on railway deploy