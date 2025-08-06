# Railway Deployment Guide

## Prerequisites
1. GitHub account with your repo pushed
2. Railway account (free at railway.app)

## Deployment Steps

### 1. Connect to Railway
1. Go to [railway.app](https://railway.app)
2. Sign up/login with GitHub
3. Click "New Project"
4. Select "Deploy from GitHub repo"
5. Choose your `AVIRD_frontend` repository

### 2. Add PostgreSQL Database
1. In your Railway project dashboard
2. Click "New Service" → "Database" → "PostgreSQL"
3. Railway will automatically set the `DATABASE_URL` environment variable

### 3. Deploy Application
1. Railway will automatically detect Python and install dependencies
2. The app will deploy using the settings in `nixpacks.toml`
3. You'll get a public URL like `https://your-app.railway.app`

### 4. Load Your Data
After deployment:
1. In Railway dashboard, go to your service
2. Click "Settings" → "Variables"
3. Add a new deployment with:
   ```bash
   python deploy_load_data.py
   ```
   Or use Railway's CLI to run:
   ```bash
   railway run python deploy_load_data.py
   ```

### 5. Visit Your App
- Your app will be live at the Railway-provided URL
- Both the main page and `/api/data` should work
- All 137 columns of your NHTSA data will be available

## Cost
- **Free tier**: Includes PostgreSQL database and hosting
- No credit card required for basic usage
- Much simpler than Heroku's paid tiers

## Environment Variables
Railway automatically provides:
- `DATABASE_URL` - PostgreSQL connection string
- `PORT` - Application port
- All other variables can be added in Settings → Variables