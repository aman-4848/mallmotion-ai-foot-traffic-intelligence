"""
FastAPI Application
Simple API for displaying model predictions and results (no database)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
import pandas as pd
import joblib
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from streamlit_app.utils.data_loader import load_processed_data

app = FastAPI(
    title="Mall Movement Tracking API",
    description="API for displaying predictions and insights from mall movement tracking models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mall Movement Tracking API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "data_info": "/api/data/info",
            "classification_results": "/api/results/classification",
            "clustering_results": "/api/results/clustering",
            "forecasting_results": "/api/results/forecasting"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/data/info")
async def get_data_info():
    """Get information about the processed dataset"""
    try:
        df = load_processed_data()
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/classification")
async def get_classification_results():
    """Get classification model results"""
    try:
        results_file = RESULTS_DIR / "classification" / "metrics.json"
        if not results_file.exists():
            return {"message": "Classification models not trained yet"}
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/clustering")
async def get_clustering_results():
    """Get clustering model results"""
    try:
        results_file = RESULTS_DIR / "clustering" / "silhouette_score.json"
        if not results_file.exists():
            return {"message": "Clustering models not trained yet"}
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/forecasting")
async def get_forecasting_results():
    """Get forecasting model results"""
    try:
        results_file = RESULTS_DIR / "forecasting" / "rmse.json"
        if not results_file.exists():
            return {"message": "Forecasting models not trained yet"}
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/list")
async def list_available_models():
    """List all available trained models"""
    models = {
        "classification": [],
        "clustering": [],
        "forecasting": []
    }
    
    # Check classification models
    classification_dir = MODELS_DIR / "classification"
    if classification_dir.exists():
        models["classification"] = [f.name for f in classification_dir.glob("*.pkl")]
    
    # Check clustering models
    clustering_dir = MODELS_DIR / "clustering"
    if clustering_dir.exists():
        models["clustering"] = [f.name for f in clustering_dir.glob("*.pkl")]
    
    # Check forecasting models
    forecasting_dir = MODELS_DIR / "forecasting"
    if forecasting_dir.exists():
        models["forecasting"] = [f.name for f in forecasting_dir.glob("*.pkl")]
    
    return models
