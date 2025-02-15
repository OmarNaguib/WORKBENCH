from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import tempfile
import os
from pathlib import Path
from llm_utils import LLMFilter, SemanticFilter, Frameworker, Classifier, DataWizard, ComparativeAnalyzer, CompetitiveAnalyzer
from typing import List, Optional
import json
from pydantic import BaseModel
import numpy as np

app = FastAPI(
    title="Social Media Analysis Dashboard",
    description="API for analyzing social media data using LLMs",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Request Models
class FilterRequest(BaseModel):
    file: str
    criteria: str
    use_semantic: bool = True
    batch_size: int = 300

class FrameworkerRequest(BaseModel):
    file: str
    target_classes: Optional[int] = None
    context: Optional[str] = None

class ClassifierRequest(BaseModel):
    file: str
    classes: List[str]
    is_multi_class: bool = False

class WizardRequest(BaseModel):
    file: str
    question: str
    use_semantic: bool = True

class ComparativeRequest(BaseModel):
    file1: str
    file2: str
    description1: str
    description2: str
    question: str
    use_semantic: bool = True
    is_competitive: bool = False

def clean_for_json(obj):
    """Clean data to ensure JSON compatibility."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    return obj

@app.get("/api/files")
async def list_files():
    """List all available Excel files in the data directory."""
    files = []
    for file in DATA_DIR.glob("*.xlsx"):
        print(file.name)
        files.append({
            "name": file.name,
            "path": str(file),
            "size": file.stat().st_size,
            "modified": file.stat().st_mtime
        })
    return {"files": files}

@app.post("/api/filter")
async def filter_data(request: FilterRequest):
    """
    Filter Excel data based on semantic or LLM analysis.
    
    Parameters:
    - file: Name of the Excel file in the data directory
    - criteria: Text describing what kind of content to filter for
    - use_semantic: Whether to use semantic filter (True) or LLM filter (False)
    - batch_size: Number of comments to process in each batch
    """
    file_path = DATA_DIR / request.file
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        if 'Post/comments' not in df.columns:
            raise HTTPException(status_code=400, detail="Excel file must contain a 'Post/comments' column")
        
        # Initialize filter
        filter_obj = SemanticFilter() if request.use_semantic else LLMFilter()
        
        # Get texts to analyze
        texts = df['Post/comments'].fillna('').tolist()
        
        # Filter texts
        if request.use_semantic:
            results, similarities = await filter_obj.filter_texts(texts, request.criteria)
        else:
            results = await filter_obj.filter_texts(texts, request.criteria, request.batch_size)
        
        # Create filtered dataframe
        mask = pd.Series([results.get(i, False) for i in range(len(df))], index=df.index)
        filtered_df = df[mask].copy()
        
        # If using semantic filter, add similarity scores
        if request.use_semantic:
            filtered_df['similarity_score'] = [similarities[i] for i in filtered_df.index]
        
        # Convert to dict and clean for JSON serialization
        filtered_data = filtered_df.to_dict(orient='records')
        cleaned_data = clean_for_json(filtered_data)
        
        return {
            "total": len(df),
            "filtered": len(cleaned_data),
            "data": cleaned_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/frameworker")
async def run_frameworker(request: FrameworkerRequest):
    """
    Run frameworker analysis on Excel data.
    
    Parameters:
    - file: Name of the Excel file in the data directory
    - target_classes: Optional number of classes to generate
    - context: Optional context about the data
    """
    file_path = DATA_DIR / request.file
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        texts = df['Post/comments'].fillna('').tolist()
        
        # Initialize frameworker
        frameworker = Frameworker()
        
        # Process texts
        results = await frameworker.analyze_texts(
            texts,
            target_classes=request.target_classes,
            context=request.context
        )
        
        # Add comma-separated class names to the results
        class_names = [class_info['name'] for class_info in results['classes']]
        results['class_names'] = ', '.join(class_names)
        
        return clean_for_json(results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classifier")
async def run_classifier(request: ClassifierRequest):
    """
    Run classifier analysis on Excel data.
    
    Parameters:
    - file: Name of the Excel file in the data directory
    - classes: List of class names to use
    - is_multi_class: Whether to allow multiple classes per text
    """
    file_path = DATA_DIR / request.file
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        texts = df['Post/comments'].fillna('').tolist()
        
        # Initialize classifier
        classifier = Classifier()
        
        # Process texts
        results = await classifier.classify_texts(
            texts,
            classes=request.classes,
            is_multi_class=request.is_multi_class
        )
        
        return clean_for_json(results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/wizard")
async def run_wizard(request: WizardRequest):
    """
    Run data wizard analysis on Excel data.
    
    Parameters:
    - file: Name of the Excel file in the data directory
    - question: Question to analyze
    - use_semantic: Whether to use semantic filter
    """
    file_path = DATA_DIR / request.file
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        texts = df['Post/comments'].fillna('').tolist()
        
        # Initialize wizard
        wizard = DataWizard(use_semantic_filter=request.use_semantic)
        
        # Process question
        results = await wizard.analyze_question(texts, request.question)
        
        return clean_for_json(results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/comparative")
async def run_comparative(request: ComparativeRequest):
    """
    Run comparative analysis on two Excel files.
    
    Parameters:
    - file1: Name of the first Excel file
    - file2: Name of the second Excel file
    - description1: Description of the first dataset
    - description2: Description of the second dataset
    - question: Question to analyze
    - use_semantic: Whether to use semantic filter
    - is_competitive: Whether to use competitive analysis
    """
    file1_path = DATA_DIR / request.file1
    file2_path = DATA_DIR / request.file2
    
    if not file1_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file1}")
    if not file2_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.file2}")
    
    try:
        # Create config
        config = {
            "dataset1": {
                "file": str(file1_path),
                "description": request.description1
            },
            "dataset2": {
                "file": str(file2_path),
                "description": request.description2
            }
        }
        
        # Initialize analyzer
        analyzer_class = CompetitiveAnalyzer if request.is_competitive else ComparativeAnalyzer
        analyzer = analyzer_class(use_semantic_filter=request.use_semantic)
        
        # Process datasets
        results = await analyzer.analyze_datasets(config, request.question)
        
        return clean_for_json(results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an Excel file to the data directory."""
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx)")
    
    try:
        # Save file to data directory
        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {"message": f"File uploaded successfully: {file.filename}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Welcome endpoint with basic API information."""
    return {
        "message": "Welcome to the Social Media Analysis Dashboard API",
        "version": "0.1.0",
        "endpoints": {
            "/filter": "Filter Excel data based on LLM analysis",
            "/docs": "API documentation"
        }
    } 