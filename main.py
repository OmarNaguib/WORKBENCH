from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import tempfile
import os
from llm_utils import LLMFilter

app = FastAPI(
    title="Social Media Analysis Dashboard",
    description="API for analyzing social media data using LLMs",
    version="0.1.0"
)

@app.post("/filter")
async def filter_data(
    file: UploadFile = File(...),
    criteria: str = "boycott",  # Default criteria
    batch_size: int = 10  # Default batch size
):
    """
    Filter Excel data based on LLM analysis.
    
    Parameters:
    - file: Excel file with a 'Post/comments' column
    - criteria: Text describing what kind of content to filter for
    - batch_size: Number of comments to process in each batch (default: 300)
    
    Returns:
    - Excel file containing only the matching rows
    """
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx)")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Read the Excel file
        df = pd.read_excel(tmp_path)
        
        if 'Post/comments' not in df.columns:
            raise HTTPException(status_code=400, detail="Excel file must contain a 'Post/comments' column")
        
        # Initialize LLM filter
        llm_filter = LLMFilter()
        
        # Get texts to analyze
        texts = df['Post/comments'].fillna('').tolist()
        
        # Filter texts with batching
        results_dict = await llm_filter.filter_texts(texts, criteria, batch_size)
        
        # Create a boolean mask for filtering
        mask = pd.Series([results_dict.get(i, False) for i in range(len(df))], index=df.index)
        
        # Create filtered dataframe
        filtered_df = df[mask]
        
        # Save filtered results
        output_path = tmp_path.replace('.xlsx', '_filtered.xlsx')
        filtered_df.to_excel(output_path, index=False)
        
        return FileResponse(
            output_path,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=f"{file.filename.replace('.xlsx', '')}_filtered.xlsx"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary files
        os.unlink(tmp_path)
        if 'output_path' in locals():
            os.unlink(output_path)

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