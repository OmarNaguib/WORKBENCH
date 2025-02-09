# Social Media Analysis Dashboard

This project provides an intelligent dashboard for analyzing social media data using LLMs. The system allows for various types of analysis including filtering, classification, and sentiment analysis of social media comments.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
uvicorn main:app --reload
```

## Features

### Currently Implemented:
- Filterer: Filter comments based on custom prompts (e.g., find comments related to boycotts)

### Upcoming Features:
- Threshold Filter: Get top N% most relevant comments
- Classifier: Classify comments into user-defined buckets
- Frameworker: Auto-discover classification buckets
- Weighter: Calculate intensity scores
- Generic Map Reducer: Custom data processing pipelines
- Comparative Analysis
- Custom Filters

## Usage

1. Prepare your data in an XLSX file with a "Post/comments" column
2. Use the /filter endpoint to process your data
3. Receive filtered results in a new XLSX file

## API Documentation

Once the server is running, visit http://localhost:8000/docs for complete API documentation. 