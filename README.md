# Social Media Analysis Dashboard

This project provides an intelligent dashboard for analyzing social media data using LLMs. The system allows for various types of analysis including filtering, classification, sentiment analysis, and comparative analysis of social media comments.

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

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key_here  # Required for semantic filtering
GOOGLE_API_KEY=your_google_key_here   # Required for LLM operations
```

4. Run the application:
```bash
uvicorn main:app --reload
```

## Features

### Currently Implemented:
- Semantic Filter: Find relevant comments using semantic similarity
- LLM Filter: Filter comments using LLM-based analysis
- Classifier: Classify comments into user-defined categories
- Frameworker: Auto-discover classification categories
- Data Wizard: Answer questions about the data
- Comparative Analysis: Compare two datasets
- Competitive Analysis: Compare competing businesses/services

## Test Scripts

### 1. Semantic Filter Test
Test the semantic filtering functionality:
```bash
python test_semantic_filter.py data.xlsx
```
- Filters comments based on semantic similarity
- Generates similarity distribution plots
- Saves results with similarity scores

### 2. LLM Filter Test
Test the LLM-based filtering:
```bash
python test_filter.py data.xlsx
```
- Filters comments using LLM analysis
- Processes data in batches
- Saves filtered results

### 3. Classifier Test
Test the classification functionality:
```bash
# Single-class classification
python test_classifier.py data.xlsx "Technical Issues, Customer Service, Product Feedback"

# Multi-class classification
python test_classifier.py data.xlsx "Bugs, Features, Documentation" --multi-class
```
- Classifies comments into predefined categories
- Supports both single and multi-class classification
- Generates detailed classification reports

### 4. Frameworker Test
Test the automatic class discovery:
```bash
# Basic usage
python test_frameworker.py data.xlsx

# Specify number of classes
python test_frameworker.py data.xlsx 15

# With context
python test_frameworker.py data.xlsx 15 "This is telecom customer feedback data"
```
- Automatically discovers themes and topics
- Can specify target number of classes
- Supports additional context for better classification

### 5. Data Wizard Test
Test the question-answering functionality:
```bash
# Using semantic filter (default)
python test_wizard.py data.xlsx "What are the main complaints?"

# Using LLM filter
python test_wizard.py data.xlsx "Analyze customer satisfaction trends" --llm-filter
```
- Answers questions about the data
- Provides insights and recommendations
- Supports both semantic and LLM filtering

### 6. Comparative Analysis Test
Test the comparative analysis functionality:
```bash
# General comparison
python test_comparative.py config.json "Compare distribution patterns"

# Competitive analysis
python test_comparative.py config.json "Compare customer satisfaction" --competitive
```

Example `config.json`:
```json
{
    "dataset1": {
        "file": "dataset1.xlsx",
        "description": "Description of dataset 1"
    },
    "dataset2": {
        "file": "dataset2.xlsx",
        "description": "Description of dataset 2"
    }
}
```
- Compares two datasets
- Supports both general and competitive analysis
- Generates detailed comparison reports

## API Documentation

Once the server is running, visit http://localhost:8000/docs for complete API documentation.

## Output Files

Each test script generates specific output files:

- `*_filtered.xlsx`: Filtered results from filter tests
- `classification_results.json/xlsx`: Classification results
- `frameworker_results.json`: Auto-discovered classes
- `wizard_results.json`: Question-answering results
- `comparative_results.json`: Comparative analysis results
- `competitive_results.json`: Competitive analysis results
- `similarity_distribution_*.png`: Similarity distribution plots

## Error Handling

All scripts include robust error handling for:
- Missing API keys
- Invalid file formats
- Processing errors
- Invalid configurations

Make sure to check the console output for detailed error messages and instructions. 