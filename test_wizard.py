import asyncio
from llm_utils import DataWizard
import pandas as pd
import sys
import os
from dotenv import load_dotenv
import json

async def test_wizard(input_file: str, question: str, use_semantic_filter: bool = True):
    """Test the DataWizard functionality with provided data file."""
    
    # Check for API keys
    load_dotenv()
    google_api_key = os.getenv('GOOGLE_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY') if use_semantic_filter else None
    
    if not google_api_key:
        print("\nError: Missing Google API key")
        print("\nTo fix this:")
        print("1. Get your API key from https://makersuite.google.com/app/apikey")
        print("2. Add it to your .env file like this:")
        print("\nGOOGLE_API_KEY=your-api-key-here")
        sys.exit(1)
    
    if use_semantic_filter and not openai_api_key:
        print("\nError: Missing OpenAI API key (required for semantic filtering)")
        print("\nTo fix this:")
        print("1. Get your API key from https://platform.openai.com/account/api-keys")
        print("2. Add it to your .env file like this:")
        print("\nOPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
    
    print(f"Reading data from: {input_file}")
    
    try:
        # Read the data
        df = pd.read_excel(input_file)
        texts = df['Post/comments'].fillna('').tolist()
        
        print(f"Loaded {len(texts)} comments for analysis")
        print(f"Using {'semantic' if use_semantic_filter else 'LLM'} filter")
        print(f"Question: {question}")
        
        # Initialize wizard
        wizard = DataWizard(use_semantic_filter=use_semantic_filter)
        
        # Process the question
        print("\nAnalyzing data...")
        results = await wizard.analyze_question(texts, question)
        
        # Save results
        output_file = "wizard_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        print("\nAnalysis Summary:")
        print("-" * 50)
        print(f"\nSummary: {results['summary']}")
        
        print("\nKey Insights:")
        for i, insight in enumerate(results['insights'], 1):
            print(f"\n{i}. {insight['observation']}")
            print(f"   Confidence: {insight['confidence']}")
            print(f"   Implications: {insight['implications']}")
            print(f"   Evidence: {', '.join(insight['evidence'][:2])}...")
        
        print("\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_wizard.py <path_to_excel_file> <question> [--llm-filter]")
        print("\nArguments:")
        print("  path_to_excel_file : Path to the Excel file containing the data")
        print("  question          : Question or analysis request in quotes")
        print("  --llm-filter     : (Optional) Use LLM filter instead of semantic filter")
        print("\nExamples:")
        print('python test_wizard.py data.xlsx "What are the main complaints about Vodafone Cash?"')
        print('python test_wizard.py data.xlsx "Analyze customer satisfaction trends" --llm-filter')
        sys.exit(1)
        
    file_path = sys.argv[1]
    question = sys.argv[2]
    use_semantic_filter = "--llm-filter" not in sys.argv
    
    asyncio.run(test_wizard(file_path, question, use_semantic_filter)) 