import asyncio
from llm_utils import Frameworker
import pandas as pd
import sys
import os
from dotenv import load_dotenv
import json
from typing import Optional

async def test_frameworker(input_file: str, target_classes: Optional[int] = None, context: str = None):
    """Test the frameworker functionality with provided data file."""
    
    # Check for Google API key
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("\nError: Missing Google API key")
        print("\nTo fix this:")
        print("1. Get your API key from https://makersuite.google.com/app/apikey")
        print("2. Add it to your .env file like this:")
        print("\nGOOGLE_API_KEY=your-api-key-here")
        sys.exit(1)
    
    print(f"Reading data from: {input_file}")
    
    try:
        # Read the data
        df = pd.read_excel(input_file)
        texts = df['Post/comments'].fillna('').tolist()
        
        print(f"Loaded {len(texts)} comments for analysis")
        if target_classes:
            print(f"Target number of classes: {target_classes}")
        else:
            print("Letting LLM determine optimal number of classes")
        
        # Initialize frameworker
        frameworker = Frameworker()
        
        # Process the texts
        print("\nAnalyzing and classifying texts...")
        results = await frameworker.analyze_texts(
            texts,
            target_classes=target_classes,
            context=context,
            batch_size=10000
        )
        
        # Save results
        output_file = "classification_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        # Print summary
        print("\nClassification Summary:")
        print("-" * 50)
        print(f"Generated {len(results['classes'])} classes:")
        for i, class_info in enumerate(results['classes'], 1):
            print(f"\n{i}. {class_info['name']}")
            print(f"   Description: {class_info['description']}")
            print(f"   Keywords: {', '.join(class_info['keywords'])}")
            print(f"   Examples: {', '.join(class_info['examples'][:2])}...")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_frameworker.py <path_to_excel_file> [num_classes] [context]")
        print("\nArguments:")
        print("  path_to_excel_file  : Path to the Excel file containing the data")
        print("  num_classes         : (Optional) Number of classes to generate. If not provided, LLM will determine optimal number")
        print("  context            : (Optional) Context about the data")
        print("\nExamples:")
        print('python test_frameworker.py data.xlsx')
        print('python test_frameworker.py data.xlsx 15')
        print('python test_frameworker.py data.xlsx 15 "This is telecom customer feedback data"')
        sys.exit(1)
        
    file_path = sys.argv[1]
    target_classes = int(sys.argv[2]) if len(sys.argv) > 2 else None
    context = sys.argv[3] if len(sys.argv) > 3 else None
    
    asyncio.run(test_frameworker(file_path, target_classes, context)) 