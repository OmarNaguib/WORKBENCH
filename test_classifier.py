import asyncio
from llm_utils import Classifier
import pandas as pd
import sys
import os
from dotenv import load_dotenv
import json
from typing import List

async def test_classifier(input_file: str, classes: List[str], is_multi_class: bool = False):
    """Test the classifier functionality with provided data file."""
    
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
        print(f"Classes to use: {', '.join(classes)}")
        print(f"Classification type: {'Multi-class' if is_multi_class else 'Single-class'}")
        
        # Initialize classifier
        classifier = Classifier()
        
        # Process the texts
        print("\nClassifying texts...")
        results = await classifier.classify_texts(
            texts,
            classes=classes,
            is_multi_class=is_multi_class,
            batch_size=300
        )
        
        # Save detailed results
        output_file = "classification_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Create Excel file with classifications
        excel_results = []
        for idx, text in enumerate(texts):
            # Find all classes this text belongs to
            text_classes = []
            for cls in classes + ['Other']:
                for entry in results[cls]:
                    if entry['index'] == idx:
                        text_classes = entry['classes']
                        break
                if text_classes:
                    break
            
            excel_results.append({
                'Post/comments': text,
                'Class' if not is_multi_class else 'Classes': '|'.join(text_classes),
                'Original_Index': idx
            })
        
        df_results = pd.DataFrame(excel_results)
        excel_output = "classification_results.xlsx"
        df_results.to_excel(excel_output, index=False)
        print(f"Excel results saved to: {excel_output}")
        
        # Print summary
        print("\nClassification Summary:")
        print("-" * 50)
        summary = results['summary']
        print(f"Total texts processed: {summary['total_texts']}")
        print("\nClass distribution:")
        for cls, count in summary['class_distribution'].items():
            percentage = (count / summary['total_texts']) * 100
            print(f"- {cls}: {count} texts ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

def parse_classes(classes_str: str) -> List[str]:
    """Parse comma-separated class names into a list."""
    return [cls.strip() for cls in classes_str.split(',')]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_classifier.py <path_to_excel_file> <comma_separated_classes> [--multi-class]")
        print("\nArguments:")
        print("  path_to_excel_file     : Path to the Excel file containing the data")
        print("  comma_separated_classes : Comma-separated list of class names to use")
        print("  --multi-class          : (Optional) Allow texts to belong to multiple classes")
        print("\nExamples:")
        print('python test_classifier.py data.xlsx "Technical Issues, Customer Service, Product Feedback"')
        print('python test_classifier.py data.xlsx "Bugs, Features, Documentation" --multi-class')
        sys.exit(1)
        
    file_path = sys.argv[1]
    classes = parse_classes(sys.argv[2])
    is_multi_class = "--multi-class" in sys.argv
    
    asyncio.run(test_classifier(file_path, classes, is_multi_class)) 