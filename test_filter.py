import asyncio
from llm_utils import LLMFilter
import pandas as pd
import sys
import os
from dotenv import load_dotenv

def validate_api_key(api_key: str) -> bool:
    """Validate that the API key looks like a real OpenAI key."""
    if not api_key or api_key == "your-actual-api-key-here" or "your_api" in api_key:
        return False
    # OpenAI API keys typically start with 'sk-' and are ~51 characters long
    if not api_key.startswith('sk-') or len(api_key) < 40:
        return False
    return True

async def test_filter(input_file: str):
    """Test the LLM filter functionality with provided data file."""
    
    # Check for API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key or not validate_api_key(api_key):
        print("\nError: Invalid or missing OpenAI API key")
        print("\nTo fix this:")
        print("1. Get your API key from https://platform.openai.com/account/api-keys")
        print("2. Create a .env file in the same directory as this script")
        print("3. Add your API key to the .env file like this:")
        print("\nOPENAI_API_KEY=sk-your-actual-key-here")
        print("\nMake sure:")
        print("- The key starts with 'sk-'")
        print("- You've replaced the placeholder with your actual API key")
        print("- There are no quotes around the API key")
        sys.exit(1)
    
    print(f"Reading data from: {input_file}")
    
    try:
        # Read the data
        df = pd.read_excel(input_file)
        texts = df['Post/comments'].fillna('').tolist()
        
        print(f"Loaded {len(texts)} comments for analysis")
        
        # Initialize filter
        llm_filter = LLMFilter()
        
        # Test cases specific to Vodafone data analysis
        test_cases = [
            "مشاكل فودافون كاش",    # Vodafone Cash issues
            "شكاوى خدمة العملاء",   # Customer service complaints
            "تعليقات إيجابية",      # Positive feedback
            "مشاكل تقنية",          # Technical issues
            "استفسارات عن الخدمات"  # Service inquiries
        ]
        
        for criteria in test_cases:
            print(f"\nTesting criteria: {criteria}")
            print("-" * 50)
            
            try:
                # Get results
                results = await llm_filter.filter_texts(texts, criteria, batch_size=300)
                
                # Create mask and filter
                mask = pd.Series([results.get(i, False) for i in range(len(df))], index=df.index)
                filtered_df = df[mask]
                
                # Print results with additional context
                print(f"Found {len(filtered_df)} matching comments:")
                for _, row in filtered_df.iterrows():
                    print(f"\nPlatform: {row['Platform']}")
                    print(f"Comment: {row['Post/comments']}")
                    print(f"Sentiment: {row['Sentiment']}")
                    if pd.notna(row['Number of shares/retweets']):
                        print(f"Shares: {row['Number of shares/retweets']}")
                    print("-" * 30)
                    
                # Save results with all columns
                output_file = f"vodafone_results_{criteria}.xlsx"
                filtered_df.to_excel(output_file, index=False)
                print(f"Results saved to: {output_file}")
                
            except Exception as e:
                print(f"Error processing criteria '{criteria}': {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_filter.py <path_to_excel_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    asyncio.run(test_filter(file_path)) 