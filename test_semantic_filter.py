import asyncio
from llm_utils import SemanticFilter
import pandas as pd
import sys
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns

def validate_api_key(api_key: str) -> bool:
    """Validate that the API key looks like a real OpenAI key."""
    if not api_key or api_key == "your-actual-api-key-here" or "your_api" in api_key:
        return False
    if not api_key.startswith('sk-') or len(api_key) < 40:
        return False
    return True

def plot_similarity_distribution(similarities, threshold, output_file="similarity_distribution.png"):
    """Plot the distribution of similarities and the threshold."""
    plt.figure(figsize=(10, 6))
    sns.histplot(similarities, bins=50)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
    plt.title('Distribution of Semantic Similarities')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(output_file)
    plt.close()

async def test_semantic_filter(input_file: str):
    """Test the semantic filter functionality with provided data file."""
    
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
        semantic_filter = SemanticFilter()
        
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
                # Get results and similarities
                results, similarities = await semantic_filter.filter_texts(texts, criteria)
                
                # Create mask and get matching indices
                matching_indices = [i for i, v in results.items() if v]
                
                # Create a new DataFrame with only the matching rows
                filtered_df = df.loc[matching_indices].copy()
                
                # Plot similarity distribution
                plot_similarity_distribution(
                    similarities, 
                    semantic_filter.last_threshold,  # Use the threshold from the filter
                    f"similarity_distribution_{criteria}.png"
                )
                

                # Add similarity scores to the filtered DataFrame
                filtered_df.loc[:, 'similarity_score'] = [similarities[i] for i in matching_indices]
                
                # Save results with all columns and similarity scores
                output_file = f"semantic_results_{criteria}.xlsx"
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
        print("Usage: python test_semantic_filter.py <path_to_excel_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    asyncio.run(test_semantic_filter(file_path)) 