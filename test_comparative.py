import asyncio
from llm_utils import ComparativeAnalyzer
import sys
import os
from dotenv import load_dotenv
import json

async def test_comparative(config_file: str, question: str, use_semantic_filter: bool = True):
    """Test the ComparativeAnalyzer functionality with provided configuration."""
    
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
    
    try:
        # Read configuration
        print(f"Reading configuration from: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"\nDataset 1: {config['dataset1']['description']}")
        print(f"Dataset 2: {config['dataset2']['description']}")
        print(f"Using {'semantic' if use_semantic_filter else 'LLM'} filter")
        print(f"Question: {question}")
        
        # Initialize analyzer
        analyzer = ComparativeAnalyzer(use_semantic_filter=use_semantic_filter)
        
        # Process the datasets
        print("\nAnalyzing datasets...")
        results = await analyzer.analyze_datasets(config, question)
        
        # Save results
        output_file = "comparative_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Print summary
        if "error" in results:
            print("\nError in analysis:")
            print(results["message"])
            return
        
        print("\nComparative Analysis Summary:")
        print("-" * 50)
        print(f"\nOverall Summary: {results['summary']}")
        
        print("\nKey Insights:")
        for i, insight in enumerate(results['comparative_insights'], 1):
            print(f"\n{i}. Comparing {insight['aspect']}")
            print(f"   Dataset 1: {insight['dataset1_position']}")
            print(f"   Dataset 2: {insight['dataset2_position']}")
            print(f"   Key Differences: {insight['key_differences']}")
            print(f"   Implications: {insight['implications']}")
        
        print("\nSentiment Comparison:")
        sentiment = results['key_metrics_comparison']['sentiment']
        print(f"Dataset 1: {sentiment['dataset1']}")
        print(f"Dataset 2: {sentiment['dataset2']}")
        print(f"Key Differences: {sentiment['difference']}")
        
        print("\nUser Satisfaction Scores:")
        satisfaction = results['key_metrics_comparison']['user_satisfaction']
        print(f"Dataset 1: {satisfaction['dataset1_score']}")
        print(f"Dataset 2: {satisfaction['dataset2_score']}")
        print(f"Analysis: {satisfaction['analysis']}")
        
        print("\nCompetitive Advantages:")
        print("\nDataset 1:")
        for adv in results['competitive_advantages']['dataset1']:
            print(f"- {adv}")
        print("\nDataset 2:")
        for adv in results['competitive_advantages']['dataset2']:
            print(f"- {adv}")
        
        print("\nRecommendations:")
        print("\nFor Dataset 1:")
        for rec in results['recommendations']['dataset1']:
            print(f"- {rec}")
        print("\nFor Dataset 2:")
        for rec in results['recommendations']['dataset2']:
            print(f"- {rec}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_comparative.py <config_file.json> <question> [--llm-filter]")
        print("\nArguments:")
        print("  config_file.json : JSON file containing dataset configurations")
        print("  question        : Question or analysis request in quotes")
        print("  --llm-filter   : (Optional) Use LLM filter instead of semantic filter")
        print("\nExample config.json format:")
        print("""
{
    "dataset1": {
        "file": "vodafone_data.xlsx",
        "description": "Customer feedback for Vodafone services"
    },
    "dataset2": {
        "file": "etisalat_data.xlsx",
        "description": "Customer feedback for Etisalat services"
    }
}
        """)
        print("\nExamples:")
        print('python test_comparative.py config.json "How do people view network quality?"')
        print('python test_comparative.py config.json "Compare customer service satisfaction" --llm-filter')
        sys.exit(1)
        
    config_file = sys.argv[1]
    question = sys.argv[2]
    use_semantic_filter = "--llm-filter" not in sys.argv
    
    asyncio.run(test_comparative(config_file, question, use_semantic_filter)) 