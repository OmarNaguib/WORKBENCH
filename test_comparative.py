import asyncio
from llm_utils import ComparativeAnalyzer, CompetitiveAnalyzer
import sys
import os
from dotenv import load_dotenv
import json

async def test_analysis(config_file: str, question: str, use_semantic_filter: bool = True, is_competitive: bool = False):
    """Test the ComparativeAnalyzer or CompetitiveAnalyzer functionality with provided configuration."""
    
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
        print(f"Analysis type: {'Competitive' if is_competitive else 'General'} comparison")
        print(f"Question: {question}")
        
        # Initialize analyzer
        analyzer_class = CompetitiveAnalyzer if is_competitive else ComparativeAnalyzer
        analyzer = analyzer_class(use_semantic_filter=use_semantic_filter)
        
        # Process the datasets
        print("\nAnalyzing datasets...")
        results = await analyzer.analyze_datasets(config, question)
        
        # Save results
        output_file = "competitive_results.json" if is_competitive else "comparative_results.json"
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
            if is_competitive:
                print(f"   Competitor 1: {insight['competitor1_position']}")
                print(f"   Competitor 2: {insight['competitor2_position']}")
            else:
                print(f"   Dataset 1: {insight['dataset1_position']}")
                print(f"   Dataset 2: {insight['dataset2_position']}")
            print(f"   Key Differences: {insight['key_differences']}")
            print(f"   Implications: {insight['implications']}")
        
        if is_competitive:
            print("\nSentiment Comparison:")
            sentiment = results['key_metrics_comparison']['sentiment']
            print(f"Competitor 1: {sentiment['competitor1']}")
            print(f"Competitor 2: {sentiment['competitor2']}")
            print(f"Key Differences: {sentiment['difference']}")
            
            print("\nUser Satisfaction Scores:")
            satisfaction = results['key_metrics_comparison']['user_satisfaction']
            print(f"Competitor 1: {satisfaction['competitor1_score']}")
            print(f"Competitor 2: {satisfaction['competitor2_score']}")
            print(f"Analysis: {satisfaction['analysis']}")
            
            print("\nCompetitive Advantages:")
            print("\nCompetitor 1:")
            for adv in results['competitive_advantages']['competitor1']:
                print(f"- {adv}")
            print("\nCompetitor 2:")
            for adv in results['competitive_advantages']['competitor2']:
                print(f"- {adv}")
            
            print("\nStrategic Recommendations:")
            print("\nFor Competitor 1:")
            for rec in results['recommendations']['competitor1']:
                print(f"- {rec}")
            print("\nFor Competitor 2:")
            for rec in results['recommendations']['competitor2']:
                print(f"- {rec}")
        else:
            print("\nMetric Comparisons:")
            metrics = results['metric_comparisons']
            print("\nDistributions:")
            print(f"Dataset 1: {metrics['distributions']['dataset1']}")
            print(f"Dataset 2: {metrics['distributions']['dataset2']}")
            print(f"Key Differences: {metrics['distributions']['differences']}")
            
            print("\nPatterns:")
            print("\nCommon Patterns:")
            for pattern in metrics['patterns']['common_patterns']:
                print(f"- {pattern}")
            print("\nUnique to Dataset 1:")
            for pattern in metrics['patterns']['unique_to_dataset1']:
                print(f"- {pattern}")
            print("\nUnique to Dataset 2:")
            for pattern in metrics['patterns']['unique_to_dataset2']:
                print(f"- {pattern}")
            
            print("\nKey Findings:")
            print("\nSimilarities:")
            for sim in results['key_findings']['similarities']:
                print(f"- {sim}")
            print("\nDifferences:")
            for diff in results['key_findings']['differences']:
                print(f"- {diff}")
            
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"- {rec}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_comparative.py <config_file.json> <question> [--llm-filter] [--competitive]")
        print("\nArguments:")
        print("  config_file.json : JSON file containing dataset configurations")
        print("  question        : Question or analysis request in quotes")
        print("  --llm-filter   : (Optional) Use LLM filter instead of semantic filter")
        print("  --competitive  : (Optional) Use competitive analysis mode")
        print("\nExample config.json format:")
        print("""
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
        """)
        print("\nExamples:")
        print('python test_comparative.py config.json "Compare distribution patterns"')
        print('python test_comparative.py config.json "Compare customer satisfaction" --competitive')
        sys.exit(1)
        
    config_file = sys.argv[1]
    question = sys.argv[2]
    use_semantic_filter = "--llm-filter" not in sys.argv
    is_competitive = "--competitive" in sys.argv
    
    asyncio.run(test_analysis(config_file, question, use_semantic_filter, is_competitive)) 