from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import BooleanOutputParser

load_dotenv()

class LLMFilter:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
        self.boolean_parser = BooleanOutputParser()
        
    def create_filter_prompt(self, user_criteria: str) -> ChatPromptTemplate:
        """Creates a prompt template for filtering based on user criteria."""
        template = """You are a precise content filter.
        Your task is to determine which of the following texts match this criteria: {criteria}
        
        For each text, respond with its index followed by true or false, separated by a comma.
        Example format:
        0,true
        1,false
        2,true
        
        Texts to analyze:
        {text_batch}
        
        Remember to respond with ONLY the index and true/false for each text, one per line.
        """
        
        return ChatPromptTemplate.from_template(template)
        
    def format_batch(self, texts_with_indices: List[Tuple[int, str]]) -> str:
        """Format a batch of texts with their indices for the prompt."""
        return "\n".join([f"Index {idx}: {text}" for idx, text in texts_with_indices])
        
    async def process_batch(self, texts_with_indices: List[Tuple[int, str]], criteria: str) -> Dict[int, bool]:
        """Process a batch of texts and return results mapped to their indices."""
        print(f"Processing batch with criteria: {criteria}")
        prompt = self.create_filter_prompt(criteria)
        
        # Format the batch text
        batch_text = self.format_batch(texts_with_indices)
        
        # Get raw response from LLM
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "criteria": criteria,
            "text_batch": batch_text
        })
        
        # Parse response into dictionary
        results = {}
        for line in response.content.strip().split('\n'):
            try:
                idx_str, result_str = line.strip().split(',')
                idx = int(idx_str)
                results[idx] = result_str.strip().lower() == 'true'
            except (ValueError, IndexError):
                continue
                
        return results

    async def filter_texts(self, texts: List[str], criteria: str, batch_size: int = 300) -> Dict[int, bool]:
        """Filters a list of texts based on the given criteria using batched processing."""
        all_results = {}
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Create list of (index, text) tuples for this batch
            batch_with_indices = [(idx, text) for idx, text in enumerate(batch_texts, start=i)]
            
            # Process batch and update results
            batch_results = await self.process_batch(batch_with_indices, criteria)
            all_results.update(batch_results)
            
        return all_results 