from typing import List, Dict, Tuple, Optional
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import BooleanOutputParser, CommaSeparatedListOutputParser
import numpy as np
from tqdm import tqdm
import google.generativeai as genai
import json

load_dotenv()

# Configure Google AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

class Frameworker:
    """Discovers themes and topics in raw text data."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Using pro instead of flash for more reliable JSON output
            temperature=0,
            convert_system_message_to_human=True
        )
    
    def create_class_discovery_prompt(self, context: Optional[str] = None, target_classes: Optional[int] = None) -> ChatPromptTemplate:
        """Creates a prompt for class discovery."""
        base_template = """You are an expert data analyst specializing in discovering patterns in social media data.

        Your task is to analyze a collection of social media comments and identify distinct classes.
        {target_classes_instruction}
        
        {context_section}
        
        For the following collection of comments, please identify {class_count_instruction} distinct classes that best categorize these comments.
        Each class should be specific enough to be meaningful but broad enough to capture related content.
        
        Comments to analyze:
        {comments}
        
        You must respond with valid JSON only, using this exact format:
        {{
            "classes": [
                {{
                    "name": "Class name",
                    "description": "Detailed description of what belongs in this class",
                    "keywords": ["relevant", "keywords", "that", "identify", "this", "class"],
                    "examples": ["Example phrase 1", "Example phrase 2"]
                }}
            ]
        }}
        
        {reminder_instruction}
        
        Important: Your entire response must be valid JSON. Do not include any other text before or after the JSON.
        """
        
        context_section = f"Additional context about the data:\n{context}" if context else ""
        target_classes_instruction = f"You must identify exactly {target_classes} classes." if target_classes else "You should determine the optimal number of classes based on the data, but aim for 5-15 classes unless the data clearly requires more or fewer."
        class_count_instruction = f"exactly {target_classes}" if target_classes else "the optimal number of"
        reminder_instruction = f"Remember: Return EXACTLY {target_classes} classes, no more, no less." if target_classes else "Remember: Each class should be distinct and meaningful. Don't create too many granular classes or too few broad ones."
        
        return ChatPromptTemplate.from_template(
            base_template.replace("{context_section}", context_section)
                       .replace("{target_classes_instruction}", target_classes_instruction)
                       .replace("{class_count_instruction}", class_count_instruction)
                       .replace("{reminder_instruction}", reminder_instruction)
        )
    
    async def process_batch(self, texts: List[str], context: Optional[str] = None, target_classes: Optional[int] = None) -> Dict:
        """Process a batch of texts to discover classes."""
        prompt = self.create_class_discovery_prompt(context, target_classes)
        
        # Format texts for the prompt
        formatted_texts = "\n".join([f"- {text}" for text in texts[:100]])  # Limit to 100 examples to avoid token limits
        if len(texts) > 100:
            formatted_texts += f"\n... and {len(texts) - 100} more comments"
        
        # Get analysis from LLM
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "comments": formatted_texts,
            "target_classes": target_classes if target_classes else "optimal"
        })
        
        # Parse JSON response
        try:
            # Clean the response content
            content = response.content.strip()
            # If response starts with ``` or ends with ```, remove them
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            
            # Validate the result structure
            if not isinstance(result, dict) or 'classes' not in result:
                raise ValueError("Response missing 'classes' key")
            if not isinstance(result['classes'], list):
                raise ValueError("'classes' must be a list")
            if not result['classes']:
                raise ValueError("No classes were generated")
            
            # Validate each class has required fields
            for class_info in result['classes']:
                if not all(k in class_info for k in ['name', 'description', 'keywords', 'examples']):
                    raise ValueError("Class missing required fields")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print("Raw response:", response.content)
            raise ValueError("Could not parse LLM response as JSON")
        except Exception as e:
            print(f"Validation Error: {str(e)}")
            print("Parsed response:", result if 'result' in locals() else 'No result')
            raise ValueError(f"Invalid response structure: {str(e)}")
    
    def create_merge_prompt(self, all_classes: List[Dict], target_classes: Optional[int] = None) -> ChatPromptTemplate:
        """Creates a prompt for merging classes."""
        template = """You are an expert at organizing and consolidating classification systems.
        
        Your task is to analyze the following classes discovered from different batches of data and merge them into {target_instruction} distinct, well-defined classes.
        
        When merging:
        1. Identify similar classes and combine them
        2. Keep the most distinctive and important classes
        3. Ensure each final class is specific enough to be meaningful but broad enough to be useful
        4. Maintain the most relevant keywords and examples from the source classes
        
        Classes to merge:
        {classes}
        
        You must respond with valid JSON only, using this exact format:
        {{
            "classes": [
                {{
                    "name": "Class name",
                    "description": "Detailed description of what belongs in this class",
                    "keywords": ["relevant", "keywords", "that", "identify", "this", "class"],
                    "examples": ["Example phrase 1", "Example phrase 2"]
                }}
            ]
        }}
        
        {reminder_instruction}
        
        Important: Your entire response must be valid JSON. Do not include any other text before or after the JSON.
        """
        
        target_instruction = f"EXACTLY {target_classes}" if target_classes else "an optimal number of (aim for 5-15 unless data clearly requires more or fewer)"
        reminder_instruction = f"Remember: Return EXACTLY {target_classes} classes, no more, no less." if target_classes else "Remember: Each class should be distinct and meaningful. Don't create too many granular classes or too few broad ones."
        
        return ChatPromptTemplate.from_template(
            template.replace("{target_instruction}", target_instruction)
                   .replace("{reminder_instruction}", reminder_instruction)
        )
    
    async def merge_batch_results(self, results: List[Dict], target_classes: Optional[int] = None) -> Dict:
        """Merge results from multiple batches into classes using LLM."""
        if not results:
            return {"classes": []}
            
        # Collect all classes from all batches
        all_classes = []
        for result in results:
            if isinstance(result, dict) and 'classes' in result:
                all_classes.extend(result.get("classes", []))
            
        # If no valid classes were collected, return empty result
        if not all_classes:
            return {"classes": []}
            
        # If target_classes is specified and we have fewer classes, return as is
        if target_classes and len(all_classes) <= target_classes:
            return {"classes": all_classes}
            
        # Create merge prompt
        prompt = self.create_merge_prompt(target_classes)
        
        # Get merged classes from LLM
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "classes": json.dumps(all_classes, ensure_ascii=False, indent=2),
            "target_classes": target_classes if target_classes else "optimal"
        })
        
        # Parse response with same validation as process_batch
        try:
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            elif content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            
            # Validate the result structure
            if not isinstance(result, dict) or 'classes' not in result:
                raise ValueError("Response missing 'classes' key")
            if not isinstance(result['classes'], list):
                raise ValueError("'classes' must be a list")
            if not result['classes']:
                raise ValueError("No classes were generated")
            
            # Validate each class has required fields
            for class_info in result['classes']:
                if not all(k in class_info for k in ['name', 'description', 'keywords', 'examples']):
                    raise ValueError("Class missing required fields")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error in merge: {str(e)}")
            print("Raw response:", response.content)
            # If merge fails, return the original classes
            return {"classes": all_classes}
        except Exception as e:
            print(f"Validation Error in merge: {str(e)}")
            # If merge fails, return the original classes
            return {"classes": all_classes}
    
    async def analyze_texts(self, texts: List[str], target_classes: Optional[int] = None, batch_size: int = 10000, context: Optional[str] = None) -> Dict:
        """Analyze texts in batches to discover classes."""
        all_results = []
        
        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing classes"):
            batch = texts[i:i + batch_size]
            try:
                batch_result = await self.process_batch(batch, context, target_classes)
                all_results.append(batch_result)
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        # Merge results
        merged_result = await self.merge_batch_results(all_results, target_classes)
        return merged_result

class SemanticFilter:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0
        )
        self.boolean_parser = BooleanOutputParser()
        self.last_threshold = None
        self.last_similarities = None
        
    async def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Get embeddings for a list of texts in batches."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Getting embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self.embeddings.aembed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
        return np.array(all_embeddings)
    
    def calculate_similarities(self, text_embeddings: np.ndarray, criteria_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between text embeddings and criteria embedding."""
        # Normalize embeddings
        text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        criteria_norm = np.linalg.norm(criteria_embedding)
        
        normalized_texts = text_embeddings / text_norms
        normalized_criteria = criteria_embedding / criteria_norm
        
        # Calculate cosine similarities
        similarities = np.dot(normalized_texts, normalized_criteria)
        return similarities
    
    async def check_relevance(self, texts: List[str], criteria: str) -> bool:
        """Ask LLM if the given texts are relevant to the criteria."""
        template = """You are a precise content filter.
        Your task is to determine if the following texts are relevant to this criteria: {criteria}
        
        Consider these texts as a group and determine if this similarity threshold is appropriate.
        If these texts are clearly relevant to the criteria, respond with 'YES'.
        If these texts are not clearly relevant or are borderline, respond with 'NO'.
        
        Texts to analyze:
        {texts}
        
        Remember to respond with ONLY 'YES' or 'NO'.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | self.boolean_parser
        
        formatted_texts = "\n".join([f"- {text}" for text in texts])
        result = await chain.ainvoke({
            "criteria": criteria,
            "texts": formatted_texts
        })
        
        return result
    
    async def binary_search_threshold(self, texts: List[str], similarities: np.ndarray, criteria: str, 
                                    sample_size: int = 5) -> float:
        """Find optimal similarity threshold using binary search and LLM verification."""
        # Sort all indices by similarity score
        sorted_indices = np.argsort(similarities)[::-1]  # Descending order
        sorted_similarities = similarities[sorted_indices]
        sorted_texts = [texts[i] for i in sorted_indices]
        
        left, right = 0, len(sorted_texts) - 1
        best_threshold = sorted_similarities[0]  # Start with highest similarity
        
        print("\nFinding optimal threshold using binary search...")
        max_iterations = 10  # Prevent too many API calls
        
        for iteration in range(max_iterations):
            mid = (left + right) // 2
            print(f"\nIteration {iteration + 1}, checking texts at position {mid}/{len(sorted_texts)}")
            print(f"Current similarity threshold: {sorted_similarities[mid]:.3f}")
            
            # Get sample texts around the middle point
            start_idx = max(0, mid - sample_size // 2)
            end_idx = min(len(sorted_texts), mid + sample_size // 2 + 1)
            sample_texts = sorted_texts[start_idx:end_idx]
            
            # Check if these texts are relevant
            is_relevant = await self.check_relevance(sample_texts, criteria)
            
            if is_relevant:
                # These are relevant, try lower similarity scores (move right)
                best_threshold = sorted_similarities[mid]
                left = mid + 1
            else:
                # These are not relevant, try higher similarity scores (move left)
                right = mid - 1
            
            # Break if we've converged
            if left > right:
                break
            
            print(f"Current range: {sorted_similarities[right]:.3f} - {sorted_similarities[left]:.3f}")
        
        return best_threshold
    
    async def filter_texts(self, texts: List[str], criteria: str) -> Tuple[Dict[int, bool], np.ndarray]:
        """Filter texts based on semantic similarity to criteria."""
        print("Getting embeddings for texts...")
        text_embeddings = await self.get_embeddings_batch(texts)
        
        print("Getting embedding for criteria...")
        criteria_embedding = (await self.embeddings.aembed_documents([criteria]))[0]
        
        print("Calculating similarities...")
        similarities = self.calculate_similarities(text_embeddings, criteria_embedding)
        
        # Find threshold using binary search with LLM verification
        self.last_threshold = await self.binary_search_threshold(texts, similarities, criteria)
        print(f"\nFinal threshold: {self.last_threshold:.3f}")
        
        # Create results dictionary
        results = {i: sim >= self.last_threshold for i, sim in enumerate(similarities)}
        
        # Store similarities for potential use
        self.last_similarities = similarities
        
        return results, similarities

class LLMFilter:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
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