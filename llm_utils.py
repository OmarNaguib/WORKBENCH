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
import pandas as pd

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
        
        context_section = f"your classes directly related to this criteria:\n{context}" if context else ""
        target_classes_instruction = f"You must identify exactly {target_classes} classes." if target_classes else "You should determine the optimal number of classes based on the data, but aim for 5-15 classes unless the data clearly requires more or fewer."
        class_count_instruction = f"exactly {target_classes}" if target_classes else "the optimal number of"
        reminder_instruction = f"Remember: Return EXACTLY {target_classes} classes, no more, no less." if target_classes else "Remember: Each class should be distinct and meaningful. Don't create too many granular classes or too few broad ones."+f"choose only classes directly related to this criteria:\n{context}"
        
        return ChatPromptTemplate.from_template(
            base_template.replace("{context_section}", context_section)
                       .replace("{target_classes_instruction}", target_classes_instruction)
                       .replace("{class_count_instruction}", class_count_instruction)
                       .replace("{reminder_instruction}", reminder_instruction)
        )
    
    async def process_batch(self, texts: List[str], context: Optional[str] = None, target_classes: Optional[int] = None) -> Dict:
        """Process a batch of texts to discover classes."""
        prompt = self.create_class_discovery_prompt(context, target_classes)
        print(f"Prompt: {prompt}")
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
            temperature=0,
            safety_settings={
                "HARASSMENT": "block_none",
                "HATE_SPEECH": "block_none",
                "SEXUALLY_EXPLICIT": "block_none",
                "DANGEROUS_CONTENT": "block_none"
            },
            generation_config={
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "candidate_count": 1,
            }
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
        return "\n".join([f"{idx}: {text}" for idx, text in texts_with_indices])
        
    async def process_batch(self, texts_with_indices: List[Tuple[int, str]], criteria: str) -> Dict[int, bool]:
        """Process a batch of texts and return results mapped to their indices."""
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
        try:
            # Clean the response content
            content = response.content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
            
            # Process each line
            for line in content.strip().split('\n'):
                try:
                    # Remove any non-essential characters
                    line = line.strip().replace(' ', '')
                    if ',' not in line:
                        continue
                        
                    idx_str, result_str = line.split(',', 1)
                    # Extract just the number from the index
                    idx = int(''.join(filter(str.isdigit, idx_str)))
                    # Clean up the result string
                    result_str = result_str.lower().strip()
                    results[idx] = result_str == 'true'
                except (ValueError, IndexError):
                    continue
                    
            # If we got no valid results, try one more time with a simpler format
            if not results:
                print("No valid results found, retrying with simpler format...")
                # Just look for true/false in each line
                for line in content.strip().split('\n'):
                    if 'true' in line.lower() or 'false' in line.lower():
                        try:
                            # Find the first number in the line
                            idx = int(''.join(filter(str.isdigit, line.split()[0])))
                            # Find if true or false appears in the line
                            results[idx] = 'true' in line.lower()
                        except (ValueError, IndexError):
                            continue
                            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print("Raw response:", response.content)
            
        return results

    async def filter_texts(self, texts: List[str], criteria: str, batch_size: int = 300) -> Dict[int, bool]:
        """Filters a list of texts based on the given criteria using batched processing."""
        all_results = {}
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Create list of (index, text) tuples for this batch
            batch_with_indices = [(idx, text) for idx, text in enumerate(batch_texts, start=i)]
            
            try:
                # Process batch and update results
                batch_results = await self.process_batch(batch_with_indices, criteria)
                if batch_results:  # Only update if we got valid results
                    all_results.update(batch_results)
                else:
                    print(f"Warning: No valid results for batch starting at index {i}")
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {str(e)}")
                continue
        
        # If we got no results at all, return all texts as relevant
        if not all_results:
            print("Warning: No valid results across all batches, accepting all texts")
            return {i: True for i in range(len(texts))}
            
        return all_results

class Classifier:
    """Classifies texts into user-defined classes."""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Using pro for more reliable classification
            temperature=0
        )
        
    def create_classification_prompt(self, classes: List[str], is_multi_class: bool = False) -> ChatPromptTemplate:
        """Creates a prompt for classifying texts into predefined classes."""
        template = """You are a precise text classifier specializing in social media content analysis.

        Your task is to classify each text into {classification_type} from the following list:
        {class_list}
        - Other (for texts that don't clearly fit into any of the above classes)

        For each text, respond with its index followed by the class name(s), separated by a comma.
        {format_instruction}
        
        Example format:
        {example_format}
        
        Texts to classify:
        {text_batch}
        
        Important:
        1. Use ONLY the exact class names provided above (including 'Other')
        2. {class_rule}
        3. If a text doesn't clearly fit any class, assign it to 'Other'
        4. Respond with ONLY the index and class assignment(s) for each text, one per line
        """
        
        return ChatPromptTemplate.from_template(template)
        
    def format_batch(self, texts_with_indices: List[Tuple[int, str]]) -> str:
        """Format a batch of texts with their indices for the prompt."""
        return "\n".join([f"Index {idx}: {text}" for idx, text in texts_with_indices])
        
    async def process_batch(self, texts_with_indices: List[Tuple[int, str]], classes: List[str], is_multi_class: bool) -> Dict[int, List[str]]:
        """Process a batch of texts and return their classifications."""
        prompt = self.create_classification_prompt(classes, is_multi_class)
        
        # Format the batch text
        batch_text = self.format_batch(texts_with_indices)
        
        # Prepare prompt variables
        if is_multi_class:
            format_instruction = "For multiple classes, separate them with '|' after the index"
            example_format = "0,Class A|Class B\n1,Class C\n2,Other"
            class_rule = "A text can belong to multiple classes if it clearly fits them"
            classification_type = "multiple classes"
        else:
            format_instruction = "Assign exactly one class that best fits the text"
            example_format = "0,Class A\n1,Class B\n2,Other"
            class_rule = "Choose the SINGLE most appropriate class for each text"
            classification_type = "exactly one class"
            
        # Get classifications from LLM
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "text_batch": batch_text,
            "class_list": "\n".join([f"- {cls}" for cls in classes]),
            "classification_type": classification_type,
            "format_instruction": format_instruction,
            "example_format": example_format,
            "class_rule": class_rule
        })
        
        # Parse response into dictionary
        results = {}
        valid_classes = set(classes + ['Other'])
        
        for line in response.content.strip().split('\n'):
            try:
                idx_str, class_str = line.strip().split(',', 1)
                idx = int(idx_str)
                
                if is_multi_class:
                    # For multi-class, split on | and validate each class
                    assigned_classes = [cls.strip() for cls in class_str.split('|')]
                    valid_assignments = [cls for cls in assigned_classes if cls in valid_classes]
                    if not valid_assignments:  # If no valid classes found, assign to Other
                        valid_assignments = ['Other']
                    results[idx] = valid_assignments
                else:
                    # For single-class, validate the assigned class
                    assigned_class = class_str.strip()
                    if assigned_class not in valid_classes:
                        assigned_class = 'Other'
                    results[idx] = [assigned_class]
                    
            except (ValueError, IndexError):
                # If there's any error parsing, assign to Other
                if 'idx' in locals():
                    results[idx] = ['Other']
                continue
                
        return results

    async def classify_texts(
        self, 
        texts: List[str], 
        classes: List[str], 
        is_multi_class: bool = False,
        batch_size: int = 300
    ) -> Dict[str, List[Dict]]:
        """
        Classify texts into predefined classes.
        
        Args:
            texts: List of texts to classify
            classes: List of class names to use for classification
            is_multi_class: Whether a text can belong to multiple classes
            batch_size: Number of texts to process in each batch
            
        Returns:
            Dictionary with class names as keys and lists of matching texts as values
        """
        all_results = {}
        
        # Process texts in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Classifying texts"):
            batch_texts = texts[i:i + batch_size]
            # Create list of (index, text) tuples for this batch
            batch_with_indices = [(idx, text) for idx, text in enumerate(batch_texts, start=i)]
            
            # Process batch and update results
            batch_results = await self.process_batch(batch_with_indices, classes, is_multi_class)
            all_results.update(batch_results)
        
        # Organize results by class
        results_by_class = {cls: [] for cls in classes + ['Other']}
        
        for idx, assigned_classes in all_results.items():
            text_entry = {
                'text': texts[idx],
                'index': idx,
                'classes': assigned_classes
            }
            # Add the text to each of its assigned classes
            for cls in assigned_classes:
                results_by_class[cls].append(text_entry)
        
        # Add summary statistics
        summary = {
            'total_texts': len(texts),
            'class_distribution': {
                cls: len(entries) for cls, entries in results_by_class.items()
            }
        }
        results_by_class['summary'] = summary
        
        return results_by_class 

class DataWizard:
    """Analyzes data based on user questions using filtering and batch processing."""
    
    def __init__(self, use_semantic_filter: bool = True):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            convert_system_message_to_human=True
        )
        self.semantic_filter = SemanticFilter() if use_semantic_filter else LLMFilter()
        
    def create_analysis_prompt(self) -> ChatPromptTemplate:
        """Creates a prompt for analyzing filtered data."""
        template = """You are an expert data analyst specializing in social media content analysis.
        
        Your task is to analyze the following collection of social media comments and provide insights related to this question: {question}
        
        Comments to analyze:
        {comments}
        
        You must respond with valid JSON only, using this exact format:
        {{
            "insights": [
                {{
                    "observation": "Key observation or finding",
                    "evidence": ["Supporting comment 1", "Supporting comment 2"],
                    "implications": "What this means for the business/users",
                    "confidence": "High/Medium/Low based on data quality and quantity"
                }}
            ],
            "summary": "Overall summary of findings",
            "recommendations": [
                "Action-oriented recommendation 1",
                "Action-oriented recommendation 2"
            ]
        }}
        
        Important: Your entire response must be valid JSON. Do not include any other text before or after the JSON.
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def create_merge_prompt(self) -> ChatPromptTemplate:
        """Creates a prompt for merging multiple analysis results."""
        template = """You are an expert at synthesizing analytical findings.
        
        Your task is to analyze multiple sets of insights and merge them into a comprehensive analysis.
        
        Previous analyses to merge:
        {previous_analyses}
        
        Original question: {question}
        
        You must respond with valid JSON only, using this exact format:
        {{
            "insights": [
                {{
                    "observation": "Key observation or finding",
                    "evidence": ["Supporting comment 1", "Supporting comment 2"],
                    "implications": "What this means for the business/users",
                    "confidence": "High/Medium/Low based on data quality and quantity"
                }}
            ],
            "summary": "Overall summary of findings",
            "recommendations": [
                "Action-oriented recommendation 1",
                "Action-oriented recommendation 2"
            ]
        }}
        
        Important:
        1. Combine similar insights while preserving unique perspectives
        2. Ensure the summary captures the full scope of findings
        3. Prioritize recommendations based on evidence strength
        4. Your entire response must be valid JSON with no other text
        """
        
        return ChatPromptTemplate.from_template(template)
    
    async def process_batch(self, texts: List[str], question: str) -> Dict:
        """Process a batch of texts to generate insights."""
        prompt = self.create_analysis_prompt()
        
        # Format texts for the prompt
        formatted_texts = "\n".join([f"- {text}" for text in texts[:100]])  # Limit to 100 examples
        if len(texts) > 100:
            formatted_texts += f"\n... and {len(texts) - 100} more comments"
        
        # Get analysis from LLM
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "question": question,
            "comments": formatted_texts
        })
        
        # Parse JSON response
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
            required_keys = {'insights', 'summary', 'recommendations'}
            if not all(key in result for key in required_keys):
                raise ValueError("Response missing required keys")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print("Raw response:", response.content)
            raise ValueError("Could not parse LLM response as JSON")
        except Exception as e:
            print(f"Validation Error: {str(e)}")
            print("Parsed response:", result if 'result' in locals() else 'No result')
            raise ValueError(f"Invalid response structure: {str(e)}")
    
    async def merge_results(self, results: List[Dict], question: str) -> Dict:
        """Merge multiple analysis results into a comprehensive analysis."""
        if not results:
            return {
                "insights": [],
                "summary": "No analysis results to merge",
                "recommendations": []
            }
            
        if len(results) == 1:
            return results[0]
            
        prompt = self.create_merge_prompt()
        
        # Get merged analysis from LLM
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "previous_analyses": json.dumps(results, ensure_ascii=False, indent=2),
            "question": question
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
            required_keys = {'insights', 'summary', 'recommendations'}
            if not all(key in result for key in required_keys):
                raise ValueError("Response missing required keys")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error in merge: {str(e)}")
            print("Raw response:", response.content)
            return results[0]  # Return first result if merge fails
        except Exception as e:
            print(f"Validation Error in merge: {str(e)}")
            return results[0]  # Return first result if merge fails
    
    async def analyze_question(self, texts: List[str], question: str, batch_size: int = 10000) -> Dict:
        """
        Analyze texts based on a user question.
        
        Args:
            texts: List of texts to analyze
            question: User's question or analysis request
            batch_size: Number of texts to process in each batch
            
        Returns:
            Dictionary containing insights, summary, and recommendations
        """
        print("Filtering relevant texts...")
        if isinstance(self.semantic_filter, SemanticFilter):
            results, _ = await self.semantic_filter.filter_texts(texts, question)
        else:
            results = await self.semantic_filter.filter_texts(texts, question)
            
        # Get relevant texts
        relevant_texts = [text for i, text in enumerate(texts) if results.get(i, False)]
        print(f"\nFound {len(relevant_texts)} relevant texts")
        
        if not relevant_texts:
            return {
                "insights": [],
                "summary": "No relevant texts found for the given question",
                "recommendations": ["Consider rephrasing the question or adjusting the filtering criteria"]
            }
        
        print("\nGenerating insights...")
        all_results = []
        
        # Process texts in batches
        for i in tqdm(range(0, len(relevant_texts), batch_size), desc="Analyzing batches"):
            batch = relevant_texts[i:i + batch_size]
            try:
                batch_result = await self.process_batch(batch, question)
                all_results.append(batch_result)
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue
        
        # Merge results
        final_result = await self.merge_results(all_results, question)
        return final_result 

class ComparativeAnalyzer:
    """Performs general comparative analysis between two datasets based on user questions."""
    
    def __init__(self, use_semantic_filter: bool = True):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            convert_system_message_to_human=True,
            safety_settings={
                "HARASSMENT": "block_none",
                "HATE_SPEECH": "block_none",
                "SEXUALLY_EXPLICIT": "block_none",
                "DANGEROUS_CONTENT": "block_none"
            },
            generation_config={
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "candidate_count": 1,
            }
        )
        self.semantic_filter = SemanticFilter() if use_semantic_filter else LLMFilter()
        
    def create_analysis_prompt(self) -> ChatPromptTemplate:
        """Creates a prompt for analyzing filtered data for one dataset."""
        template = """You are an expert data analyst specializing in data analysis and pattern recognition.
        
        Dataset Context: {dataset_description}
        
        Your task is to analyze the following collection of data points and provide insights related to this question: {question}
        
        Data to analyze:
        {comments}
        
        You must respond with valid JSON only, using this exact format:
        {{
            "insights": [
                {{
                    "observation": "Key observation or finding",
                    "evidence": ["Supporting data point 1", "Supporting data point 2"],
                    "implications": "What this means in the context of the analysis",
                    "confidence": "High/Medium/Low based on data quality and quantity"
                }}
            ],
            "summary": "Overall summary of findings",
            "key_metrics": {{
                "distribution_metrics": {{"category1": "x%", "category2": "y%"}},
                "main_themes": ["theme1", "theme2", "theme3"],
                "relevance_score": "score indicating data relevance to question"
            }},
            "notable_patterns": ["Pattern 1", "Pattern 2"]
        }}
        
        Important: Your entire response must be valid JSON. Do not include any other text before or after the JSON.
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def create_comparison_prompt(self) -> ChatPromptTemplate:
        """Creates a prompt for comparing analyses of two datasets."""
        template = """You are an expert at comparative data analysis.
        
        Your task is to compare the analyses of two datasets and provide a comprehensive comparison.
        
        Dataset 1 ({dataset1_name}):
        {dataset1_analysis}
        
        Dataset 2 ({dataset2_name}):
        {dataset2_analysis}
        
        Original question for comparison: {question}
        
        You must respond with valid JSON only, using this exact format:
        {{
            "comparative_insights": [
                {{
                    "aspect": "Aspect being compared",
                    "dataset1_position": "How dataset1 presents in this aspect",
                    "dataset2_position": "How dataset2 presents in this aspect",
                    "key_differences": "Main differences between the two",
                    "implications": "What these differences mean for the analysis"
                }}
            ],
            "summary": "Overall comparative summary",
            "metric_comparisons": {{
                "distributions": {{
                    "dataset1": "summary of dataset1 distributions",
                    "dataset2": "summary of dataset2 distributions",
                    "differences": "key distribution differences"
                }},
                "patterns": {{
                    "common_patterns": ["pattern1", "pattern2"],
                    "unique_to_dataset1": ["pattern1", "pattern2"],
                    "unique_to_dataset2": ["pattern1", "pattern2"]
                }}
            }},
            "key_findings": {{
                "similarities": ["similarity1", "similarity2"],
                "differences": ["difference1", "difference2"]
            }},
            "recommendations": [
                "General recommendation 1 based on the comparison",
                "General recommendation 2 based on the comparison"
            ]
        }}
        
        Important:
        1. Focus on meaningful comparisons and contrasts
        2. Support insights with evidence from both datasets
        3. Be objective and data-driven
        4. Your entire response must be valid JSON with no other text
        """
        
        return ChatPromptTemplate.from_template(template)

    async def process_batch(self, texts: List[str], question: str, dataset_description: str) -> Dict:
        """Process a batch of texts to generate insights for one dataset."""
        prompt = self.create_analysis_prompt()
        
        # Format texts for the prompt
        formatted_texts = "\n".join([f"- {text}" for text in texts[:100]])  # Limit to 100 examples
        if len(texts) > 100:
            formatted_texts += f"\n... and {len(texts) - 100} more comments"
        
        # Get analysis from LLM
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "question": question,
            "comments": formatted_texts,
            "dataset_description": dataset_description
        })
        
        # Parse JSON response
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
            required_keys = {'insights', 'summary', 'key_metrics', 'notable_patterns'}
            if not all(key in result for key in required_keys):
                raise ValueError("Response missing required keys")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {str(e)}")
            print("Raw response:", response.content)
            raise ValueError("Could not parse LLM response as JSON")
        except Exception as e:
            print(f"Validation Error: {str(e)}")
            print("Parsed response:", result if 'result' in locals() else 'No result')
            raise ValueError(f"Invalid response structure: {str(e)}")
    
    async def merge_dataset_results(self, results: List[Dict], question: str, dataset_description: str) -> Dict:
        """Merge multiple analysis results for one dataset."""
        if not results:
            return {
                "insights": [],
                "summary": "No analysis results to merge",
                "key_metrics": {
                    "distribution_metrics": {},
                    "main_themes": [],
                    "relevance_score": "0"
                },
                "notable_patterns": []
            }
            
        if len(results) == 1:
            return results[0]
            
        # For merging dataset results, we'll reuse the analysis prompt but with merged data
        prompt = self.create_analysis_prompt()
        
        # Get merged analysis from LLM
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "question": question,
            "comments": "Previous analyses to merge:\n" + json.dumps(results, ensure_ascii=False, indent=2),
            "dataset_description": dataset_description
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
            required_keys = {'insights', 'summary', 'key_metrics', 'notable_patterns'}
            if not all(key in result for key in required_keys):
                raise ValueError("Response missing required keys")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error in merge: {str(e)}")
            print("Raw response:", response.content)
            return results[0]  # Return first result if merge fails
        except Exception as e:
            print(f"Validation Error in merge: {str(e)}")
            return results[0]  # Return first result if merge fails
    
    async def compare_analyses(self, dataset1_analysis: Dict, dataset2_analysis: Dict, 
                             dataset1_name: str, dataset2_name: str, question: str) -> Dict:
        """Compare analyses of two datasets."""
        prompt = self.create_comparison_prompt()
        
        # Get comparative analysis from LLM
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "dataset1_analysis": json.dumps(dataset1_analysis, ensure_ascii=False, indent=2),
            "dataset2_analysis": json.dumps(dataset2_analysis, ensure_ascii=False, indent=2),
            "dataset1_name": dataset1_name,
            "dataset2_name": dataset2_name,
            "question": question
        })
        
        # Parse response
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
            required_keys = {'comparative_insights', 'summary', 'metric_comparisons', 
                           'key_findings', 'recommendations'}
            if not all(key in result for key in required_keys):
                raise ValueError("Response missing required keys")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error in comparison: {str(e)}")
            print("Raw response:", response.content)
            raise ValueError("Could not parse LLM response as JSON")
        except Exception as e:
            print(f"Validation Error in comparison: {str(e)}")
            raise ValueError(f"Invalid response structure: {str(e)}")
    
    async def analyze_datasets(self, config: Dict, question: str, batch_size: int = 10000) -> Dict:
        """
        Analyze and compare two datasets based on a question.
        """
        results = {}
        errors = []
        dataset_stats = {}
        
        # Process each dataset
        for dataset_key in ["dataset1", "dataset2"]:
            dataset_config = config[dataset_key]
            print(f"\n{'='*50}")
            print(f"Processing {dataset_key}...")
            print(f"Reading data from: {dataset_config['file']}")
            
            try:
                # Read data
                df = pd.read_excel(dataset_config['file'])
                texts = df['Post/comments'].fillna('').tolist()
                dataset_stats[dataset_key] = {
                    'total_texts': len(texts)
                }
                
                # Log first 10 rows
                print(f"\nFirst 10 rows of {dataset_key}:")
                for i, text in enumerate(texts[:10]):
                    print(f"{i+1}. {text[:100]}...")
                
                # Filter relevant texts
                print(f"\nFiltering relevant texts for {dataset_key}...")
                print(f"Filter type: {'Semantic' if isinstance(self.semantic_filter, SemanticFilter) else 'LLM'}")
                
                try:
                    if isinstance(self.semantic_filter, SemanticFilter):
                        filter_results, similarities = await self.semantic_filter.filter_texts(texts, question)
                        print(f"Semantic filter threshold: {self.semantic_filter.last_threshold:.3f}")
                    else:
                        filter_results = await self.semantic_filter.filter_texts(texts, question)
                    
                    relevant_texts = [text for i, text in enumerate(texts) if filter_results.get(i, False)]
                    dataset_stats[dataset_key]['filtered_texts'] = len(relevant_texts)
                    
                    print(f"\nFiltering results for {dataset_key}:")
                    print(f"Total texts: {len(texts)}")
                    print(f"Filtered texts: {len(relevant_texts)}")
                    print(f"Filter ratio: {len(relevant_texts)/len(texts)*100:.1f}%")
                    
                    if relevant_texts:
                        print("\nFirst 5 filtered texts:")
                        for i, text in enumerate(relevant_texts[:5]):
                            print(f"{i+1}. {text[:100]}...")
                    
                except Exception as filter_error:
                    error_msg = f"Error during filtering for {dataset_key}: {str(filter_error)}"
                    print(f"ERROR: {error_msg}")
                    errors.append(error_msg)
                    continue
                
                if len(relevant_texts) < 5:  # Minimum threshold for analysis
                    error_msg = f"Insufficient relevant texts found for {dataset_key}. "
                    error_msg += f"Only {len(relevant_texts)} texts matched out of {len(texts)} total texts. "
                    error_msg += "Try adjusting your question to be more general or use different filter criteria."
                    print(f"ERROR: {error_msg}")
                    errors.append(error_msg)
                    continue
                
                # Process texts in batches
                print(f"\nGenerating insights for {dataset_key}...")
                batch_results = []
                
                for i in tqdm(range(0, len(relevant_texts), batch_size), desc=f"Analyzing {dataset_key}"):
                    batch = relevant_texts[i:i + batch_size]
                    try:
                        batch_result = await self.process_batch(
                            batch, 
                            question, 
                            dataset_config['description']
                        )
                        batch_results.append(batch_result)
                        
                        # Log first batch result structure
                        if i == 0:
                            print(f"\nSample batch result structure for {dataset_key}:")
                            print(json.dumps(batch_result, indent=2)[:500] + "...")
                            
                    except Exception as batch_error:
                        error_msg = f"Error processing batch {i//batch_size + 1}: {str(batch_error)}"
                        print(f"ERROR: {error_msg}")
                        continue
                
                # Merge results for this dataset
                if batch_results:
                    try:
                        results[dataset_key] = await self.merge_dataset_results(
                            batch_results,
                            question,
                            dataset_config['description']
                        )
                        print(f"\nSuccessfully merged results for {dataset_key}")
                        print(f"Number of insights: {len(results[dataset_key].get('insights', []))}")
                        print(f"Has key_metrics: {'key_metrics' in results[dataset_key]}")
                        print(f"Has notable_patterns: {'notable_patterns' in results[dataset_key]}")
                    except Exception as merge_error:
                        error_msg = f"Error merging results for {dataset_key}: {str(merge_error)}"
                        print(f"ERROR: {error_msg}")
                        errors.append(error_msg)
                else:
                    error_msg = f"Failed to process any batches for {dataset_key}. "
                    error_msg += "This might indicate an issue with the analysis model."
                    print(f"ERROR: {error_msg}")
                    errors.append(error_msg)
                
            except Exception as e:
                error_msg = f"Error processing {dataset_key}: {str(e)}"
                print(f"ERROR: {error_msg}")
                errors.append(error_msg)
                continue
        
        # If we don't have results for both datasets, return error with details
        if len(results) < 2:
            error_response = {
                "error": "Insufficient data for comparison",
                "message": "Could not get results for both datasets",
                "details": {
                    "errors": errors,
                    "dataset_stats": dataset_stats,
                    "available_results": results,
                    "recommendations": [
                        "Try adjusting your question to be more general",
                        "Check if your datasets contain relevant information for the comparison",
                        "Consider using a different filter type (semantic vs LLM)",
                        "Ensure both datasets have sufficient data for comparison"
                    ]
                }
            }
            print("\nError Response:")
            print(json.dumps(error_response, indent=2))
            return error_response
        
        # Compare the results
        print("\nGenerating comparative analysis...")
        try:
            comparison = await self.compare_analyses(
                results["dataset1"],
                results["dataset2"],
                config["dataset1"]["description"],
                config["dataset2"]["description"],
                question
            )
            
            # Add individual analyses and stats to the result
            comparison["individual_analyses"] = results
            comparison["dataset_stats"] = dataset_stats
            
            print("\nSuccessful comparison:")
            print(f"Number of comparative insights: {len(comparison.get('comparative_insights', []))}")
            print(f"Has metric_comparisons: {'metric_comparisons' in comparison}")
            print(f"Has key_findings: {'key_findings' in comparison}")
            
            return comparison
            
        except Exception as e:
            error_response = {
                "error": "Comparison failed",
                "message": str(e),
                "details": {
                    "errors": errors + [f"Comparison error: {str(e)}"],
                    "dataset_stats": dataset_stats,
                    "available_results": results,
                    "recommendations": [
                        "Try simplifying your comparison question",
                        "Check if the data in both datasets is comparable",
                        "Ensure the question is relevant to both datasets"
                    ]
                }
            }
            print("\nError Response:")
            print(json.dumps(error_response, indent=2))
            return error_response

class CompetitiveAnalyzer(ComparativeAnalyzer):
    """Specialized analyzer for comparing competing businesses/services."""
    
    def __init__(self, use_semantic_filter: bool = True):
        super().__init__(use_semantic_filter)  # Use parent class initialization

    def create_analysis_prompt(self) -> ChatPromptTemplate:
        """Creates a prompt for analyzing filtered data for one competitor."""
        template = """You are an expert data analyst specializing in competitive analysis.
        
        Dataset Context: {dataset_description}
        
        Your task is to analyze the following collection of customer feedback and provide competitive insights related to this question: {question}
        
        Comments to analyze:
        {comments}
        
        You must respond with valid JSON only, using this exact format:
        {{
            "insights": [
                {{
                    "observation": "Key observation or finding",
                    "evidence": ["Supporting comment 1", "Supporting comment 2"],
                    "implications": "What this means for competitive positioning",
                    "confidence": "High/Medium/Low based on data quality and quantity"
                }}
            ],
            "summary": "Overall summary of findings",
            "key_metrics": {{
                "sentiment_distribution": {{"positive": "x%", "negative": "y%", "neutral": "z%"}},
                "main_topics": ["topic1", "topic2", "topic3"],
                "user_satisfaction_score": "score out of 10"
            }},
            "strengths": ["Competitive strength 1", "Competitive strength 2"],
            "weaknesses": ["Competitive weakness 1", "Competitive weakness 2"]
        }}
        
        Important: Your entire response must be valid JSON. Do not include any other text before or after the JSON.
        """
        
        return ChatPromptTemplate.from_template(template)
    
    def create_comparison_prompt(self) -> ChatPromptTemplate:
        """Creates a prompt for comparing competitive analyses."""
        template = """You are an expert at competitive analysis.
        
        Your task is to compare the analyses of two competitors and provide a comprehensive competitive comparison.
        
        Competitor 1 ({dataset1_name}):
        {dataset1_analysis}
        
        Competitor 2 ({dataset2_name}):
        {dataset2_analysis}
        
        Original question for comparison: {question}
        
        You must respond with valid JSON only, using this exact format:
        {{
            "comparative_insights": [
                {{
                    "aspect": "Competitive aspect being compared",
                    "competitor1_position": "How competitor1 performs in this aspect",
                    "competitor2_position": "How competitor2 performs in this aspect",
                    "key_differences": "Main competitive differences",
                    "implications": "What these differences mean for market position"
                }}
            ],
            "summary": "Overall competitive comparison summary",
            "key_metrics_comparison": {{
                "sentiment": {{
                    "competitor1": "summary of competitor1 sentiment",
                    "competitor2": "summary of competitor2 sentiment",
                    "difference": "key differences in sentiment"
                }},
                "user_satisfaction": {{
                    "competitor1_score": "x/10",
                    "competitor2_score": "y/10",
                    "analysis": "comparison of satisfaction scores"
                }}
            }},
            "competitive_advantages": {{
                "competitor1": ["advantage1", "advantage2"],
                "competitor2": ["advantage1", "advantage2"]
            }},
            "recommendations": {{
                "competitor1": ["strategic recommendation1", "strategic recommendation2"],
                "competitor2": ["strategic recommendation1", "strategic recommendation2"]
            }}
        }}
        
        Important:
        1. Focus on meaningful competitive comparisons
        2. Support insights with evidence from both datasets
        3. Be objective and data-driven
        4. Your entire response must be valid JSON with no other text
        """
        
        return ChatPromptTemplate.from_template(template) 