import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from loguru import logger
import sys
from artemis_client.vision.client import VisionAsyncClient, VisionSettings
from artemis_client.falcon.client import ThanosSettings, FalconSettings, FalconClient
from vision_models import LLMInferenceRequest, LLMConversationMessage, LLMRole
from vision_models.service.llm import LLMType
from dotenv import load_dotenv
from elommr import EloMMR, Player as EloPlayer
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Define available LLMs
AVAILABLE_LLMS = [
    LLMType("gpt-4-o-mini"),
    LLMType("gemini-v15-flash"),
    LLMType("llama-3-1-8b"),
    LLMType("gpt-4-o"),
    LLMType("claude-v35-sonnet"),
    LLMType("claude-v37-sonnet")
]

# Define optimization tasks
OPTIMIZATION_TASKS = {
    "runtime_performance": {
        "description": "Optimize code for better runtime performance while maintaining functionality",
        "objective": "improving runtime performance",
        "default_prompt": "Improve the performance of the provided code. Try to find ways to reduce runtime, while keeping the main functionality of the code unchanged.",
        "instruction": "Generate an optimized version of the code that improves runtime performance while maintaining the same functionality.",
        "data_format": "\n\nOriginal Code:\n{}\n",
        "considerations": """1. Algorithmic complexity (Big O notation)
2. Data structure efficiency and access patterns
3. Loop optimizations and unnecessary iterations
4. Memory access patterns and caching
5. I/O operations and system calls
6. Parallel processing opportunities
7. Redundant computations"""
    },
    "memory_usage": {
        "description": "Optimize code for reduced memory consumption",
        "objective": "reducing memory usage",
        "default_prompt": "Improve the performance of the provided code. Try to find ways to reduce memory usage, while keeping the main functionality of the code unchanged.",
        "instruction": "Generate an optimized version of the code that reduces memory consumption while maintaining the same functionality.",
        "data_format": "\n\nOriginal Code:\n{}\n",
        "considerations": """1. Memory allocation and deallocation patterns
2. Memory leaks and resource cleanup
3. Data structure memory footprint
4. Buffer sizes and memory pools
5. Memory fragmentation
6. Garbage collection impact
7. Shared memory usage"""
    }
}

# Define the meta-prompt template
META_PROMPT_TEMPLATE = """
You are an expert in code optimization. We need to generate a prompt that will help an LLM optimize code for {objective}.

## Project Context
Project Name: {project_name}
Project Description: {project_description}
Primary Languages: {project_languages}

## Task Context
{task_description}

## Current Prompt
{current_prompt}

## Target LLM
{target_llm}

## Instructions
1. Analyze the current prompt, project context, and task context
2. Consider the target LLM's capabilities and limitations
3. Generate an improved prompt that will instruct the LLM to optimize code for {objective}
4. The prompt should be specific, clear, and focused on {objective} 
5. Your response should contain only the improved prompt text, without any placeholders, formatting instructions, or additional text.
6. The generated prompt should also not contain any additional text like placeholders or formatting instructions, and should not ask the LLM to explain the optimization.
"""

JUDGE_PROMPT_TEMPLATE = """You are an expert in code optimization and performance analysis. Compare the following two code snippets and determine which one would be better for {objective}.

## Task Context
{task_description}

Code A:
```python
{code_a}
```

Code B:
```python
{code_b}
```

Consider the following aspects specific to {objective}:
{task_considerations}

Respond with ONLY ONE of these exact strings:
- "A" if Code A is likely to be better for {objective}
- "B" if Code B is likely to be better for {objective}
- "TIE" if both codes would have similar performance for {objective}

Your response should contain only A, B, or TIE, nothing else."""

class LLMJudge:
    def __init__(self, vision_client: VisionAsyncClient, llm_type: LLMType, task_name: str):
        self.vision_client = vision_client
        self.llm_type = llm_type
        self.task_name = task_name
        
    async def compare_code(self, code_a: str, code_b: str) -> float:
        """Compare two code snippets and return a score between 0 and 1.
        Returns:
        - 1.0 if code_a is better
        - 0.0 if code_b is better
        - 0.5 if tie
        """
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            code_a=code_a,
            code_b=code_b,
            objective=OPTIMIZATION_TASKS[self.task_name]["objective"],
            task_description=OPTIMIZATION_TASKS[self.task_name]["description"],
            task_considerations=OPTIMIZATION_TASKS[self.task_name]["considerations"]
        )
        
        request = LLMInferenceRequest(
            model_type=self.llm_type,
            messages=[LLMConversationMessage(role=LLMRole.USER, content=prompt)]
        )

        try:
            response = await self.vision_client.ask(request)
            if not response.messages or len(response.messages) < 2:
                logger.error("No response received from LLM")
                return 0.5  # Return tie in case of error
                
            result = response.messages[1].content.strip().upper()
            
            # Convert response to score
            if result == "A":
                return 1.0  # code_a is better
            elif result == "B":
                return 0.0  # code_b is better
            else:  # "TIE" or any other response
                return 0.5  # tie
                
        except Exception as e:
            logger.error(f"Error during code comparison: {str(e)}")
            return 0.5  # Return tie in case of error

class CodeRatingSystem:
    def __init__(self):
        self.elo_mmr = EloMMR()
        self.players = {}
        
    def get_or_create_player(self, code_id: str) -> EloPlayer:
        """Get existing player or create a new one"""
        if code_id not in self.players:
            self.players[code_id] = EloPlayer()
        return self.players[code_id]
        
    def update_ratings(self, standings: List[Tuple[str, int, int]], contest_time: Optional[int] = None):
        """Update ratings based on standings
        standings: List of (code_id, place_start, place_end) tuples
        """
        if contest_time is None:
            contest_time = round(datetime.now().timestamp())
            
        # Convert standings to elommr format
        elo_standings = []
        for code_id, place_start, place_end in standings:
            player = self.get_or_create_player(code_id)
            elo_standings.append((player, place_start, place_end))
            
        # Update ratings
        self.elo_mmr.round_update(elo_standings, contest_time)
        
    def get_rating(self, code_id: str) -> float:
        """Get current rating for a code version"""
        if code_id not in self.players:
            return 1500.0  # Default rating
        player = self.players[code_id]
        if not player.event_history:
            return 1500.0
        # Convert the display rating to float by extracting the mean value
        rating_str = player.event_history[-1].display_rating()
        # Extract the numeric part before the ± symbol
        rating_value = float(rating_str.split('±')[0].strip())
        return rating_value

class BenchmarkEvaluator:
    # Define fixed system prompt for code output
    SYSTEM_PROMPT = """
IMPORTANT: Your response must contain ONLY the optimized code itself, with NO additional text before and after the code, and with NO code block markers (```, ```cpp, etc.)
"""

    def __init__(self,
                 task_name: str,
                 llm_type: LLMType,
                 judge_llm_type: Optional[LLMType] = None,
                 current_prompt: Optional[str] = None,
                 custom_task_description: Optional[str] = None,
                 custom_meta_prompt: Optional[str] = None,
                 optimisation_id: Optional[str] = "49b08c56-620f-4ae8-96d3-1675e6a17b2a",
                 progress_callback: Optional[callable] = None):
        self.task_name = task_name
        self.llm_type = llm_type
        self.judge_llm_type = judge_llm_type or LLMType("gpt-4-o")
        self.task = OPTIMIZATION_TASKS[task_name]
        self.current_prompt = current_prompt or self.task["default_prompt"]
        self.custom_task_description = custom_task_description
        self.custom_meta_prompt = custom_meta_prompt
        self.optimisation_id = optimisation_id
        self.progress_callback = progress_callback
        self.vision_async_client = None
        self.falcon_client = None
        self.llm_judge = None
        self.rating_system = CodeRatingSystem()
        self.filled_meta_prompt = None

    async def setup_clients(self):
        """Initialize API clients"""
        if self.progress_callback:
            self.progress_callback({"status": "setup", "message": "Setting up API clients..."})
            
        # Setup Vision client
        vision_settings = VisionSettings.with_env_prefix("vision", _env_file=".env")
        thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
        self.vision_async_client = VisionAsyncClient(vision_settings, thanos_settings)
        self.llm_judge = LLMJudge(self.vision_async_client, self.judge_llm_type, self.task_name)

        # Setup Falcon client
        falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
        self.falcon_client = FalconClient(falcon_settings, thanos_settings)
        self.falcon_client.authenticate()
        
        if self.progress_callback:
            self.progress_callback({"status": "setup_complete"})

    async def get_optimisation_info(self) -> Dict[str, Any]:
        """Get information about the optimization run"""
        if not self.optimisation_id:
            return None

        try:
            # Get optimization details
            optimisation = self.falcon_client.get_optimisation(self.optimisation_id)
            
            # Get all solutions
            solutions = self.falcon_client.get_solutions(self.optimisation_id, per_page=-1)  # Get all solutions
            
            # Process solutions to get runtime information
            solution_info = []
            for solution in solutions.docs:
                # Get detailed solution info
                solution_detail = self.falcon_client.get_solution(str(solution.id))
                
                # Extract runtime information
                runtime_info = {
                    'solution_id': str(solution.id),
                    'name': solution.name,
                    'status': solution.status,
                    'runtime': None,
                    'memory_usage': None,
                    'error': None
                }
                
                # Try to get runtime metrics from solution results
                if hasattr(solution_detail, 'results') and solution_detail.results:
                    results = solution_detail.results
                    if hasattr(results, 'metrics'):
                        metrics = results.metrics
                        if hasattr(metrics, 'runtime'):
                            runtime_info['runtime'] = metrics.runtime
                        if hasattr(metrics, 'memory_usage'):
                            runtime_info['memory_usage'] = metrics.memory_usage
                
                # Get error if any
                if hasattr(solution_detail, 'error') and solution_detail.error:
                    runtime_info['error'] = solution_detail.error
                
                solution_info.append(runtime_info)
            
            return {
                'optimisation_id': self.optimisation_id,
                'status': optimisation.status,
                'solutions': solution_info
            }
            
        except Exception as e:
            logger.error(f"Error getting optimization info: {str(e)}")
            return None

    async def generate_optimization_prompt(self, project_info: Dict[str, Any]) -> str:
        """Generate an optimized prompt using meta-prompting"""
        if self.progress_callback:
            self.progress_callback({"status": "generating_meta_prompt", "message": "Generating meta-prompt..."})
            
        # Use custom meta-prompt template if provided, otherwise use default
        meta_prompt_template = self.custom_meta_prompt or META_PROMPT_TEMPLATE
        
        # Store the filled meta-prompt for later use
        self.filled_meta_prompt = meta_prompt_template.format(
            objective=self.task["objective"],
            task_description=self.custom_task_description or self.task["description"],
            current_prompt=self.current_prompt,
            target_llm=self.llm_type,
            project_name=project_info.get("name", "Unknown"),
            project_description=project_info.get("description", "No description available"),
            project_languages=project_info.get("language", "unknown")
        )
        
        if self.progress_callback:
            self.progress_callback({
                "status": "meta_prompt_ready",
                "filled_meta_prompt": self.filled_meta_prompt
            })
        
        logger.info("Generating optimization prompt")
        if self.progress_callback:
            self.progress_callback({"status": "generating_prompt", "message": "Generating optimization prompt..."})
            
        request = LLMInferenceRequest(
            model_type=self.llm_type,
            messages=[LLMConversationMessage(role=LLMRole.USER, content=self.filled_meta_prompt)]
        )
        
        try:
            response = await self.vision_async_client.ask(request)
            generated_prompt = response.messages[1].content.strip()
            logger.info(f"Generated prompt: {generated_prompt}")
            
            if self.progress_callback:
                self.progress_callback({
                    "status": "prompt_ready",
                    "generated_prompt": generated_prompt
                })
                
            return generated_prompt
        except Exception as e:
            if self.progress_callback:
                self.progress_callback({"status": "error", "message": str(e)})
            return self.current_prompt

    async def optimize_code(self, code: str, prompt: str, max_retries: int = 2) -> Optional[str]:
        """Generate optimized code using the given prompt"""
        if self.progress_callback:
            self.progress_callback({
                "status": "optimizing_code",
                "message": "Optimizing code..."
            })
        
        for attempt in range(max_retries):
            try:
                # Format the prompt with the code and system prompt
                formatted_prompt = prompt + self.task["data_format"].format(code) + self.SYSTEM_PROMPT
                
                # Create LLM request
                request = LLMInferenceRequest(
                    model_type=self.llm_type,
                    messages=[LLMConversationMessage(role=LLMRole.USER, content=formatted_prompt)]
                )
                
                # Get response from LLM
                response = await self.vision_async_client.ask(request)
                if not response.messages or len(response.messages) < 2:
                    logger.error("No response received from LLM")
                    continue
                
                optimized_code = response.messages[1].content.strip()
                
                # Basic validation
                if not optimized_code or optimized_code == code:
                    logger.warning("LLM returned empty or unchanged code")
                    continue
                
                return optimized_code
                
            except Exception as e:
                logger.error(f"Error in optimization attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)  # Wait 5 seconds before retrying
                continue
        
        return None

    async def evaluate_benchmark(self, benchmark_file: str) -> Dict[str, Any]:
        """Evaluate a benchmark file"""
        try:
            # Load benchmark data
            with open(benchmark_file, 'r') as f:
                benchmark_data = json.load(f)
            
            # Setup clients
            await self.setup_clients()
            
            # Get optimization info if optimisation_id is provided
            optimisation_info = None
            if self.optimisation_id:
                if self.progress_callback:
                    self.progress_callback({"status": "getting_optimisation_info", "message": "Getting optimization information..."})
                optimisation_info = await self.get_optimisation_info()
            
            # Generate prompts using project info
            project_info = benchmark_data["metadata"]["project_info"]
            generated_prompt = await self.generate_optimization_prompt(project_info)
            
            results = []
            contest_time = round(datetime.now().timestamp())
            
            # Process each code snippet
            total_snippets = len(benchmark_data["code_snippets"])
            successful_snippets = 0
            
            for idx, snippet in enumerate(benchmark_data["code_snippets"], 1):
                if self.progress_callback:
                    self.progress_callback({
                        "status": "processing_snippet",
                        "message": f"Processing snippet {idx}/{total_snippets} (Successfully processed: {successful_snippets})",
                        "progress": idx / total_snippets
                    })
                
                # Get snippet language from metadata
                snippet_language = snippet.get("metadata", {}).get("language", project_info.get("language", "unknown"))
                
                # Generate optimized versions
                try:
                    baseline_code = await self.optimize_code(snippet["content"], self.current_prompt)
                    generated_code = await self.optimize_code(snippet["content"], generated_prompt)
                except Exception as e:
                    logger.error(f"Error optimizing snippet {snippet['id']}: {str(e)}")
                    continue
                
                # Skip if both optimizations failed
                if baseline_code is None and generated_code is None:
                    logger.warning(f"Both optimizations failed for snippet {snippet['id']}, skipping...")
                    continue
                
                # Create unique IDs for each code version
                snippet_id = snippet["id"]
                original_id = f"{snippet_id}_original"
                baseline_id = f"{snippet_id}_baseline"
                generated_id = f"{snippet_id}_generated"
                
                # Run comparisons in both orders
                comparison_orders = [
                    [
                        (original_id, generated_id, snippet["content"], generated_code),
                        (original_id, baseline_id, snippet["content"], baseline_code),
                        (baseline_id, generated_id, baseline_code, generated_code)
                    ],
                    [
                        (original_id, baseline_id, snippet["content"], baseline_code),
                        (original_id, generated_id, snippet["content"], generated_code),
                        (baseline_id, generated_id, baseline_code, generated_code)
                    ]
                ]
                
                all_comparison_results = []
                all_ratings = {
                    'original': [],
                    'baseline': [],
                    'generated': []
                }
                
                # Run each order
                for order_idx, comparisons in enumerate(comparison_orders):
                    # Reset rating system for each order
                    self.rating_system = CodeRatingSystem()
                    comparison_results = []
                    
                    # Only include comparisons where we have both codes
                    valid_comparisons = [
                        comp for comp in comparisons
                        if (comp[2] is not None and comp[3] is not None)
                    ]
                    
                    for code_a_id, code_b_id, code_a, code_b in valid_comparisons:
                        # Skip if either code is None
                        if code_a is None or code_b is None:
                            continue
                            
                        # Get LLM judge's verdict
                        score = await self.llm_judge.compare_code(code_a, code_b)
                        logger.info(f"Order {order_idx + 1}, Comparison {code_a_id} vs {code_b_id}: {score}")
                        
                        # Update ELO ratings based on comparison
                        if score == 1.0:  # code_a wins
                            self.rating_system.update_ratings([(code_a_id, 0, 0), (code_b_id, 1, 1)], contest_time)
                        elif score == 0.0:  # code_b wins
                            self.rating_system.update_ratings([(code_b_id, 0, 0), (code_a_id, 1, 1)], contest_time)
                        else:  # tie
                            self.rating_system.update_ratings([(code_a_id, 0, 0), (code_b_id, 0, 0)], contest_time)
                        
                        # Format comparison result
                        code_a_name = code_a_id.split('_')[-1]
                        code_b_name = code_b_id.split('_')[-1]
                        comparison_results.append({
                            'order': order_idx + 1,
                            'comparison': f"{code_a_name} vs {code_b_name}",
                            'score': score
                        })
                    
                    # Store ratings for this order
                    all_ratings['original'].append(self.rating_system.get_rating(original_id))
                    all_ratings['baseline'].append(self.rating_system.get_rating(baseline_id))
                    all_ratings['generated'].append(self.rating_system.get_rating(generated_id))
                    all_comparison_results.extend(comparison_results)
                    
                    # Update contest time for next iteration
                    contest_time += 1000
                
                # Calculate final results for this snippet
                result = {
                    'snippet_id': snippet_id,
                    'original_rating': np.mean(all_ratings['original']),
                    'baseline_rating': np.mean(all_ratings['baseline']),
                    'generated_rating': np.mean(all_ratings['generated']),
                    'ratings_by_order': {
                        'forward_order': {
                            'original': all_ratings['original'][0],
                            'baseline': all_ratings['baseline'][0],
                            'generated': all_ratings['generated'][0]
                        },
                        'reverse_order': {
                            'original': all_ratings['original'][1],
                            'baseline': all_ratings['baseline'][1],
                            'generated': all_ratings['generated'][1]
                        }
                    },
                    'comparisons': all_comparison_results,
                    'original_code': snippet["content"],
                    'baseline_code': baseline_code if baseline_code else "Failed to generate baseline optimization",
                    'generated_code': generated_code if generated_code else "Failed to generate meta-prompt optimization"
                }
                results.append(result)
                
                # Send progress update with the snippet result
                if self.progress_callback:
                    self.progress_callback({"status": "snippet_complete", "result": result})
                
                successful_snippets += 1
            
            # Calculate average ratings across all snippets
            if results:
                avg_ratings = {
                    'original': np.mean([r['original_rating'] for r in results]),
                    'baseline': np.mean([r['baseline_rating'] for r in results]),
                    'generated': np.mean([r['generated_rating'] for r in results])
                }
                
                avg_ratings_by_order = {
                    'forward_order': {
                        'original': np.mean([r['ratings_by_order']['forward_order']['original'] for r in results]),
                        'baseline': np.mean([r['ratings_by_order']['forward_order']['baseline'] for r in results]),
                        'generated': np.mean([r['ratings_by_order']['forward_order']['generated'] for r in results])
                    },
                    'reverse_order': {
                        'original': np.mean([r['ratings_by_order']['reverse_order']['original'] for r in results]),
                        'baseline': np.mean([r['ratings_by_order']['reverse_order']['baseline'] for r in results]),
                        'generated': np.mean([r['ratings_by_order']['reverse_order']['generated'] for r in results])
                    }
                }
            else:
                avg_ratings = {
                    'original': 1500.0,
                    'baseline': 1500.0,
                    'generated': 1500.0
                }
                avg_ratings_by_order = {
                    'forward_order': {'original': 1500.0, 'baseline': 1500.0, 'generated': 1500.0},
                    'reverse_order': {'original': 1500.0, 'baseline': 1500.0, 'generated': 1500.0}
                }
            
            # Prepare final results
            final_results = {
                'benchmark_info': benchmark_data['metadata'],
                'prompts': {
                    'baseline': self.current_prompt,
                    'generated': generated_prompt
                },
                'meta_prompt_used': self.filled_meta_prompt,
                'task_name': self.task_name,
                'results': results,
                'average_ratings': avg_ratings,
                'average_ratings_by_order': avg_ratings_by_order,
                'statistics': {
                    'total_snippets': total_snippets,
                    'successful_snippets': successful_snippets,
                    'failed_snippets': total_snippets - successful_snippets
                }
            }
            
            # Add optimization info if available
            if optimisation_info:
                final_results['optimisation_info'] = optimisation_info
            
            if self.progress_callback:
                self.progress_callback({
                    "status": "complete",
                    "final_results": final_results
                })
                
            return final_results
            
        except Exception as e:
            logger.error(f"Error evaluating benchmark: {str(e)}")
            if self.progress_callback:
                self.progress_callback({"status": "error", "message": str(e)})
            return None

def save_evaluation_results(results: Dict[str, Any], output_dir: str = "results") -> str:
    """Save evaluation results to a JSON file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename using project name and timestamp
    project_name = results["benchmark_info"]["project_info"]["name"].replace(" ", "_")
    task_name = results.get("task_name", "unknown_task")  # Get task name from results
    filename = f"{project_name}_{task_name}_{timestamp}_evaluation.json"
    filepath = os.path.join(output_dir, filename)
    
    # Add configuration information to results
    results["configuration"] = {
        "task_name": task_name,  # Use the same task name
        "llm_type": str(results.get("llm_type", "unknown")),
        "judge_llm_type": str(results.get("judge_llm_type", "unknown")),
        "baseline_prompt": results.get("prompts", {}).get("baseline", ""),
        "generated_prompt": results.get("prompts", {}).get("generated", ""),
        "meta_prompt_used": results.get("meta_prompt_used", ""),
        "custom_task_description": results.get("custom_task_description", ""),
        "custom_meta_prompt": results.get("custom_meta_prompt", ""),
        "evaluation_timestamp": timestamp
    }
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Results saved to: {filepath}")
    return filepath

def load_evaluation_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from a JSON file"""
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        logger.info(f"Successfully loaded results from: {filepath}")
        return results
    except Exception as e:
        logger.error(f"Error loading results from {filepath}: {str(e)}")
        return None

async def main():
    # Example usage
    evaluator = BenchmarkEvaluator(
        task_name="runtime_performance",
        llm_type=LLMType("gpt-4-o")
    )
    
    # Example benchmark file
    benchmark_file = "benchmarks/example_project_20240101_000000.json"
    
    # Run evaluation
    results = await evaluator.evaluate_benchmark(benchmark_file)
    
    if results:
        # Save results
        save_evaluation_results(results)

if __name__ == "__main__":
    asyncio.run(main()) 