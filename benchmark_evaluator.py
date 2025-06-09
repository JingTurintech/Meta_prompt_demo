import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from loguru import logger
import sys
from artemis_client.vision.client import VisionAsyncClient, VisionSettings
from artemis_client.falcon.client import ThanosSettings
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
        "description": "Optimize code for better runtime performance",
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

# Define meta-prompt templates
META_PROMPT_TEMPLATES = {
    "simplified": {
        "name": "Simplified Template",
        "description": "A concise, step-by-step template focusing on essential optimization goals",
        "template": """You are an expert in code optimization. We need to generate a prompt that will help the LLM {target_llm} optimize code for {objective}. 
        
NOTE: Your response should contain only the prompt, without any placeholders for the code, formatting instructions, or additional text. The generated prompt should not contain any additional text like placeholders for the code or formatting instructions.
"""
    },
    "standard": {
        "name": "Standard Template",
        "description": "A balanced template focusing on project context and optimization goals",
        "template": """You are an expert in code optimization. Please generate a prompt that will instruct the target LLM {target_llm} to optimize code for {objective}. Consider the project context, task context, and adapt the prompt complexity and style based on the target LLM's capabilities.

## Project Context
Project Name: {project_name}
Project Description: {project_description}
Primary Languages: {project_languages}

## Task Context
- Description: {task_description}

## Target LLM Context
- Target Model: {target_llm}
- For cost-efficient LLMs (e.g., gpt-4-o-mini, gemini-v15-flash, llama-3-1-8b): these models have limited internal chain-of-thought, so the generated prompt should give short, clear and succinct instructions, without internal reasoning.
- For larger LLMs (e.g., gpt-4-o, claude-v35-sonnet, claude-v37-sonnet): The generated prompt should allow for more complex and extensive internal reasoning, and encourage internal verification of any assumptions related to metrics based on the task description. 

NOTE: Your response should contain only the prompt, without any placeholders for the code, formatting instructions, or additional text. The generated prompt should not contain any additional text like placeholders for the code or formatting instructions.
"""
    },
    "enhanced": {
        "name": "Enhanced Template",
        "description": "A comprehensive template that includes detailed context about LLM capabilities and adapts the prompt accordingly",
        "template": """You are an expert in code optimization. Please generate a prompt that will instruct the target LLM {target_llm} to optimize code for {objective}. Consider the project context, task context, and adapt the prompt complexity and style based on the target LLM's capabilities.

## Project Context
Project Name: {project_name}
Project Description: {project_description}
Primary Languages: {project_languages}

## Task Context
- Description: {task_description}
- Considerations: {task_considerations}

## Target LLM Context
- Target Model: {target_llm}
- For cost-efficient LLMs (e.g., gpt-4-o-mini, gemini-v15-flash, llama-3-1-8b): these models have limited internal chain-of-thought, so the generated prompt should give short, clear and succinct instructions, without internal reasoning.
- For larger LLMs (e.g., gpt-4-o, claude-v35-sonnet, claude-v37-sonnet): The generated prompt should allow for more complex and extensive internal reasoning, and encourage internal verification of any assumptions related to metrics based on the task description. 

NOTE: Your response should contain only the prompt, without any placeholders for the code, formatting instructions, or additional text. The generated prompt should not contain any additional text like placeholders for the code or formatting instructions.
"""
    }
}

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
                 synthesis_llm_type: Optional[LLMType] = None,
                 current_prompt: Optional[str] = None,
                 custom_task_description: Optional[str] = None,
                 custom_meta_prompt: Optional[str] = None,
                 selected_templates: Optional[List[str]] = None,
                 enable_reverse_comparisons: bool = False,
                 progress_callback: Optional[callable] = None):
        self.task_name = task_name
        self.llm_type = llm_type
        self.judge_llm_type = judge_llm_type or LLMType("gpt-4-o")
        self.synthesis_llm_type = synthesis_llm_type or LLMType("gpt-4-o")  # Default to gpt-4-o if not specified
        self.task = OPTIMIZATION_TASKS[task_name]
        self.current_prompt = current_prompt or self.task["default_prompt"]
        self.custom_task_description = custom_task_description
        self.custom_meta_prompt = custom_meta_prompt
        self.selected_templates = selected_templates or ["standard"]  # Default to standard template if none selected
        self.enable_reverse_comparisons = enable_reverse_comparisons
        self.progress_callback = progress_callback
        self.vision_async_client = None
        self.llm_judge = None
        self.rating_system = CodeRatingSystem()
        self.filled_meta_prompts = {}  # Store filled meta-prompts for each template

    async def setup_clients(self):
        """Initialize API clients"""
        if self.progress_callback:
            self.progress_callback({"status": "setup", "message": "Setting up API clients..."})
            
        # Setup Vision client
        vision_settings = VisionSettings.with_env_prefix("vision", _env_file=".env")
        thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
        self.vision_async_client = VisionAsyncClient(vision_settings, thanos_settings)
        self.llm_judge = LLMJudge(self.vision_async_client, self.judge_llm_type, self.task_name)
        
        if self.progress_callback:
            self.progress_callback({"status": "setup_complete"})

    async def generate_optimization_prompts(self, project_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate optimized prompts using selected meta-prompting templates"""
        if self.progress_callback:
            self.progress_callback({"status": "generating_meta_prompt", "message": "Generating meta-prompts..."})
            
        generated_prompts = {}
        
        for template_id in self.selected_templates:
            if template_id not in META_PROMPT_TEMPLATES:
                logger.warning(f"Template {template_id} not found, skipping...")
                continue
                
            template = META_PROMPT_TEMPLATES[template_id]
            
            # Fill the meta-prompt template
            filled_meta_prompt = template["template"].format(
                objective=self.task["objective"],
                task_description=self.custom_task_description or self.task["description"],
                current_prompt=self.current_prompt,
                target_llm=self.llm_type,
                project_name=project_info.get("name", "Unknown"),
                project_description=project_info.get("description", "No description available"),
                project_languages=project_info.get("language", "unknown"),
                task_considerations=OPTIMIZATION_TASKS[self.task_name]["considerations"]
            )
            
            self.filled_meta_prompts[template_id] = filled_meta_prompt
            
            if self.progress_callback:
                self.progress_callback({
                    "status": "meta_prompt_ready",
                    "template_id": template_id,
                    "template_name": template["name"],
                    "filled_meta_prompt": filled_meta_prompt
                })
            
            logger.info(f"Generating optimization prompt using template: {template['name']}")
            if self.progress_callback:
                self.progress_callback({
                    "status": "generating_prompt",
                    "message": f"Generating optimization prompt using {template['name']}..."
                })
                
            request = LLMInferenceRequest(
                model_type=self.synthesis_llm_type,  # Use synthesis LLM instead of optimization LLM
                messages=[LLMConversationMessage(role=LLMRole.USER, content=filled_meta_prompt)]
            )
            
            try:
                response = await self.vision_async_client.ask(request)
                generated_prompt = response.messages[1].content.strip()
                generated_prompts[template_id] = generated_prompt
                
                if self.progress_callback:
                    self.progress_callback({
                        "status": "prompt_ready",
                        "template_id": template_id,
                        "template_name": template["name"],
                        "generated_prompt": generated_prompt
                    })
                    
            except Exception as e:
                logger.error(f"Error generating prompt for template {template_id}: {str(e)}")
                if self.progress_callback:
                    self.progress_callback({
                        "status": "error",
                        "message": f"Error generating prompt for {template['name']}: {str(e)}"
                    })
                generated_prompts[template_id] = self.current_prompt
                
        return generated_prompts

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

    async def evaluate_benchmark(self, benchmark_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a benchmark file or benchmark data dictionary
        
        Args:
            benchmark_input: Either a path to a benchmark file or a benchmark data dictionary
            
        Returns:
            Dict containing evaluation results
        """
        try:
            # Load benchmark data if input is a file path
            if isinstance(benchmark_input, str):
                with open(benchmark_input, 'r') as f:
                    benchmark_data = json.load(f)
            else:
                benchmark_data = benchmark_input
            
            # Setup clients
            await self.setup_clients()
            
            # Generate prompts using project info and selected templates
            project_info = benchmark_data["metadata"]["project_info"]
            generated_prompts = await self.generate_optimization_prompts(project_info)
            
            results = []
            contest_time = round(datetime.now().timestamp())
            
            # Process each code snippet
            total_snippets = len(benchmark_data["code_snippets"])
            successful_snippets = 0
            
            for idx, snippet in enumerate(benchmark_data["code_snippets"], 1):
                snippet_id = snippet.get("id", f"snippet_{idx}")
                logger.info(f"🔄 Starting processing of snippet {idx}/{total_snippets}: {snippet_id}")
                
                if self.progress_callback:
                    self.progress_callback({
                        "status": "processing_snippet",
                        "message": f"Processing snippet {idx}/{total_snippets}: {snippet_id} (Successfully processed: {successful_snippets})",
                        "progress": idx / total_snippets
                    })
                
                # Get snippet language from metadata
                snippet_language = snippet.get("metadata", {}).get("language", project_info.get("language", "unknown"))
                
                # Generate optimized versions using each template
                optimized_versions = {}
                try:
                    logger.info(f"  📝 Generating optimizations for snippet {snippet_id}")
                    # First, generate baseline optimization
                    logger.info(f"    • Generating baseline optimization...")
                    baseline_code = await self.optimize_code(snippet["content"], self.current_prompt)
                    optimized_versions["baseline"] = baseline_code
                    
                    # Then generate optimizations for each template
                    for template_id, prompt in generated_prompts.items():
                        logger.info(f"    • Generating {template_id} optimization...")
                        optimized_code = await self.optimize_code(snippet["content"], prompt)
                        optimized_versions[template_id] = optimized_code
                    
                    logger.info(f"  ✅ Generated {len(optimized_versions)} optimizations for snippet {snippet_id}")
                except Exception as e:
                    logger.error(f"❌ Error optimizing snippet {snippet_id}: {str(e)}")
                    continue
                
                # Skip if all optimizations failed
                if all(code is None for code in optimized_versions.values()):
                    logger.warning(f"⚠️  All optimizations failed for snippet {snippet_id}, skipping...")
                    continue
                
                # Create unique IDs for each code version
                code_ids = {
                    "original": f"{snippet_id}_original",
                    "baseline": f"{snippet_id}_baseline",
                    **{template_id: f"{snippet_id}_{template_id}" for template_id in generated_prompts.keys()}
                }
                
                # Run all possible comparisons
                all_versions = ["original"] + list(generated_prompts.keys()) + ["baseline"]  # Move baseline to end
                comparison_results = []
                
                # Reset rating system for this snippet
                self.rating_system = CodeRatingSystem()
                
                logger.info(f"  ⚔️  Running comparisons for snippet {snippet_id} ({len(all_versions)} versions)")
                comparison_count = 0
                total_comparisons = len(all_versions) * (len(all_versions) - 1) // 2
                if self.enable_reverse_comparisons:
                    total_comparisons *= 2  # Double for reverse comparisons
                
                # Compare each version with every other version
                for i, version1 in enumerate(all_versions):
                    for version2 in all_versions[i+1:]:
                        code1 = snippet["content"] if version1 == "original" else optimized_versions[version1]
                        code2 = snippet["content"] if version2 == "original" else optimized_versions[version2]
                        
                        # Skip if either code is None
                        if code1 is None or code2 is None:
                            continue
                        
                        comparison_count += 1
                        
                        # Forward comparison: version1 vs version2
                        logger.info(f"    [{snippet_id}] Comparison {comparison_count}/{total_comparisons}: {version1} vs {version2}")
                        score_forward = await self.llm_judge.compare_code(code1, code2)
                        logger.info(f"    [{snippet_id}] → Forward result: {score_forward}")
                        
                        if self.enable_reverse_comparisons:
                            comparison_count += 1
                            # Reverse comparison: version2 vs version1
                            logger.info(f"    [{snippet_id}] Comparison {comparison_count}/{total_comparisons}: {version2} vs {version1} (reverse)")
                            score_reverse = await self.llm_judge.compare_code(code2, code1)
                            logger.info(f"    [{snippet_id}] → Reverse result: {score_reverse}")
                            
                            # Average the scores for more robust results
                            # Note: reverse score needs to be inverted (1.0 - score_reverse)
                            avg_score = (score_forward + (1.0 - score_reverse)) / 2.0
                            logger.info(f"    [{snippet_id}] → Averaged score: {avg_score} (forward: {score_forward}, reverse: {score_reverse})")
                            
                            # Store both individual comparisons for transparency
                            # Note: We don't store the averaged result to keep the detailed table clean
                            comparison_results.append({
                                'comparison': f"{version1} vs {version2}",
                                'score': score_forward,
                                'type': 'forward'
                            })
                            comparison_results.append({
                                'comparison': f"{version2} vs {version1}",
                                'score': score_reverse,
                                'type': 'reverse'
                            })
                            
                            # Use averaged score for ELO rating updates
                            final_score = avg_score
                        else:
                            # Use only forward comparison
                            final_score = score_forward
                            comparison_results.append({
                                'comparison': f"{version1} vs {version2}",
                                'score': score_forward,
                                'type': 'single'
                            })
                        
                        # Update ELO ratings based on final score
                        if final_score > 0.6:  # Strong preference for version1
                            self.rating_system.update_ratings([(code_ids[version1], 0, 0), (code_ids[version2], 1, 1)], contest_time)
                        elif final_score < 0.4:  # Strong preference for version2
                            self.rating_system.update_ratings([(code_ids[version2], 0, 0), (code_ids[version1], 1, 1)], contest_time)
                        else:  # Close match, treat as tie
                            self.rating_system.update_ratings([(code_ids[version1], 0, 0), (code_ids[version2], 0, 0)], contest_time)
                        
                        contest_time += 1
                
                logger.info(f"  🏆 Completed all comparisons for snippet {snippet_id}")
                
                # Get final ratings for all versions
                final_ratings = {
                    version: self.rating_system.get_rating(code_ids[version])
                    for version in all_versions
                }
                
                # Log final ratings for this snippet
                logger.info(f"  📊 ELO ratings for snippet {snippet_id}:")
                for version, rating in sorted(final_ratings.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"    • {version}: {rating:.1f}")
                
                # Prepare result for this snippet
                result = {
                    'snippet_id': snippet_id,
                    'ratings': final_ratings,
                    'comparisons': comparison_results,
                    'original_code': snippet["content"],
                    'optimized_versions': optimized_versions
                }
                
                results.append(result)
                successful_snippets += 1
                
                logger.info(f"✅ Successfully completed snippet {snippet_id} ({successful_snippets}/{total_snippets} total)")
                
                # Send progress update with the snippet result
                if self.progress_callback:
                    self.progress_callback({"status": "snippet_complete", "result": result})
            
            # Calculate average ratings across all snippets
            if results:
                avg_ratings = {}
                for version in all_versions:
                    ratings = [r['ratings'][version] for r in results]
                    avg_ratings[version] = sum(ratings) / len(ratings)
            else:
                avg_ratings = {version: 1500.0 for version in all_versions}
            
            # Prepare final results with complete configuration
            final_results = {
                'benchmark_info': benchmark_data['metadata'],
                'prompts': {
                    'baseline': self.current_prompt,
                    **generated_prompts
                },
                'meta_prompts': {
                    template_id: {
                        'name': META_PROMPT_TEMPLATES[template_id]['name'],
                        'description': META_PROMPT_TEMPLATES[template_id]['description'],
                        'filled_template': self.filled_meta_prompts[template_id]
                    }
                    for template_id in self.selected_templates
                },
                'task_name': self.task_name,
                'task_description': self.custom_task_description or OPTIMIZATION_TASKS[self.task_name]["description"],
                'task_objective': OPTIMIZATION_TASKS[self.task_name]["objective"],
                'task_considerations': OPTIMIZATION_TASKS[self.task_name]["considerations"],
                'llm_type': str(self.llm_type),
                'judge_llm_type': str(self.judge_llm_type),
                'synthesis_llm_type': str(self.synthesis_llm_type),  # Add synthesis LLM type
                'selected_templates': self.selected_templates,
                'enable_reverse_comparisons': self.enable_reverse_comparisons,
                'results': results,
                'average_ratings': avg_ratings,
                'statistics': {
                    'total_snippets': total_snippets,
                    'successful_snippets': successful_snippets,
                    'failed_snippets': total_snippets - successful_snippets
                }
            }
            
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
    
    # Create a copy of results to modify
    results_to_save = results.copy()
    
    # Add configuration information to results
    results_to_save["configuration"] = {
        "task_name": results.get("task_name"),
        "task_description": results.get("task_description"),
        "task_objective": results.get("task_objective"),
        "task_considerations": results.get("task_considerations"),
        "llm_type": results.get("llm_type"),
        "judge_llm_type": results.get("judge_llm_type"),
        "baseline_prompt": results.get("prompts", {}).get("baseline", ""),
        "generated_prompts": results.get("prompts", {}),  # Save all prompts
        "meta_prompts": results.get("meta_prompts", {}),  # Save all meta-prompts
        "selected_templates": results.get("selected_templates", []),  # Save which templates were used
        "enable_reverse_comparisons": results.get("enable_reverse_comparisons", False),  # Save comparison setting
        "evaluation_timestamp": timestamp,
        "statistics": results.get("statistics", {
            "total_snippets": 0,
            "successful_snippets": 0,
            "failed_snippets": 0
        })
    }
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2)
        
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