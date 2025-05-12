import asyncio
import json
import httpx
import numpy as np
import re
import time
import csv
import sys
import pandas as pd
from loguru import logger
from artemis_client.vision.client import VisionAsyncClient, VisionSettings
from artemis_client.falcon.client import FalconClient, FalconSettings, ThanosSettings
from vision_models import LLMInferenceRequest, LLMConversationMessage, LLMRole
from vision_models.service.llm import LLMType
from dotenv import load_dotenv
import os
from uuid import UUID
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from elommr import EloMMR, Player as EloPlayer
from datetime import datetime, timezone
from artemis_client.falcon.client import ProjectPromptRequest, CodeAIMultiOptimiseRequest
from falcon_models.rest_api.ai_models import AIInferenceTask

# Patch numpy bool deprecation for PyMC3 compatibility
np.bool = np.bool_

# Load environment variables from .env file
load_dotenv()

# Configure logger to use INFO level
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
        "default_prompt": "Analyze the following code and suggest optimizations to improve its runtime performance. Focus on algorithmic improvements, data structure choices, and computational efficiency.",
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
        "default_prompt": "Analyze the following code and suggest optimizations to reduce memory usage. Focus on efficient data structures, memory leaks, and resource management.",
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

## Task Context
{task_description}

## Current Prompt
{current_prompt}

## Target LLM Information
Target LLM: {target_llm}

## Instructions
1. Analyze the current prompt and task context
2. Consider the target LLM's capabilities and limitations
3. Generate an improved prompt that will help the LLM better optimize code for {objective}
4. The prompt should be specific, clear, and focused on {objective}
5. Your response should contain only the improved prompt text, without any placeholders, formatting instructions, or additional text.
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
            contest_time = round(datetime.now(timezone.utc).timestamp())
            
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

class MetaPromptOptimizer:
    def __init__(self, 
                 project_id: str,
                 task_name: str,
                 llm_type: LLMType,
                 judge_llm_type: Optional[LLMType] = None,
                 current_prompt: Optional[str] = None,
                 custom_task_description: Optional[str] = None):
        self.project_id = project_id
        self.task_name = task_name
        self.llm_type = llm_type
        self.judge_llm_type = judge_llm_type or LLMType("gpt-4-o")
        self.task = OPTIMIZATION_TASKS[task_name]
        self.current_prompt = current_prompt or self.task["default_prompt"]
        self.custom_task_description = custom_task_description
        self.vision_async_client = None
        self.falcon_client = None
        self.llm_judge = None
        self.rating_system = CodeRatingSystem()

    async def setup_clients(self):
        """Initialize API clients"""
        # Setup Vision client
        vision_settings = VisionSettings.with_env_prefix("vision", _env_file=".env")
        thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
        self.vision_async_client = VisionAsyncClient(vision_settings, thanos_settings)
        self.llm_judge = LLMJudge(self.vision_async_client, self.judge_llm_type, self.task_name)

        # Setup Falcon client
        falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
        self.falcon_client = FalconClient(falcon_settings, thanos_settings)
        self.falcon_client.authenticate()

    def get_project_specs(self) -> List[Dict[str, Any]]:
        """Get all original specs from the project"""
        logger.info(f"Getting specs for project {self.project_id}")
        constructs = self.falcon_client.get_constructs_info(self.project_id)
        logger.info(f"Found {len(constructs)} constructs")
        
        specs = []
        for construct_id, construct in constructs.items():
            logger.info(f"Processing construct {construct_id}")
            if hasattr(construct, 'custom_specs'):
                for spec in construct.custom_specs:
                    # Only get original specs (those without source_ids)
                    if not hasattr(spec, 'source_ids') or not spec.source_ids:
                        try:
                            spec_details = self.falcon_client.get_spec(
                                str(spec.id), 
                                sources="sources",
                                construct=True
                            )
                            specs.append({
                                'id': str(spec.id),
                                'content': spec_details.content,
                                'construct_id': str(construct_id)
                            })
                            logger.info(f"Added original spec {spec.id}")
                        except Exception as e:
                            logger.error(f"Error getting spec {spec.id}: {e}")
                            continue
        
        return specs
    
    async def generate_optimization_prompt(self) -> str:
        """Generate an optimized prompt using meta-prompting"""
        meta_prompt = META_PROMPT_TEMPLATE.format(
            objective=self.task["objective"],
            task_description=self.custom_task_description or self.task["description"],
            current_prompt=self.current_prompt,
            target_llm=self.llm_type
        )
        
        logger.info("Generating optimization prompt")
        request = LLMInferenceRequest(
            model_type=self.llm_type,
            messages=[LLMConversationMessage(role=LLMRole.USER, content=meta_prompt)]
        )
        
        try:
            response = await self.vision_async_client.ask(request)
            generated_prompt = response.messages[1].content.strip()
            logger.info(f"Generated prompt: {generated_prompt}")
            return generated_prompt
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return self.current_prompt
            
    async def optimize_code(self, spec: Dict[str, Any], prompt: str) -> Optional[str]:
        """Generate optimized code using the given prompt"""
        try:
            # Get initial list of spec IDs
            initial_constructs = self.falcon_client.get_constructs_info(self.project_id)
            initial_spec_ids = set()
            for construct in initial_constructs.values():
                if hasattr(construct, 'custom_specs'):
                    for spec_obj in construct.custom_specs:
                        initial_spec_ids.add(str(spec_obj.id))
            logger.info(f"Initial spec IDs: {initial_spec_ids}")

            # Create a unique prompt name using timestamp
            prompt_name = f"optimization_prompt_{int(time.time())}"
            
            try:
                # Add prompt to project
                prompt_request = ProjectPromptRequest(
                    name=prompt_name,
                    body=prompt,  # Use the prompt directly without formatting
                    task="code-generation"  # Use the correct task type that API accepts
                )
                prompt_response = self.falcon_client.add_prompt(prompt_request, self.project_id)
                prompt_id = str(prompt_response.id)
                logger.info(f"Successfully created prompt with ID {prompt_id} for spec {spec['id']}")
            except Exception as e:
                logger.error(f"Error creating prompt for spec {spec['id']}: {str(e)}")
                logger.error(f"Request details: name={prompt_name}, task=code-generation")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                return None
            
            try:
                # Create and execute recommendation request
                request = CodeAIMultiOptimiseRequest(
                    project_id=UUID(self.project_id),
                    prompt_id=UUID(prompt_id),
                    spec_ids=[UUID(spec['id'])],
                    models=[self.llm_type.value],
                    align=False,
                    raw_output=True,
                    method="zero_shot"
                )
                
                # Execute recommendation task
                response = self.falcon_client.execute_recommendation_task(request=request, create_process=True)
                logger.info(f"Successfully created recommendation task for spec {spec['id']}")
                
                try:
                    # Extract the process ID from the response
                    if not isinstance(response, dict) or 'content' not in response or 'item' not in response['content']:
                        logger.error(f"Invalid response format for spec {spec['id']}. Response: {response}")
                        return None
                        
                    process_id = response['content']['item']['processId']
                    if not process_id:
                        logger.error(f"No process_id in response for spec {spec['id']}. Response: {response}")
                        return None
                        
                    logger.info(f"Waiting for process {process_id} to complete for spec {spec['id']}")
                    
                    # Add timeout handling
                    start_time = time.time()
                    timeout = 60  # 1 minute timeout
                    last_status = None
                    
                    while True:
                        if time.time() - start_time > timeout:
                            logger.error(f"Process {process_id} timed out after {timeout} seconds")
                            return None
                            
                        process_status = self.falcon_client.get_process(UUID(process_id))
                        current_status = process_status.status
                        
                        # Only log status changes to avoid spam
                        if current_status != last_status:
                            if current_status in ['completed', 'success']:  # Handle both completed and success states
                                logger.info(f"Process {process_id} completed successfully")
                                # Add longer delay to allow the spec to be ready
                                await asyncio.sleep(10)  # Increased from 5 to 10 seconds
                                
                                try:
                                    # Get updated list of spec IDs
                                    updated_constructs = self.falcon_client.get_constructs_info(self.project_id)
                                    updated_spec_ids = set()
                                    for construct in updated_constructs.values():
                                        if hasattr(construct, 'custom_specs'):
                                            for spec_obj in construct.custom_specs:
                                                updated_spec_ids.add(str(spec_obj.id))
                                    
                                    # Find new spec IDs
                                    new_spec_ids = updated_spec_ids - initial_spec_ids
                                    logger.info(f"New spec IDs found: {new_spec_ids}")
                                    
                                    if not new_spec_ids:
                                        logger.error(f"No new specs found after optimization for spec {spec['id']}")
                                        return None
                                    
                                    # Get the first new spec (there should typically be only one)
                                    new_spec_id = list(new_spec_ids)[0]
                                    logger.info(f"Using new spec ID {new_spec_id} for optimized code")
                                    
                                    # Get the optimized spec
                                    optimized_spec = self.falcon_client.get_spec(
                                        spec_id=str(new_spec_id),
                                        sources="sources",
                                        construct=False
                                    )
                                    
                                    if optimized_spec and hasattr(optimized_spec, 'content'):
                                        optimized_code = optimized_spec.content
                                        if optimized_code:
                                            # Log which prompt was used for this optimization
                                            if prompt == self.current_prompt:
                                                logger.info(f"Successfully retrieved baseline optimized code for spec {new_spec_id}")
                                            else:
                                                logger.info(f"Successfully retrieved meta-prompt optimized code for spec {new_spec_id}")
                                            return optimized_code
                                        else:
                                            logger.error(f"Empty content in optimized spec {new_spec_id}")
                                    else:
                                        logger.error(f"Invalid spec response format or missing content for spec {new_spec_id}")
                                except Exception as e:
                                    logger.error(f"Error getting optimized spec: {str(e)}")
                                break
                                
                            elif current_status in ['failed', 'cancelled', 'error']:
                                logger.error(f"Process {process_id} failed with status: {current_status}")
                                if hasattr(process_status, 'error'):
                                    logger.error(f"Process error: {process_status.error}")
                                return None
                            elif current_status == 'pending':
                                logger.info(f"Process {process_id} is pending...")
                            elif current_status in ['created', 'running']:  # Handle both created and running states
                                progress = getattr(process_status, 'progress', None)
                                if progress is not None:
                                    logger.info(f"Process {process_id} is {current_status}. Progress: {progress:.1%}")
                                else:
                                    logger.info(f"Process {process_id} is {current_status}")
                            else:
                                logger.warning(f"Process {process_id} has unknown status: {current_status}")
                            
                            last_status = current_status
                        
                        await asyncio.sleep(2)  # Check every 2 seconds
                          
                    return None
                    
                except Exception as e:
                    logger.error(f"Error processing recommendation result for spec {spec['id']}: {str(e)}")
                    logger.error(f"Process details: process_id={response.get('process_id')}")
                    import traceback
                    logger.error(f"Traceback:\n{traceback.format_exc()}")
                    return None
                
            except Exception as e:
                logger.error(f"Error executing recommendation task for spec {spec['id']}: {str(e)}")
                logger.error(f"Request details: project_id={self.project_id}, prompt_id={prompt_id}, spec_id={spec['id']}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                return None
            
        except Exception as e:
            logger.error(f"Unexpected error in optimize_code for spec {spec['id']}: {str(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return None
            

    async def run_optimization_workflow(self) -> Dict[str, Any]:
        """Run the complete optimization workflow"""
        await self.setup_clients()
        
        # Get original specs
        specs = self.get_project_specs()
        if not specs:
            return {"error": "No specs found in project"}
            
        # Generate prompts
        generated_prompt = await self.generate_optimization_prompt()
        
        results = []
        contest_time = round(datetime.now(timezone.utc).timestamp())
        
        # Process each spec
        for spec in specs:
            logger.info(f"Processing spec {spec['id']}")
            
            # Generate optimized code using both prompts
            logger.info("Generating baseline optimization...")
            baseline_code = await self.optimize_code(spec, self.current_prompt)
            logger.info("Generating meta-prompt optimization...")
            generated_code = await self.optimize_code(spec, generated_prompt)
            
            # Create unique IDs for each code version
            spec_id = spec['id']
            original_id = f"{spec_id}_original"
            baseline_id = f"{spec_id}_baseline"
            generated_id = f"{spec_id}_generated"
            
            if baseline_code is None and generated_code is None:
                logger.error(f"Both baseline and meta-prompt optimizations failed for spec {spec_id}")
                # Add result with default ratings and error messages
                results.append({
                    'spec_id': spec_id,
                    'original_rating': 1500.0,  # Default rating
                    'baseline_rating': 1500.0,
                    'generated_rating': 1500.0,
                    'comparisons': [],
                    'original_code': spec['content'],
                    'baseline_code': "Failed to generate baseline optimization",
                    'generated_code': "Failed to generate meta-prompt optimization"
                })
                continue
            
            # Run comparisons in both forward and reverse order
            comparison_orders = [
                # Forward order - Generated prompt first
                [
                    (original_id, generated_id, spec['content'], generated_code),
                    (original_id, baseline_id, spec['content'], baseline_code),
                    (baseline_id, generated_id, baseline_code, generated_code)
                ],
                # Reverse order - Still keeping Generated vs Baseline last
                [
                    (original_id, generated_id, spec['content'], generated_code),
                    (original_id, baseline_id, spec['content'], baseline_code),
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
                    
                    # Format comparison result with simplified names
                    code_a_name = code_a_id.split('_')[-1]  # Get just 'original', 'baseline', or 'generated'
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
            
            # Average the ratings from both orders
            results.append({
                'spec_id': spec_id,
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
                'original_code': spec['content'],
                'baseline_code': baseline_code if baseline_code else "Failed to generate baseline optimization",
                'generated_code': generated_code if generated_code else "Failed to generate meta-prompt optimization"
            })
        
        # Calculate average ratings across all specs
        avg_ratings = {
            'original': np.mean([r['original_rating'] for r in results]),
            'baseline': np.mean([r['baseline_rating'] for r in results]),
            'generated': np.mean([r['generated_rating'] for r in results])
        }
        
        # Calculate average ratings by order
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
        
        return {
            'prompts': {
                'baseline': self.current_prompt,
                'generated': generated_prompt
            },
            'results': results,
            'average_ratings': avg_ratings,
            'average_ratings_by_order': avg_ratings_by_order
        }

async def main():
    # Example usage
    project_id = "c05998b8-d588-4c8d-a4bf-06d163c1c1d8"
    optimizer = MetaPromptOptimizer(
        project_id=project_id,
        task_name="runtime_performance",
        llm_type=LLMType("gpt-4-o")
    )
    
    results = await optimizer.run_optimization_workflow()
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())