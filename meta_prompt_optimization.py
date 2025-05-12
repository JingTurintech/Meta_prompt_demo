import asyncio
import json
import httpx
import numpy as np
import re
import time
import csv
import sys
import pymc3 as pm
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
        "objective": "Improve execution speed and reduce computational complexity",
        "default_prompt": "Analyze the following code and suggest optimizations to improve its runtime performance. Focus on algorithmic improvements, data structure choices, and computational efficiency.",
        "instruction": "Generate an optimized version of the code that improves runtime performance while maintaining the same functionality.",
        "data_format": "\n\nOriginal Code:\n{}\n"
    },
    "memory_usage": {
        "description": "Optimize code for reduced memory consumption",
        "objective": "Minimize memory usage and improve memory management",
        "default_prompt": "Analyze the following code and suggest optimizations to reduce memory usage. Focus on efficient data structures, memory leaks, and resource management.",
        "instruction": "Generate an optimized version of the code that reduces memory consumption while maintaining the same functionality.",
        "data_format": "\n\nOriginal Code:\n{}\n"
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

Your response should contain only the improved prompt, without any explanations or additional text.
"""

class EloRatingSystem:
    def __init__(self, k_factor: float = 32.0):
        self.k_factor = k_factor
        
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using the ELO formula"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, rating_a: float, rating_b: float, score: float) -> Tuple[float, float]:
        """Update ratings based on the game outcome"""
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        new_rating_a = rating_a + self.k_factor * (score - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - score) - expected_b)
        
        return new_rating_a, new_rating_b

class BayesianEloSystem:
    def __init__(self, n_iterations: int = 2000):
        self.n_iterations = n_iterations
        
    def compute_ratings(self, outcomes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute Bayesian ELO ratings for all players"""
        # Extract unique players
        players = set()
        for outcome in outcomes:
            players.add(outcome['player_a'])
            players.add(outcome['player_b'])
        player_list = list(players)
        n_players = len(player_list)
        
        # Create player index mapping
        player_idx = {player: idx for idx, player in enumerate(player_list)}
        
        with pm.Model() as model:
            # Prior for player skills
            skills = pm.Normal('skills', mu=1500, sd=100, shape=n_players)
            
            # Likelihood
            for outcome in outcomes:
                idx_a = player_idx[outcome['player_a']]
                idx_b = player_idx[outcome['player_b']]
                
                # Expected score
                diff = (skills[idx_a] - skills[idx_b]) / 400
                expected = pm.math.sigmoid(diff * pm.math.log(10))
                
                # Observed outcome
                pm.Bernoulli('game_%d' % outcome['game_id'], 
                           p=expected, 
                           observed=outcome['score'])
            
            # Inference
            trace = pm.sample(self.n_iterations, return_inferencedata=False)
        
        # Compute mean ratings
        skills_mean = trace['skills'].mean(axis=0)
        return {player: skills_mean[idx] for player, idx in player_idx.items()}

class MetaPromptOptimizer:
    def __init__(self, 
                 project_id: str,
                 task_name: str,
                 llm_type: LLMType,
                 current_prompt: Optional[str] = None,
                 custom_task_description: Optional[str] = None):
        self.project_id = project_id
        self.task_name = task_name
        self.llm_type = llm_type
        self.task = OPTIMIZATION_TASKS[task_name]
        self.current_prompt = current_prompt or self.task["default_prompt"]
        self.custom_task_description = custom_task_description
        self.vision_async_client = None
        self.falcon_client = None
        
    async def setup_clients(self):
        """Initialize API clients"""
        # Setup Vision client
        vision_settings = VisionSettings.with_env_prefix("vision", _env_file=".env")
        thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
        self.vision_async_client = VisionAsyncClient(vision_settings, thanos_settings)

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
        formatted_prompt = prompt + self.task["instruction"] + \
                         self.task["data_format"].format(spec['content'])
        
        request = LLMInferenceRequest(
            model_type=self.llm_type,
            messages=[LLMConversationMessage(role=LLMRole.USER, content=formatted_prompt)]
        )
        
        try:
            response = await self.vision_async_client.ask(request)
            optimized_code = response.messages[1].content.strip()
            logger.info(f"Generated optimized code for spec {spec['id']}")
            return optimized_code
        except Exception as e:
            logger.error(f"Error optimizing code for spec {spec['id']}: {e}")
            return None
            
    def execute_optimization_task(self, spec_id: str, optimized_code: str) -> Dict[str, Any]:
        """Execute optimization task using Falcon client"""
        try:
            request = {
                "project_id": UUID(self.project_id),
                "spec_ids": [UUID(spec_id)],
                "models": [self.llm_type.value],
                "align": False,
                "raw_output": False,
                "method": "zero_shot",
                "optimized_code": optimized_code
            }
            result = self.falcon_client.execute_recommendation_task(request)
            logger.info(f"Executed optimization task for spec {spec_id}")
            return result
        except Exception as e:
            logger.error(f"Error executing optimization task: {e}")
            return {"error": str(e)}
            
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
        elo_system = EloRatingSystem()
        
        # Process each spec
        for spec in specs:
            logger.info(f"Processing spec {spec['id']}")
            
            # Generate optimized code using both prompts
            baseline_code = await self.optimize_code(spec, self.current_prompt)
            generated_code = await self.optimize_code(spec, generated_prompt)
            
            if baseline_code and generated_code:
                # Execute optimization tasks
                baseline_result = self.execute_optimization_task(spec['id'], baseline_code)
                generated_result = self.execute_optimization_task(spec['id'], generated_code)
                
                # Prepare for ELO rating
                code_versions = {
                    'original': {'code': spec['content'], 'rating': 1500},
                    'baseline': {'code': baseline_code, 'rating': 1500},
                    'generated': {'code': generated_code, 'rating': 1500}
                }
                
                # Compare pairs using execution results
                comparisons = [
                    ('original', 'baseline'),
                    ('original', 'generated'),
                    ('baseline', 'generated')
                ]
                
                for code_a, code_b in comparisons:
                    # Compare performance (you'll need to implement this based on your metrics)
                    score = self.compare_performance(
                        code_versions[code_a]['code'],
                        code_versions[code_b]['code'],
                        baseline_result,
                        generated_result
                    )
                    
                    # Update ELO ratings
                    new_rating_a, new_rating_b = elo_system.update_ratings(
                        code_versions[code_a]['rating'],
                        code_versions[code_b]['rating'],
                        score
                    )
                    
                    code_versions[code_a]['rating'] = new_rating_a
                    code_versions[code_b]['rating'] = new_rating_b
                
                results.append({
                    'spec_id': spec['id'],
                    'original_rating': code_versions['original']['rating'],
                    'baseline_rating': code_versions['baseline']['rating'],
                    'generated_rating': code_versions['generated']['rating'],
                    'baseline_result': baseline_result,
                    'generated_result': generated_result
                })
        
        # Calculate Bayesian ELO ratings
        bayesian_elo = BayesianEloSystem()
        game_outcomes = []
        for idx, result in enumerate(results):
            # Add outcomes for each comparison
            game_outcomes.extend([
                {
                    'game_id': idx * 3,
                    'player_a': 'original',
                    'player_b': 'baseline',
                    'score': self.get_comparison_score(result['baseline_result'])
                },
                {
                    'game_id': idx * 3 + 1,
                    'player_a': 'original',
                    'player_b': 'generated',
                    'score': self.get_comparison_score(result['generated_result'])
                },
                {
                    'game_id': idx * 3 + 2,
                    'player_a': 'baseline',
                    'player_b': 'generated',
                    'score': self.compare_results(
                        result['baseline_result'],
                        result['generated_result']
                    )
                }
            ])
        
        bayesian_ratings = bayesian_elo.compute_ratings(game_outcomes)
        
        return {
            'prompts': {
                'baseline': self.current_prompt,
                'generated': generated_prompt
            },
            'results': results,
            'bayesian_ratings': bayesian_ratings
        }
    
    def compare_performance(self, 
                          code_a: str, 
                          code_b: str, 
                          result_a: Dict[str, Any], 
                          result_b: Dict[str, Any]) -> float:
        """Compare performance between two code versions"""
        # This is a simplified comparison - you should implement based on your metrics
        if 'error' in result_a or 'error' in result_b:
            return 0.5
        
        # Compare based on execution time, memory usage, etc.
        # Return a score between 0 and 1
        # This is just a placeholder - implement your own comparison logic
        return 0.6
    
    def get_comparison_score(self, result: Dict[str, Any]) -> float:
        """Get comparison score from optimization result"""
        if 'error' in result:
            return 0.5
        # Implement your own scoring logic based on the optimization results
        return 0.6
    
    def compare_results(self, result_a: Dict[str, Any], result_b: Dict[str, Any]) -> float:
        """Compare two optimization results"""
        if 'error' in result_a or 'error' in result_b:
            return 0.5
        # Implement your own comparison logic
        return 0.6

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