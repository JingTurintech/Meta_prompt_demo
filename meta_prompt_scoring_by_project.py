import asyncio
import json
import httpx
import numpy as np
import re
import time
import csv
import sys
from loguru import logger
from artemis_client.vision.client import VisionAsyncClient, VisionSettings
from artemis_client.falcon.client import FalconClient, FalconSettings, ThanosSettings
from vision_models import LLMInferenceRequest, LLMConversationMessage, LLMRole
from vision_models.service.llm import LLMType
from dotenv import load_dotenv
import os
from sklearn.metrics import cohen_kappa_score
from uuid import UUID

# Load environment variables from .env file
load_dotenv()

# Configure logger to use INFO level
logger.remove()
logger.add(sys.stderr, level="INFO")

# Define available LLMs
AVAILABLE_SCORING_LLMS = [
    LLMType("gpt-4-o-mini"),
    LLMType("gemini-v15-flash"),
    LLMType("llama-3-1-8b"),
    LLMType("gpt-4-o"),
    LLMType("claude-v35-sonnet"),
    LLMType("claude-v37-sonnet")
]

AVAILABLE_PROMPT_GENERATION_LLMS = [
    LLMType("gpt-4-o-mini"),
    LLMType("gemini-v15-flash"),
    LLMType("llama-3-1-8b"),
    LLMType("gpt-4-o"),
    LLMType("claude-v35-sonnet"),
    LLMType("claude-v37-sonnet")
]

# Define the default LLMs
default_scoring_llms = [
    LLMType("gemini-v15-flash")
]

# Define the prompt generation LLM
prompt_generation_llm = LLMType("gpt-4-o")

# Define the trusted LLM for cohen's kappa evaluation
TRUSTED_LLM = LLMType("gpt-4-o")

# Add a parameter to choose the evaluation metric
evaluation_metric = "cohen_kappa"  # Options: "accuracy", "cohen_kappa"

# Define the meta-prompt template
meta_prompt_template = """
{META_PROMPT}

## Task Details
- Task: {TASK_DESCRIPTION}
- Target LLM: {TARGET_LLM_NAME}
- Current Prompt: {CURRENT_PROMPT}
"""

# Predefined task descriptions and instructions
predefined_tasks = {
    "Pair-wise code performance scoring": {
        "description": "Evaluate if code B has improved in runtime performance over code A. The code snippets come from real-world projects.",
        "instruction": "\nOutput 1 if code B is improved over code A in terms of the target metric, output 0 otherwise. The response should contain only 0 or 1 without any other text.",
        "default_prompt": "Compare the following two code snippets and determine if the second code has improved in runtime performance.",
        "data_format": "\n\n - Code A:\n{}\n - Code B:\n{}"
    }
}

async def call_llm(client, prompt, model_type):
    request = LLMInferenceRequest(
        model_type=model_type,
        messages=[LLMConversationMessage(role=LLMRole.USER, content=prompt)]
    )
    logger.debug("Request payload: {}".format(request.model_dump(mode='json')))
    try:
        response = await client.ask(request)
        return response
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error occurred: {}".format(e))
        logger.error("Response content: {}".format(e.response.text))
        raise
    except Exception as e:
        logger.error("An error occurred: {}".format(e))
        raise

def extract_binary_score(response_content):
    """
    Extract binary score (0 or 1) from LLM response content, handling cases with additional text.
    Prioritizes checking the first token of the response.
    """
    # Clean and normalize the text
    text = response_content.strip()
    
    # First check if the entire response is just "0" or "1"
    if text in ['0', '1']:
        return int(text)
    
    # Check if response starts with 0 or 1 followed by whitespace or punctuation
    start_match = re.match(r'^([01])[\s\.,:]', text)
    if start_match:
        return int(start_match.group(1))
        
    # If no match at start, fall back to original checks
    text = text.lower()
    
    # Try to find a single digit 0 or 1 using regex
    matches = re.findall(r'\b[01]\b', text)
    if len(matches) == 1:
        return int(matches[0])
        
    # Check for common text patterns
    if any(phrase in text for phrase in ['no improvement', 'not improved', 'worse', 'slower']):
        return 0
    if any(phrase in text for phrase in ['improved', 'better', 'faster', 'more efficient']):
        return 1
    
    # More aggressive number extraction if still no match
    numbers = re.findall(r'[01]', text)
    if len(numbers) == 1:
        return int(numbers[0])
        
    raise ValueError(f"Could not extract valid binary score (0 or 1) from response: '{response_content}'")

def calculate_accuracy(actual_improvements, predicted_improvements):
    correct_predictions = sum(1 for actual, predicted in zip(actual_improvements, predicted_improvements) if actual == predicted)
    total_predictions = len(actual_improvements)
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_cohen_kappa(actual_improvements, predicted_improvements):
    return cohen_kappa_score(actual_improvements, predicted_improvements)

class MetaPromptEvaluator:
    def __init__(self, meta_prompt, meta_prompt_template, task_name, current_prompt, scoring_llms,
                 prompt_generation_llm, evaluation_metric, project_id, custom_task_description=None):
        self.meta_prompt = meta_prompt
        self.meta_prompt_template = meta_prompt_template
        self.task_name = task_name
        self.current_prompt = current_prompt
        self.scoring_llms = scoring_llms
        self.prompt_generation_llm = prompt_generation_llm
        self.evaluation_metric = evaluation_metric
        self.project_id = project_id
        self.custom_task_description = custom_task_description
        self.vision_async_client = None
        self.falcon_client = None
        self.trusted_predictions_cache = {}

    async def setup_clients(self):
        # Setup Vision client
        vision_settings = VisionSettings.with_env_prefix("vision", _env_file=".env")
        thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
        self.vision_async_client = VisionAsyncClient(vision_settings, thanos_settings)

        # Setup Falcon client
        falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
        self.falcon_client = FalconClient(falcon_settings, thanos_settings)
        self.falcon_client.authenticate()

    def get_project_info(self):
        """Get project information including name, languages, etc."""
        project_info = self.falcon_client.get_project(self.project_id)
        logger.info(f"Project info from Falcon: {project_info}")
        return project_info

    def get_spec_details(self, spec_id):
        """Get detailed information about a specific spec"""
        try:
            # First try to get specs in batch which might be more reliable
            specs = self.falcon_client.get_specs([spec_id], with_context=True, sources="sources")
            if specs and UUID(spec_id) in specs:
                return specs[UUID(spec_id)]
        except Exception as e:
            logger.warning(f"Failed to get spec details using get_specs: {str(e)}")
            
        # Fallback to individual spec fetch
        try:
            return self.falcon_client.get_spec(spec_id, sources="sources", construct=True)
        except Exception as e:
            logger.error(f"Failed to get spec details using get_spec: {str(e)}")
            raise

    def get_project_specs(self):
        """Get all specs (original and recommendations) from the project"""
        try:
            constructs = self.falcon_client.get_constructs_info(self.project_id)
            logger.info(f"Found {len(constructs)} constructs")
            
            # Get all specs from constructs
            spec_pairs = []
            for construct_id, construct in constructs.items():
                logger.info(f"Processing construct {construct_id}")
                logger.info(f"Construct details: {construct}")
                # Get original specs
                if hasattr(construct, 'custom_specs'):
                    logger.info(f"Found {len(construct.custom_specs)} custom specs in construct {construct_id}")
                    for spec in construct.custom_specs:
                        try:
                            # Get the recommendation for this spec
                            spec_details = self.get_spec_details(str(spec.id))
                            logger.info(f"Spec details for {spec.id}: {spec_details}")
                            
                            # Find recommendations in the custom_specs that have this spec as a source
                            recommendations = [s for s in construct.custom_specs 
                                            if hasattr(s, 'source_ids') and 
                                            any(src == spec.id for src in s.source_ids)]
                            
                            # Add pairs of original and recommended specs
                            for rec in recommendations:
                                spec_pairs.append({
                                    'code_a': spec_details.content,  # Original code
                                    'code_b': rec.content  # Recommended code
                                })
                                logger.info(f"Added spec pair: original={spec.id}, recommendation={rec.id}")
                        except Exception as e:
                            logger.error(f"Error getting spec details for spec {spec.id}: {str(e)}")
                            continue
                else:
                    logger.info(f"No custom specs found in construct {construct_id}")
            
            logger.info(f"Found {len(spec_pairs)} spec pairs to evaluate")
            return spec_pairs
        except Exception as e:
            logger.error(f"Error getting project specs: {str(e)}")
            return []

    async def generate_refined_prompt(self, target_llm_name):
        # Use custom task description if provided, otherwise use default
        task_description = self.custom_task_description or predefined_tasks[self.task_name]["description"]

        filled_meta_prompt = self.meta_prompt_template.format(
            META_PROMPT=self.meta_prompt,
            TASK_DESCRIPTION=task_description,
            TARGET_LLM_NAME=target_llm_name,
            CURRENT_PROMPT=self.current_prompt
        )
        logger.info("Generating refined prompt for target LLM: {}".format(target_llm_name))
        logger.info("Filled meta prompt: {}".format(filled_meta_prompt))
        response = await call_llm(self.vision_async_client, filled_meta_prompt, self.prompt_generation_llm)
        refined_prompt = response.messages[1].content.strip()
        logger.info("Generated refined prompt: {}".format(refined_prompt))
        return refined_prompt

    async def evaluate_with_llm(self, prompt, model_type, code_a, code_b):
        """Combined pipeline for LLM evaluation"""
        task = predefined_tasks[self.task_name]
        formatted_data = task['data_format'].format(code_a, code_b)
        full_prompt = f"{prompt}{task['instruction']}{formatted_data}"
        request = LLMInferenceRequest(
            model_type=model_type,
            messages=[LLMConversationMessage(role=LLMRole.USER, content=full_prompt)]
        )
        
        try:
            response = await self.vision_async_client.ask(request)
            response_content = response.messages[1].content.strip()
            return extract_binary_score(response_content)
        except (httpx.HTTPStatusError, ValueError) as e:
            logger.error(f"Error during LLM evaluation: {e}")
            return None

    async def evaluate_meta_prompts(self):
        await self.setup_clients()
        total_start_time = time.time()

        # Get project info and specs
        project_info = self.get_project_info()
        logger.info(f"Evaluating project: {project_info.name}")
        
        # Get all specs from the project
        spec_pairs = self.get_project_specs()
        logger.info(f"Found {len(spec_pairs)} spec pairs to evaluate")

        with open('meta_prompt_evaluation_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Scoring LLM', 'Prompt Type', 'Generated Prompt', 'Metric Value', 'Number of Specs', 'Total Time Spent (seconds)'])

            results = []
            
            # Get trusted LLM predictions if using cohen's kappa
            if self.evaluation_metric == "cohen_kappa":
                logger.info(f"Getting predictions from trusted LLM: {TRUSTED_LLM}")
                
                for n_p, pair in enumerate(spec_pairs):
                    code_a = pair['code_a']  # Original code
                    code_b = pair['code_b']  # Recommended code
                    pair_key = (code_a, code_b)
                    
                    if pair_key not in self.trusted_predictions_cache:
                        trusted_label = await self.evaluate_with_llm(
                            predefined_tasks[self.task_name]["default_prompt"], 
                            TRUSTED_LLM, 
                            code_a, 
                            code_b
                        )
                        self.trusted_predictions_cache[pair_key] = trusted_label

            for scoring_llm in self.scoring_llms:
                if self.evaluation_metric == "cohen_kappa" and scoring_llm == TRUSTED_LLM:
                    logger.info(f"Skipping {scoring_llm} as it is the trusted LLM")
                    continue

                logger.info(f"Evaluating scoring LLM: {scoring_llm}")
                refined_prompt = await self.generate_refined_prompt(scoring_llm)

                for prompt_type, prompt in [("Generated", refined_prompt), ("Baseline", self.current_prompt)]:
                    logger.info(f"Evaluating prompt type: {prompt_type}")
                    
                    start_time = time.time()
                    actual_improvements = []
                    predicted_improvements = []
                    code_pairs = []

                    # Evaluate each spec pair
                    for n_p, pair in enumerate(spec_pairs):
                        code_a = pair['code_a']  # Original code
                        code_b = pair['code_b']  # Recommended code
                        code_pairs.append({'code_a': code_a, 'code_b': code_b})
                        
                        predicted_label = await self.evaluate_with_llm(prompt, scoring_llm, code_a, code_b)
                        
                        if predicted_label is not None:
                            if self.evaluation_metric == "cohen_kappa":
                                pair_key = (code_a, code_b)
                                actual_improvements.append(self.trusted_predictions_cache[pair_key])
                            else:
                                # For accuracy metric, we assume the recommendation is an improvement
                                actual_improvements.append(1)
                            predicted_improvements.append(predicted_label)

                    if actual_improvements and predicted_improvements:
                        if self.evaluation_metric == "accuracy":
                            metric_value = calculate_accuracy(actual_improvements, predicted_improvements)
                        elif self.evaluation_metric == "cohen_kappa":
                            metric_value = calculate_cohen_kappa(actual_improvements, predicted_improvements)
                        else:
                            logger.error("Invalid evaluation metric: {}".format(self.evaluation_metric))
                            metric_value = 0.0
                    else:
                        metric_value = 0.0

                    end_time = time.time()
                    time_spent = end_time - start_time

                    writer.writerow([scoring_llm, prompt_type, prompt, metric_value, len(spec_pairs), time_spent])
                    logger.warning("Scoring LLM: {}, Prompt ({}): {}, Metric ({}): {:.3f}, Time Spent: {:.2f} seconds".format(
                        scoring_llm, prompt_type, prompt, self.evaluation_metric, metric_value, time_spent))
                    logger.warning("Results written to {}".format(file.name))

                    results.append({
                        'scoring_llm': scoring_llm,
                        'prompt_type': prompt_type,
                        'generated_prompt': prompt,
                        'metric_value': metric_value,
                        'num_specs': len(spec_pairs),
                        'time_spent': time_spent,
                        'actual_label': actual_improvements,
                        'predicted_label': predicted_improvements,
                        'code_pairs': code_pairs
                    })

        total_end_time = time.time()
        total_time_spent = total_end_time - total_start_time
        logger.warning("Total time spent for the whole process: {:.2f} seconds".format(total_time_spent))

        return results, project_info

async def main():
    task_name = "Pair-wise code performance scoring"
    meta_prompt = """
    You are a prompt generation expert. We are working together to improve a task-specific prompt for a particular LLM. 
    Please follow the task context and instructions below:

    ## Evaluation Questions
    - Is the current prompt clear in describing the task to the LLM?
    - Does the prompt align with the LLM's capabilities and address its limitations?
    - How can the prompt be improved to ensure better outputs for the task?

    ## Instructions
    1. Evaluate the contexts provided. Identify areas for improvement, considering clarity, alignment with the task, 
    and the target LLM's capabilities or limitations.
    2. Propose a refined version of the prompt that better aligns with the task requirements and the LLM's strengths.
    3. Your answer should only contain the refined prompt.
    """
    
    project_id = "c05998b8-d588-4c8d-a4bf-06d163c1c1d8"  # Default project ID
    
    evaluator = MetaPromptEvaluator(
        meta_prompt=meta_prompt,
        meta_prompt_template=meta_prompt_template,
        task_name=task_name,
        current_prompt=predefined_tasks[task_name]["default_prompt"],
        scoring_llms=default_scoring_llms,
        prompt_generation_llm=prompt_generation_llm,
        evaluation_metric=evaluation_metric,
        project_id=project_id
    )
    await evaluator.evaluate_meta_prompts()

if __name__ == "__main__":
    asyncio.run(main())
