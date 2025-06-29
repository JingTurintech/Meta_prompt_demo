import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from loguru import logger
import sys
from artemis_client.vision.client import VisionAsyncClient, VisionSettings
from artemis_client.falcon.client import ThanosSettings, FalconSettings, FalconClient, ProjectPromptRequest
from vision_models import LLMInferenceRequest, LLMConversationMessage, LLMRole
from vision_models.service.llm import LLMType
from falcon_models import (
    CodeAIMultiOptimiseRequest, FullSolutionInfoRequest, SolutionSpecResponseBase,
    SolutionResultsRequest, CustomSpecRequest, AIApplicationMethodEnum, SolutionStatusEnum
)
from dotenv import load_dotenv
from dataclasses import dataclass
from uuid import UUID, uuid4
import time
import httpx
import urllib3
import warnings
import requests
from shared_templates import OPTIMIZATION_TASKS, META_PROMPT_TEMPLATES, AVAILABLE_LLMS, JUDGE_PROMPT_TEMPLATE, DEFAULT_PROJECT_OPTIMISATION_IDS

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")



@dataclass
class SolutionResult:
    """Results from executing a solution"""
    solution_id: str
    spec_id: str
    construct_id: str
    status: str
    runtime_metrics: Dict[str, float]
    memory_metrics: Dict[str, float]
    error_log: Optional[str]
    execution_time: float
    success: bool

@dataclass
class RecommendationResult:
    """Results from a code recommendation"""
    spec_id: str
    construct_id: str
    original_code: str
    recommended_code: str
    meta_prompt_used: str
    generated_prompt: str
    recommendation_success: bool
    error_message: Optional[str]
    new_spec_id: Optional[str] = None  # The ID of the new spec created by Artemis

class MetaArtemisEvaluator:
    """Evaluator that combines meta-prompting with Artemis execution"""
    
    def __init__(self,
                 task_name: str,
                 meta_prompt_llm_type: LLMType,  # LLM for meta-prompt generation
                 code_optimization_llm_type: LLMType,  # LLM for code optimization
                 project_id: str,
                 current_prompt: Optional[str] = None,
                 custom_task_description: Optional[str] = None,
                 selected_templates: Optional[List[str]] = None,
                 custom_templates: Optional[Dict[str, str]] = None,  # Custom template contents
                 custom_worker_name: Optional[str] = None,
                 custom_command: Optional[str] = None,
                 evaluation_repetitions: int = 1,  # Number of times to execute each solution for reliable metrics (reduced default to avoid getting stuck)
                 progress_callback: Optional[callable] = None,
                 reuse_existing_recommendations: bool = False,  # New parameter
                 selected_existing_recommendations: Optional[List[str]] = None):  # New parameter
        self.task_name = task_name
        self.meta_prompt_llm_type = meta_prompt_llm_type  # For generating meta-prompts
        self.code_optimization_llm_type = code_optimization_llm_type  # For code optimization via Artemis
        self.project_id = project_id
        self.task = OPTIMIZATION_TASKS[task_name]
        self.current_prompt = current_prompt or self.task["default_prompt"]
        self.custom_task_description = custom_task_description
        self.selected_templates = selected_templates or []  # Changed from ["standard"] to []
        self.custom_templates = custom_templates or {}  # Custom template contents
        self.custom_worker_name = custom_worker_name
        self.custom_command = custom_command
        self.evaluation_repetitions = evaluation_repetitions
        self.progress_callback = progress_callback
        self.reuse_existing_recommendations = reuse_existing_recommendations
        self.selected_existing_recommendations = selected_existing_recommendations or []
        
        # Clients
        self.vision_async_client = None
        self.falcon_client = None
        
        # Generated prompts storage
        self.generated_prompts = {}
        self.meta_prompts = {}
        
        # Existing recommendations storage
        self.existing_recommendations = {}

    async def setup_clients(self):
        """Setup API clients with proper configuration"""
        logger.info("🔧 Starting client setup...")
        
        if self.progress_callback:
            self.progress_callback({"status": "setup", "message": "Setting up API clients..."})
            
        try:
            # Setup Vision client
            vision_settings = VisionSettings.with_env_prefix("vision", _env_file=".env")
            thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
            self.vision_async_client = VisionAsyncClient(vision_settings, thanos_settings)

            # Setup Falcon client - simplified with new version
            falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
            self.falcon_client = FalconClient(falcon_settings, thanos_settings)
            self.falcon_client.authenticate()
            logger.info("✅ API clients authenticated successfully")
            
            logger.info("✅ All clients setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up clients: {str(e)}")
            if self.progress_callback:
                self.progress_callback({
                    "status": "error",
                    "message": f"Failed to setup API clients: {str(e)}"
                })
            raise

    async def get_project_info(self) -> Dict[str, Any]:
        """Get project information from Falcon"""
        try:
            logger.info(f"🔍 Getting project info for project ID: {self.project_id}")
            
            # Get basic project info
            project_response = self.falcon_client.get_project(self.project_id)
            logger.info(f"✅ Successfully got project response for: {getattr(project_response, 'name', 'Unknown')}")
            
            # Get construct details for language detection
            construct_details = self.falcon_client.get_constructs_info(self.project_id)
            logger.info(f"✅ Found {len(construct_details) if construct_details else 0} constructs")
            
            # Initialize with safe defaults
            detected_language = "unknown"
            project_files = []
            
            if construct_details:
                for i, (construct_id, construct) in enumerate(construct_details.items()):
                    # Check for language
                    construct_language = getattr(construct, 'language', 'NOT_FOUND')
                    
                    # Check for file
                    construct_file = getattr(construct, 'file', 'NOT_FOUND')
                    
                    if i == 0:  # Use first construct for language detection
                        detected_language = construct_language if construct_language != 'NOT_FOUND' else 'unknown'
                    
                    if construct_file != 'NOT_FOUND' and construct_file:
                        project_files.append(construct_file)
                
                logger.info(f"🎯 Detected language: {detected_language}")
            else:
                logger.warning("⚠️ No construct details found")
            
            # Build final result
            result = {
                "name": getattr(project_response, 'name', 'Unknown Project'),
                "description": getattr(project_response, 'description', None) or "No description available",
                "language": detected_language,
                "runner_name": getattr(project_response, 'runner_name', None),
                "setup_command": getattr(project_response, 'setup_command', None),
                "clean_command": getattr(project_response, 'clean_command', None),
                "compile_command": getattr(project_response, 'compile_command', None),
                "perf_command": getattr(project_response, 'perf_command', None),
                "unit_test_command": getattr(project_response, 'unit_test_command', None),
                "files": project_files
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting project info: {str(e)}")
            logger.exception("Full exception details:")
            return {
                "name": "Unknown Project",
                "description": "No description available",
                "language": "unknown",
                "runner_name": None,
                "setup_command": None,
                "clean_command": None,
                "compile_command": None,
                "perf_command": None,
                "unit_test_command": None,
                "files": []
            }

    async def generate_optimization_prompts(self, project_info: Dict[str, Any]) -> Dict[str, str]:
        """Generate optimized prompts using meta-prompting for each selected template"""
        prompts = {}
        
        for template_id in self.selected_templates:
            if template_id not in META_PROMPT_TEMPLATES:
                logger.warning(f"Template {template_id} not found, skipping")
                continue
                
            template = META_PROMPT_TEMPLATES[template_id]
            
            if self.progress_callback:
                self.progress_callback({
                    "status": "generating_meta_prompt", 
                    "message": f"Generating meta-prompt for {template['name']}..."
                })
            
            # Use custom template content if available, otherwise use default
            template_content = self.custom_templates.get(template_id, template["template"])
            
            # Prepare template parameters
            template_params = {
                "objective": self.task["objective"],
                "task_description": self.custom_task_description or self.task["description"],
                "target_llm": self.code_optimization_llm_type,  # Use code optimization LLM as target
                "project_name": project_info.get("name", "Unknown"),
                "project_description": project_info.get("description", "No description available"),
                "project_languages": project_info.get("language", "unknown")
            }
            
            # Add current_prompt for templates that need it (like simplified)
            if "{current_prompt}" in template_content:
                template_params["current_prompt"] = self.current_prompt
            
            # Add task_considerations for enhanced template
            if "{task_considerations}" in template_content:
                template_params["task_considerations"] = self.task.get("considerations", "General optimization considerations")
            
            # Fill the meta-prompt template
            meta_prompt = template_content.format(**template_params)
            
            # Store meta-prompt
            self.meta_prompts[template_id] = {
                "name": template["name"],
                "filled_template": meta_prompt
            }
            
            if self.progress_callback:
                self.progress_callback({
                    "status": "meta_prompt_ready",
                    "template_id": template_id,
                    "filled_meta_prompt": meta_prompt
                })
            
            if self.progress_callback:
                self.progress_callback({
                    "status": "generating_prompt", 
                    "message": f"Generating optimization prompt using {template['name']}..."
                })
            
            # Generate the optimization prompt using meta-prompt LLM
            request = LLMInferenceRequest(
                model_type=self.meta_prompt_llm_type,
                messages=[LLMConversationMessage(role=LLMRole.USER, content=meta_prompt)]
            )
            
            try:
                response = await self.vision_async_client.ask(request)
                generated_prompt = response.messages[1].content.strip()
                prompts[template_id] = generated_prompt
                
                if self.progress_callback:
                    self.progress_callback({
                        "status": "prompt_ready",
                        "template_id": template_id,
                        "generated_prompt": generated_prompt
                    })
                    
            except Exception as e:
                logger.error(f"Error generating prompt for {template_id}: {str(e)}")
                prompts[template_id] = self.current_prompt
        
        self.generated_prompts = prompts
        return prompts

    async def get_project_specs(self) -> List[Dict[str, Any]]:
        """Get all specs for the project"""
        try:
            # Get all constructs for the project using the constructs_info method
            constructs = self.falcon_client.get_constructs_info(self.project_id)
            
            specs = []
            for construct_id, construct in constructs.items():
                if hasattr(construct, 'custom_specs') and construct.custom_specs:
                    for spec in construct.custom_specs:
                        # Get detailed spec info if needed
                        try:
                            spec_details = self.falcon_client.get_spec(
                                str(spec.id), 
                                sources="sources",
                                construct=True
                            )
                            specs.append({
                                "spec_id": str(spec.id),
                                "construct_id": str(construct_id),
                                "name": spec.name,
                                "content": spec_details.content,
                                "file": construct.file,
                                "lineno": construct.lineno,
                                "end_lineno": construct.end_lineno
                            })
                        except Exception as e:
                            logger.warning(f"Could not get detailed spec info for {spec.id}: {str(e)}")
                            # Fall back to basic spec info
                            specs.append({
                                "spec_id": str(spec.id),
                                "construct_id": str(construct_id),
                                "name": spec.name,
                                "content": spec.content,
                                "file": construct.file,
                                "lineno": construct.lineno,
                                "end_lineno": construct.end_lineno
                            })
            
            return specs
            
        except Exception as e:
            logger.error(f"Error getting project specs: {str(e)}")
            return []

    async def execute_baseline_recommendation_for_spec(self, spec_info: Dict[str, Any]) -> RecommendationResult:
        """Execute baseline recommendation task for a single spec using baseline prompt directly"""
        try:
            baseline_prompt = self.current_prompt
            
            # Get initial list of spec IDs to track new ones
            initial_constructs = self.falcon_client.get_constructs_info(self.project_id)
            initial_spec_ids = set()
            for construct in initial_constructs.values():
                if hasattr(construct, 'custom_specs'):
                    for spec_obj in construct.custom_specs:
                        initial_spec_ids.add(str(spec_obj.id))
            logger.info(f"Initial spec IDs: {initial_spec_ids}")

            # Create a unique prompt name using timestamp
            prompt_name = f"baseline_prompt_{int(time.time())}"
            
            # Add prompt to project
            prompt_request = ProjectPromptRequest(
                name=prompt_name,
                body=baseline_prompt,
                task="code-generation"
            )
            prompt_response = self.falcon_client.add_prompt(prompt_request, self.project_id)
            prompt_id = str(prompt_response.id)
            logger.info(f"Successfully created baseline prompt with ID {prompt_id} for spec {spec_info['spec_id']}")
            
            # Create and execute recommendation request
            recommendation_request = CodeAIMultiOptimiseRequest(
                project_id=UUID(self.project_id),
                prompt_id=UUID(prompt_id),
                spec_ids=[UUID(spec_info["spec_id"])],
                models=[self.code_optimization_llm_type.value],  # Use code optimization LLM
                align=False,
                raw_output=True,
                method=AIApplicationMethodEnum.zero_shot
            )
            
            # Execute recommendation task
            response = self.falcon_client.execute_recommendation_task(request=recommendation_request, create_process=True)
            logger.info(f"Successfully created baseline recommendation task for spec {spec_info['spec_id']}")
            
            # Extract the process ID from the response
            if not isinstance(response, dict) or 'content' not in response or 'item' not in response['content']:
                raise ValueError(f"Invalid response format for spec {spec_info['spec_id']}")
                
            process_id = response['content']['item']['processId']
            if not process_id:
                raise ValueError(f"No process_id in response for spec {spec_info['spec_id']}")
                
            logger.info(f"Waiting for process {process_id} to complete for spec {spec_info['spec_id']}")
            
            # Wait for process completion with timeout
            start_time = time.time()
            timeout = 120  # 2 minute timeout
            last_status = None
            
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Process {process_id} timed out after {timeout} seconds")
                    
                process_status = self.falcon_client.get_process(UUID(process_id))
                current_status = process_status.status
                
                # Only log status changes to avoid spam
                if current_status != last_status:
                    if current_status in ['completed', 'success']:
                        logger.info(f"Process {process_id} completed successfully")
                        # Add delay to allow the spec to be ready
                        await asyncio.sleep(15)
                        
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
                            raise ValueError(f"No new specs found after optimization for spec {spec_info['spec_id']}")
                        
                        # Get the first new spec
                        new_spec_id = list(new_spec_ids)[0]
                        logger.info(f"Using new spec ID {new_spec_id} for optimized code")
                        
                        # Get the optimized spec
                        optimized_spec = self.falcon_client.get_spec(
                            spec_id=str(new_spec_id),
                            sources="sources",
                            construct=False
                        )
                        
                        if optimized_spec and hasattr(optimized_spec, 'content'):
                            recommended_code = optimized_spec.content
                            if recommended_code:
                                logger.info(f"Successfully retrieved optimized code for spec {new_spec_id}")
                                
                                return RecommendationResult(
                                    spec_id=spec_info["spec_id"],
                                    construct_id=spec_info["construct_id"],
                                    original_code=spec_info["content"],
                                    recommended_code=recommended_code,
                                    meta_prompt_used="",  # No meta-prompt used
                                    generated_prompt=baseline_prompt,
                                    recommendation_success=True,
                                    error_message=None,
                                    new_spec_id=new_spec_id
                                )
                        
                        raise ValueError(f"Invalid spec response or missing content for spec {new_spec_id}")
                        
                    elif current_status in ['failed', 'cancelled', 'error']:
                        error_msg = f"Process {process_id} failed with status: {current_status}"
                        if hasattr(process_status, 'error'):
                            error_msg += f". Error: {process_status.error}"
                        raise Exception(error_msg)
                    else:
                        if current_status == 'pending':
                            logger.info(f"Process {process_id} is pending...")
                        elif current_status in ['created', 'running']:
                            progress = getattr(process_status, 'progress', None)
                            if progress is not None:
                                logger.info(f"Process {process_id} is {current_status}. Progress: {progress:.1%}")
                            else:
                                logger.info(f"Process {process_id} is {current_status}")
                        else:
                            logger.warning(f"Process {process_id} has unknown status: {current_status}")
                    
                    last_status = current_status
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
        except Exception as e:
            logger.error(f"Error executing baseline recommendation for spec {spec_info['spec_id']}: {str(e)}")
            return RecommendationResult(
                spec_id=spec_info["spec_id"],
                construct_id=spec_info["construct_id"],
                original_code=spec_info["content"],
                recommended_code="",
                meta_prompt_used="",
                generated_prompt=self.current_prompt,
                recommendation_success=False,
                error_message=str(e)
            )

    async def execute_recommendation_for_spec(self, spec_info: Dict[str, Any], template_id: str) -> RecommendationResult:
        """Execute recommendation task for a single spec using generated prompt"""
        try:
            if template_id not in self.generated_prompts:
                raise ValueError(f"No generated prompt found for template {template_id}")
            
            generated_prompt = self.generated_prompts[template_id]
            
            # Get initial list of spec IDs to track new ones
            initial_constructs = self.falcon_client.get_constructs_info(self.project_id)
            initial_spec_ids = set()
            for construct in initial_constructs.values():
                if hasattr(construct, 'custom_specs'):
                    for spec_obj in construct.custom_specs:
                        initial_spec_ids.add(str(spec_obj.id))
            logger.info(f"Initial spec IDs: {initial_spec_ids}")

            # Create a unique prompt name using timestamp and template
            prompt_name = f"meta_prompt_{template_id}_{int(time.time())}"
            
            # Add prompt to project using the same method as the working script
            prompt_request = ProjectPromptRequest(
                name=prompt_name,
                body=generated_prompt,
                task="code-generation"
            )
            prompt_response = self.falcon_client.add_prompt(prompt_request, self.project_id)
            prompt_id = str(prompt_response.id)
            logger.info(f"Successfully created prompt with ID {prompt_id} for spec {spec_info['spec_id']}")
            
            # Create and execute recommendation request
            recommendation_request = CodeAIMultiOptimiseRequest(
                project_id=UUID(self.project_id),
                prompt_id=UUID(prompt_id),
                spec_ids=[UUID(spec_info["spec_id"])],
                models=[self.code_optimization_llm_type.value],  # Use code optimization LLM
                align=False,
                raw_output=True,
                method=AIApplicationMethodEnum.zero_shot
            )
            
            # Execute recommendation task
            response = self.falcon_client.execute_recommendation_task(request=recommendation_request, create_process=True)
            logger.info(f"Successfully created recommendation task for spec {spec_info['spec_id']}")
            
            # Extract the process ID from the response
            if not isinstance(response, dict) or 'content' not in response or 'item' not in response['content']:
                raise ValueError(f"Invalid response format for spec {spec_info['spec_id']}")
                
            process_id = response['content']['item']['processId']
            if not process_id:
                raise ValueError(f"No process_id in response for spec {spec_info['spec_id']}")
                
            logger.info(f"Waiting for process {process_id} to complete for spec {spec_info['spec_id']}")
            
            # Wait for process completion with timeout
            start_time = time.time()
            timeout = 120  # 2 minute timeout
            last_status = None
            
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Process {process_id} timed out after {timeout} seconds")
                    
                process_status = self.falcon_client.get_process(UUID(process_id))
                current_status = process_status.status
                
                # Only log status changes to avoid spam
                if current_status != last_status:
                    if current_status in ['completed', 'success']:
                        logger.info(f"Process {process_id} completed successfully")
                        # Add delay to allow the spec to be ready
                        await asyncio.sleep(15)
                        
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
                            raise ValueError(f"No new specs found after optimization for spec {spec_info['spec_id']}")
                        
                        # Get the first new spec
                        new_spec_id = list(new_spec_ids)[0]
                        logger.info(f"Using new spec ID {new_spec_id} for optimized code")
                        
                        # Get the optimized spec
                        optimized_spec = self.falcon_client.get_spec(
                            spec_id=str(new_spec_id),
                            sources="sources",
                            construct=False
                        )
                        
                        if optimized_spec and hasattr(optimized_spec, 'content'):
                            recommended_code = optimized_spec.content
                            if recommended_code:
                                logger.info(f"Successfully retrieved optimized code for spec {new_spec_id}")
                                
                                return RecommendationResult(
                                    spec_id=spec_info["spec_id"],
                                    construct_id=spec_info["construct_id"],
                                    original_code=spec_info["content"],
                                    recommended_code=recommended_code,
                                    meta_prompt_used=self.meta_prompts[template_id]["filled_template"],
                                    generated_prompt=generated_prompt,
                                    recommendation_success=True,
                                    error_message=None,
                                    new_spec_id=new_spec_id
                                )
                        
                        raise ValueError(f"Invalid spec response or missing content for spec {new_spec_id}")
                        
                    elif current_status in ['failed', 'cancelled', 'error']:
                        error_msg = f"Process {process_id} failed with status: {current_status}"
                        if hasattr(process_status, 'error'):
                            error_msg += f". Error: {process_status.error}"
                        raise Exception(error_msg)
                    else:
                        if current_status == 'pending':
                            logger.info(f"Process {process_id} is pending...")
                        elif current_status in ['created', 'running']:
                            progress = getattr(process_status, 'progress', None)
                            if progress is not None:
                                logger.info(f"Process {process_id} is {current_status}. Progress: {progress:.1%}")
                            else:
                                logger.info(f"Process {process_id} is {current_status}")
                        else:
                            logger.warning(f"Process {process_id} has unknown status: {current_status}")
                    
                    last_status = current_status
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            
        except Exception as e:
            logger.error(f"Error executing recommendation for spec {spec_info['spec_id']}: {str(e)}")
            return RecommendationResult(
                spec_id=spec_info["spec_id"],
                construct_id=spec_info["construct_id"],
                original_code=spec_info["content"],
                recommended_code="",
                meta_prompt_used=self.meta_prompts.get(template_id, {}).get("filled_template", ""),
                generated_prompt=self.generated_prompts.get(template_id, ""),
                recommendation_success=False,
                error_message=str(e)
            )

    async def _generate_recommendation_direct(self, spec_info: Dict[str, Any], template_id: str) -> RecommendationResult:
        """Generate recommendation directly using our own LLM call as fallback"""
        try:
            generated_prompt = self.generated_prompts[template_id]
            
            # Format the prompt with the code
            full_prompt = f"{generated_prompt}\n\nOriginal Code:\n{spec_info['content']}\n\nOptimized Code:"
            
            # Generate the optimization using meta-prompt LLM
            request = LLMInferenceRequest(
                model_type=self.meta_prompt_llm_type,
                messages=[LLMConversationMessage(role=LLMRole.USER, content=full_prompt)]
            )
            
            response = await self.vision_async_client.ask(request)
            recommended_code = response.messages[1].content.strip()
            
            return RecommendationResult(
                spec_id=spec_info["spec_id"],
                construct_id=spec_info["construct_id"],
                original_code=spec_info["content"],
                recommended_code=recommended_code,
                meta_prompt_used=self.meta_prompts[template_id]["filled_template"],
                generated_prompt=generated_prompt,
                recommendation_success=True,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Error in direct recommendation generation: {str(e)}")
            return RecommendationResult(
                spec_id=spec_info["spec_id"],
                construct_id=spec_info["construct_id"],
                original_code=spec_info["content"],
                recommended_code="",
                meta_prompt_used=self.meta_prompts.get(template_id, {}).get("filled_template", ""),
                generated_prompt=self.generated_prompts.get(template_id, ""),
                recommendation_success=False,
                error_message=str(e)
            )

    async def _wait_for_process_completion(self, process_id: UUID, timeout: int = 300):
        """Wait for a process to complete"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                process_status = self.falcon_client.get_process(process_id)
                if process_status.status in ["completed", "failed", "cancelled"]:
                    return process_status
                await asyncio.sleep(5)  # Wait 5 seconds before checking again
            except Exception as e:
                logger.error(f"Error checking process status: {str(e)}")
                await asyncio.sleep(5)
        
        raise TimeoutError(f"Process {process_id} did not complete within {timeout} seconds")

    async def _get_or_create_optimisation(self) -> Optional[str]:
        """Get existing optimization - returns None if not found (allows proceeding without optimization)"""
        try:
            # Use default optimization IDs from shared templates
            default_optimisation_id = DEFAULT_PROJECT_OPTIMISATION_IDS.get(self.project_id)
            
            if not default_optimisation_id:
                # Fallback to the original default
                default_optimisation_id = "49b08c56-620f-4ae8-96d3-1675e6a17b2a"
            
            # Verify the optimization exists
            try:
                optimization_info = self.falcon_client.get_optimisation(default_optimisation_id)
                logger.info(f"✅ Using optimization: {getattr(optimization_info, 'name', 'Unknown')} for project {self.project_id}")
                return default_optimisation_id
            except Exception as verify_error:
                logger.warning(f"⚠️ Could not verify optimization {default_optimisation_id}: {verify_error}")
                logger.info("🔄 Proceeding without optimization ID - solution creation may still work")
                return None
                
        except Exception as e:
            logger.warning(f"Could not get optimization: {str(e)}")
            logger.info("🔄 Proceeding without optimization ID - solution creation may still work")
            return None

    async def _execute_solution_with_retry(self, solution_id: str, worker_name: str, max_retries: int = 3) -> bool:
        """Execute solution with retry logic for worker availability issues"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to execute solution {solution_id}, attempt {attempt + 1}/{max_retries}")
                
                # Log solution details before evaluation
                try:
                    pre_eval_solution = self.falcon_client.get_solution(solution_id)
                    logger.info(f"Solution {solution_id} status before evaluation: {pre_eval_solution.status}")
                    if hasattr(pre_eval_solution, 'process_id') and pre_eval_solution.process_id:
                        logger.info(f"Solution has process_id: {pre_eval_solution.process_id}")
                except Exception as e:
                    logger.warning(f"Could not get solution details before evaluation: {e}")
                
                # Try to execute the solution
                evaluation_response = self.falcon_client.evaluate_solution(
                    solution_id=UUID(solution_id),
                    custom_worker_name=worker_name,
                    custom_command=self.custom_command,
                    unit_test=True
                )
                
                logger.info(f"Evaluation response: {evaluation_response}")
                
                # Add a small delay to allow the evaluation to be queued
                await asyncio.sleep(5)
                
                # Check if the solution status changed from 'created'
                post_eval_solution = self.falcon_client.get_solution(solution_id)
                post_eval_status = str(post_eval_solution.status).lower()
                
                if post_eval_status != 'solutionstatusenum.created':
                    logger.info(f"Solution {solution_id} successfully queued for execution, status: {post_eval_status}")
                    return True
                else:
                    logger.warning(f"Solution {solution_id} still in 'created' status after evaluation call")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in 10 seconds...")
                        await asyncio.sleep(10)
                    
            except Exception as e:
                logger.error(f"Error executing solution {solution_id} on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in 10 seconds...")
                    await asyncio.sleep(10)
        
        logger.error(f"Failed to execute solution {solution_id} after {max_retries} attempts")
        return False

    async def _wait_for_solution_completion(self, solution_id: str, timeout: int = 600):
        """Wait for solution evaluation to complete"""
        start_time = time.time()
        last_status = None
        stuck_count = 0
        
        while time.time() - start_time < timeout:
            try:
                solution_details = self.falcon_client.get_solution(solution_id)
                status_str = str(solution_details.status).lower()
                
                # Only log status changes to avoid spam
                if status_str != last_status:
                    logger.info(f"Solution {solution_id} status changed: {last_status} -> {status_str}")
                    last_status = status_str
                    stuck_count = 0
                else:
                    stuck_count += 1
                
                # Check for completion
                if status_str in ['completed', 'failed', 'cancelled']:
                    logger.info(f"Solution {solution_id} finished with status: {status_str}")
                    return solution_details
                
                # Check if solution is stuck in created status for too long
                if status_str == 'solutionstatusenum.created' and stuck_count > 12:  # 2 minutes
                    logger.warning(f"Solution {solution_id} stuck in 'created' status for {stuck_count * 10} seconds")
                    logger.warning("This might indicate:")
                    logger.warning("1. Worker 'jing_runner' is not available or not running")
                    logger.warning("2. Worker is busy with other tasks")
                    logger.warning("3. Solution configuration issue")
                    
                    # Try to get more information about the solution
                    try:
                        if hasattr(solution_details, 'process_id') and solution_details.process_id:
                            process_status = self.falcon_client.get_process(solution_details.process_id)
                            logger.info(f"Associated process status: {process_status.status}")
                    except Exception as process_error:
                        logger.warning(f"Could not get process status: {process_error}")
                
                # If stuck for too long, consider it failed
                if stuck_count > 18:  # 3 minutes in created status (reduced from 5 minutes)
                    logger.error(f"Solution {solution_id} stuck in '{status_str}' for too long, treating as failed")
                    logger.error("This indicates the worker 'jing_runner' is likely not available or not responding")
                    return solution_details
                    
                await asyncio.sleep(10)  # Wait 10 seconds before checking again
            except Exception as e:
                logger.error(f"Error checking solution status: {str(e)}")
                await asyncio.sleep(10)
        
        raise TimeoutError(f"Solution {solution_id} did not complete within {timeout} seconds")

    async def create_and_execute_solution_from_spec(self, spec_info: Dict[str, Any], recommendation_result: RecommendationResult) -> SolutionResult:
        """Create a solution from the RecommendationResult and execute it using the new evaluation_repetitions parameter"""
        try:
            if not recommendation_result.new_spec_id:
                raise ValueError("No new spec ID available from recommendation")
            
            # Create solution using the new spec created by Artemis
            solution_request = FullSolutionInfoRequest(
                specs=[SolutionSpecResponseBase(spec_id=UUID(recommendation_result.new_spec_id))],
                status=SolutionStatusEnum.created
            )
            
            # Get optimization run (optional)
            optimisation_id = await self._get_or_create_optimisation()
            
            # Add solution (optimization ID can be None)
            solution_response = self.falcon_client.add_solution(
                project_id=self.project_id,
                optimisation_id=optimisation_id,
                solution=solution_request
            )
            
            # Extract solution ID (same pattern as working app)
            if isinstance(solution_response, dict):
                solution_id = solution_response.get('solution_id') or solution_response.get('id') or solution_response.get('solutionId')
            else:
                solution_id = str(solution_response)
            
            if not solution_id:
                logger.warning(f"⚠️ Could not extract solution_id from response!")
                solution_id = "UNKNOWN"
            
            logger.info(f"Created solution {solution_id} for spec {recommendation_result.new_spec_id}")
            
            # Execute the solution with multiple repetitions for reliable metrics
            logger.info(f"Executing solution {solution_id} with {self.evaluation_repetitions} repetitions")
            
            execution_start = time.time()
            
            # Use same worker name as working app and proper defaults
            worker_name = self.custom_worker_name or "jing_runner"
            logger.info(f"✅ Using worker: {worker_name}")
            
            evaluation_response = self.falcon_client.evaluate_solution(
                solution_id=UUID(solution_id),
                evaluation_repetitions=self.evaluation_repetitions,  # Use the new parameter
                custom_worker_name=worker_name,
                custom_command=self.custom_command,
                unit_test=True
            )
            
            # Wait for evaluation to complete
            await self._wait_for_solution_completion(solution_id)
            total_execution_time = time.time() - execution_start
            
            # Get solution results (now contains metrics from all repetitions)
            solution_details = self.falcon_client.get_solution(solution_id)
            
            runtime_metrics = {}
            memory_metrics = {}
            
            if solution_details.results and hasattr(solution_details.results, 'values'):
                for metric_name, values in solution_details.results.values.items():
                    if 'runtime' in metric_name.lower() or 'time' in metric_name.lower():
                        # The API now returns aggregated results from multiple repetitions
                        runtime_metrics[f"{metric_name}_avg"] = np.mean(values) if values else 0.0
                        runtime_metrics[f"{metric_name}_std"] = np.std(values) if len(values) > 1 else 0.0
                        runtime_metrics[f"{metric_name}_min"] = np.min(values) if values else 0.0
                        runtime_metrics[f"{metric_name}_max"] = np.max(values) if values else 0.0
                    elif 'memory' in metric_name.lower() or 'mem' in metric_name.lower():
                        # The API now returns aggregated results from multiple repetitions
                        memory_metrics[f"{metric_name}_avg"] = np.mean(values) if values else 0.0
                        memory_metrics[f"{metric_name}_std"] = np.std(values) if len(values) > 1 else 0.0
                        memory_metrics[f"{metric_name}_min"] = np.min(values) if values else 0.0
                        memory_metrics[f"{metric_name}_max"] = np.max(values) if values else 0.0
            
            # Get final solution status
            final_solution_details = self.falcon_client.get_solution(solution_id)
            status_str = str(final_solution_details.status).lower()
            success = status_str == 'completed'
            error_log = None
            
            return SolutionResult(
                solution_id=solution_id,
                spec_id=recommendation_result.new_spec_id,
                construct_id=spec_info["construct_id"],
                status=str(final_solution_details.status),
                runtime_metrics=runtime_metrics,
                memory_metrics=memory_metrics,
                error_log=error_log,
                execution_time=total_execution_time,
                success=success
            )
            
        except Exception as e:
            logger.error(f"Error creating/executing solution for spec {spec_info['spec_id']}: {str(e)}")
            return SolutionResult(
                solution_id="",
                spec_id=spec_info["spec_id"],
                construct_id=spec_info["construct_id"],
                status="failed",
                runtime_metrics={},
                memory_metrics={},
                error_log=str(e),
                execution_time=0.0,
                success=False
            )

    async def create_and_execute_solution(self, spec_info: Dict[str, Any], recommended_code: str) -> SolutionResult:
        """Create a solution with the recommended code and execute it multiple times for reliable metrics"""
        try:
            # First, we need to create a new spec with the recommended code
            # This involves creating a new custom spec with the optimized code
            
            # Create new spec with recommended code
            new_spec_request = CustomSpecRequest(
                name=f"optimized_{spec_info['name']}",
                content=recommended_code,
                imports=[],  # Would need to extract imports from recommended code
                enabled=True
            )
            
            # Add the new spec to the construct
            # Note: This is a simplified approach - in practice, you might want to
            # create a new version or handle this differently
            
            # For now, let's create a solution with the original spec
            # and then update it with the new code
            solution_request = FullSolutionInfoRequest(
                specs=[SolutionSpecResponseBase(spec_id=UUID(spec_info["spec_id"]))],
                status=SolutionStatusEnum.created
            )
            
            # Get optimization run (optional)
            optimisation_id = await self._get_or_create_optimisation()
            
            # Add solution (optimization ID can be None)
            solution_response = self.falcon_client.add_solution(
                project_id=self.project_id,
                optimisation_id=optimisation_id,
                solution=solution_request
            )
            
            # Extract solution ID (same pattern as working app)
            if isinstance(solution_response, dict):
                solution_id = solution_response.get('solution_id') or solution_response.get('id') or solution_response.get('solutionId')
            else:
                solution_id = str(solution_response)
            
            if not solution_id:
                logger.warning(f"⚠️ Could not extract solution_id from response!")
                solution_id = "UNKNOWN"
            
            # TODO: Add the recommended code as a file to the solution
            # This would require creating a file with the recommended code
            # and associating it with the solution
            
            # Execute the solution with multiple repetitions for reliable metrics
            logger.info(f"Executing solution {solution_id} with {self.evaluation_repetitions} repetitions")
            
            execution_start = time.time()
            
            # Use same worker name as working app and proper defaults
            worker_name = self.custom_worker_name or "jing_runner"
            logger.info(f"✅ Using worker: {worker_name}")
            
            evaluation_response = self.falcon_client.evaluate_solution(
                solution_id=UUID(solution_id),
                evaluation_repetitions=self.evaluation_repetitions,  # Use the new parameter
                custom_worker_name=worker_name,
                custom_command=self.custom_command,
                unit_test=True
            )
            
            # Wait for evaluation to complete
            await self._wait_for_solution_completion(solution_id)
            total_execution_time = time.time() - execution_start
            
            # Get solution results (now contains metrics from all repetitions)
            solution_details = self.falcon_client.get_solution(solution_id)
            
            runtime_metrics = {}
            memory_metrics = {}
            
            if solution_details.results and hasattr(solution_details.results, 'values'):
                for metric_name, values in solution_details.results.values.items():
                    if 'runtime' in metric_name.lower() or 'time' in metric_name.lower():
                        # The API now returns aggregated results from multiple repetitions
                        runtime_metrics[f"{metric_name}_avg"] = np.mean(values) if values else 0.0
                        runtime_metrics[f"{metric_name}_std"] = np.std(values) if len(values) > 1 else 0.0
                        runtime_metrics[f"{metric_name}_min"] = np.min(values) if values else 0.0
                        runtime_metrics[f"{metric_name}_max"] = np.max(values) if values else 0.0
                    elif 'memory' in metric_name.lower() or 'mem' in metric_name.lower():
                        # The API now returns aggregated results from multiple repetitions
                        memory_metrics[f"{metric_name}_avg"] = np.mean(values) if values else 0.0
                        memory_metrics[f"{metric_name}_std"] = np.std(values) if len(values) > 1 else 0.0
                        memory_metrics[f"{metric_name}_min"] = np.min(values) if values else 0.0
                        memory_metrics[f"{metric_name}_max"] = np.max(values) if values else 0.0
            
            # Get final solution status
            final_solution_details = self.falcon_client.get_solution(solution_id)
            status_str = str(final_solution_details.status).lower()
            success = status_str == 'completed'
            error_log = None
            
            return SolutionResult(
                solution_id=solution_id,
                spec_id=spec_info["spec_id"],
                construct_id=spec_info["construct_id"],
                status=str(final_solution_details.status),
                runtime_metrics=runtime_metrics,
                memory_metrics=memory_metrics,
                error_log=error_log,
                execution_time=total_execution_time,
                success=success
            )
            
        except Exception as e:
            logger.error(f"Error creating/executing solution for spec {spec_info['spec_id']}: {str(e)}")
            return SolutionResult(
                solution_id="",
                spec_id=spec_info["spec_id"],
                construct_id=spec_info["construct_id"],
                status="failed",
                runtime_metrics={},
                memory_metrics={},
                error_log=str(e),
                execution_time=0.0,
                success=False
            )

    async def create_baseline_prompt(self, project_info: Dict[str, Any]) -> str:
        """Create a baseline prompt for comparison"""
        try:
            # Create a unique prompt name for baseline
            prompt_name = f"baseline_prompt_{self.task_name}_{int(time.time())}"
            
            # Add baseline prompt to project
            prompt_request = ProjectPromptRequest(
                name=prompt_name,
                body=self.current_prompt,
                task="code-generation"
            )
            prompt_response = self.falcon_client.add_prompt(prompt_request, self.project_id)
            prompt_id = str(prompt_response.id)
            logger.info(f"Successfully created baseline prompt with ID {prompt_id}")
            
            return prompt_id
            
        except Exception as e:
            logger.error(f"Error creating baseline prompt: {str(e)}")
            return None

    async def evaluate_project_recommendations(self) -> Dict[str, Any]:
        """Main evaluation method that processes all specs with recommendations"""
        try:
            # Setup clients
            await self.setup_clients()
            
            # Get project information
            if self.progress_callback:
                self.progress_callback({"status": "getting_project_info", "message": "Getting project information..."})
            
            project_info = await self.get_project_info()
            
            # Check if we should reuse existing recommendations
            if self.reuse_existing_recommendations:
                if self.progress_callback:
                    self.progress_callback({"status": "getting_existing_recommendations", "message": "Getting existing recommendations..."})
                
                self.existing_recommendations = await self.get_existing_recommendations()
                
                # If we have selected specific recommendations, use only those
                if self.selected_existing_recommendations:
                    # Filter to only selected recommendations
                    filtered_recommendations = []
                    all_recommendations = (
                        self.existing_recommendations["meta_recommendations"] +
                        self.existing_recommendations["baseline_recommendations"] +
                        self.existing_recommendations["other_recommendations"]
                    )
                    
                    for rec_id in self.selected_existing_recommendations:
                        for rec in all_recommendations:
                            if rec["ai_run_id"] == rec_id or rec["spec_id"] == rec_id:
                                filtered_recommendations.append(rec)
                                break
                    
                    if not filtered_recommendations:
                        raise ValueError("No matching existing recommendations found for the selected IDs")
                    
                    # Process existing recommendations
                    return await self._evaluate_existing_recommendations(filtered_recommendations, project_info)
                else:
                    # Use all existing recommendations
                    all_recommendations = (
                        self.existing_recommendations["meta_recommendations"] +
                        self.existing_recommendations["baseline_recommendations"] +
                        self.existing_recommendations["other_recommendations"]
                    )
                    
                    if not all_recommendations:
                        raise ValueError("No existing recommendations found for this project")
                    
                    return await self._evaluate_existing_recommendations(all_recommendations, project_info)
            
            # Generate optimization prompts (original behavior)
            if self.progress_callback:
                self.progress_callback({"status": "generating_prompts", "message": "Generating optimization prompts..."})
            
            await self.generate_optimization_prompts(project_info)
            
            # Get all project specs
            if self.progress_callback:
                self.progress_callback({"status": "getting_specs", "message": "Getting project specifications..."})
            
            specs = await self.get_project_specs()
            
            if not specs:
                raise ValueError("No specifications found for project")
            
            # Process each spec with each template
            results = {
                "project_info": project_info,
                "meta_prompts": self.meta_prompts,
                "generated_prompts": self.generated_prompts,
                "spec_results": [],
                "summary": {
                    "total_specs": len(specs),
                    "successful_recommendations": 0,
                    "successful_executions": 0,
                    "failed_recommendations": 0,
                    "failed_executions": 0
                }
            }
            
            total_combinations = len(specs) * len(self.selected_templates)
            current_combination = 0
            
            for spec_info in specs:
                spec_results = {
                    "spec_info": spec_info,
                    "template_results": {}
                }
                
                for template_id in self.selected_templates:
                    current_combination += 1
                    
                    if self.progress_callback:
                        progress = current_combination / total_combinations
                        self.progress_callback({
                            "status": "processing_spec",
                            "message": f"Processing spec {spec_info['name']} with {META_PROMPT_TEMPLATES[template_id]['name']} ({current_combination}/{total_combinations})",
                            "progress": progress
                        })
                    
                    # Execute recommendation
                    recommendation_result = await self.execute_recommendation_for_spec(spec_info, template_id)
                    
                    if recommendation_result.recommendation_success:
                        results["summary"]["successful_recommendations"] += 1
                        
                        # Create and execute solution using the new spec ID from the recommendation
                        # The recommendation process already created a new spec, so we use that
                        solution_result = await self.create_and_execute_solution_from_spec(
                            spec_info, 
                            recommendation_result
                        )
                        
                        if solution_result.success:
                            results["summary"]["successful_executions"] += 1
                        else:
                            results["summary"]["failed_executions"] += 1
                    else:
                        results["summary"]["failed_recommendations"] += 1
                        solution_result = None
                    
                    spec_results["template_results"][template_id] = {
                        "recommendation": recommendation_result,
                        "solution": solution_result
                    }
                
                results["spec_results"].append(spec_results)
            
            # Calculate performance analysis
            results["performance_analysis"] = self._analyze_performance_impact(results)
            
            if self.progress_callback:
                self.progress_callback({
                    "status": "complete",
                    "final_results": results
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating project recommendations: {str(e)}")
            if self.progress_callback:
                self.progress_callback({"status": "error", "message": str(e)})
            return None

    async def _evaluate_existing_recommendations(self, recommendations: List[Dict[str, Any]], project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate existing recommendations by creating and executing solutions"""
        try:
            results = {
                "project_info": project_info,
                "existing_recommendations_used": True,
                "recommendations_info": recommendations,
                "spec_results": [],
                "summary": {
                    "total_recommendations": len(recommendations),
                    "successful_executions": 0,
                    "failed_executions": 0
                }
            }
            
            for i, recommendation_info in enumerate(recommendations):
                if self.progress_callback:
                    progress = (i + 1) / len(recommendations)
                    self.progress_callback({
                        "status": "processing_existing_recommendation",
                        "message": f"Processing existing recommendation {i + 1}/{len(recommendations)}: {recommendation_info['spec_name']}",
                        "progress": progress
                    })
                
                # Reuse the existing recommendation
                recommendation_result = await self.reuse_existing_recommendation(recommendation_info)
                
                if recommendation_result.recommendation_success:
                    # Create and execute solution
                    solution_result = await self.create_and_execute_solution_from_spec(
                        {
                            "spec_id": recommendation_result.spec_id,
                            "construct_id": recommendation_result.construct_id,
                            "name": recommendation_info["spec_name"],
                            "content": recommendation_result.original_code,
                            "file": recommendation_info["construct_file"],
                            "lineno": recommendation_info["construct_lines"].split("-")[0],
                            "end_lineno": recommendation_info["construct_lines"].split("-")[1]
                        },
                        recommendation_result
                    )
                    
                    if solution_result.success:
                        results["summary"]["successful_executions"] += 1
                    else:
                        results["summary"]["failed_executions"] += 1
                else:
                    solution_result = None
                    results["summary"]["failed_executions"] += 1
                
                spec_results = {
                    "recommendation_info": recommendation_info,
                    "recommendation_result": recommendation_result,
                    "solution_result": solution_result
                }
                
                results["spec_results"].append(spec_results)
            
            if self.progress_callback:
                self.progress_callback({
                    "status": "complete",
                    "final_results": results
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating existing recommendations: {str(e)}")
            if self.progress_callback:
                self.progress_callback({"status": "error", "message": str(e)})
            return None

    def _analyze_performance_impact(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the performance impact of different recommendations"""
        analysis = {
            "template_comparison": {},
            "best_improvements": [],
            "worst_regressions": [],
            "overall_statistics": {}
        }
        
        # Collect all successful executions by template
        template_metrics = {}
        
        for spec_result in results["spec_results"]:
            for template_id, template_result in spec_result["template_results"].items():
                if template_result["solution"] and template_result["solution"].success:
                    if template_id not in template_metrics:
                        template_metrics[template_id] = {
                            "runtime_improvements": [],
                            "memory_improvements": [],
                            "execution_times": []
                        }
                    
                    solution = template_result["solution"]
                    
                    # Add metrics (would need baseline comparison)
                    if solution.runtime_metrics:
                        for metric_name, value in solution.runtime_metrics.items():
                            template_metrics[template_id]["runtime_improvements"].append(value)
                    
                    if solution.memory_metrics:
                        for metric_name, value in solution.memory_metrics.items():
                            template_metrics[template_id]["memory_improvements"].append(value)
                    
                    template_metrics[template_id]["execution_times"].append(solution.execution_time)
        
        # Calculate statistics for each template
        for template_id, metrics in template_metrics.items():
            template_name = META_PROMPT_TEMPLATES[template_id]["name"]
            analysis["template_comparison"][template_name] = {
                "avg_runtime": np.mean(metrics["runtime_improvements"]) if metrics["runtime_improvements"] else 0,
                "avg_memory": np.mean(metrics["memory_improvements"]) if metrics["memory_improvements"] else 0,
                "avg_execution_time": np.mean(metrics["execution_times"]) if metrics["execution_times"] else 0,
                "successful_executions": len(metrics["execution_times"])
            }
        
        # Overall statistics
        total_successful = sum(len(metrics["execution_times"]) for metrics in template_metrics.values())
        analysis["overall_statistics"] = {
            "total_successful_executions": total_successful,
            "templates_used": len(self.selected_templates),
            "specs_processed": results["summary"]["total_specs"]
        }
        
        return analysis

    async def get_existing_prompts(self) -> Dict[str, Any]:
        """Get existing prompts for the project and categorize them"""
        try:
            # Unfortunately, there's no direct method to list all prompts for a project
            # We'll need to track prompts by their naming convention
            # For now, we'll return an empty dict and rely on the naming convention
            # when creating new prompts
            
            # TODO: If Artemis adds a method to list project prompts, implement it here
            logger.info("Note: Cannot retrieve existing prompts - no API method available")
            return {
                "meta_prompts": [],
                "baseline_prompts": [],
                "other_prompts": []
            }
        except Exception as e:
            logger.error(f"Error getting existing prompts: {str(e)}")
            return {
                "meta_prompts": [],
                "baseline_prompts": [],
                "other_prompts": []
            }

    async def get_existing_recommendations(self) -> Dict[str, Any]:
        """Get existing AI application runs (recommendations) for the project"""
        try:
            logger.info(f"🔍 Getting existing recommendations for project: {self.project_id}")
            
            # Get all constructs to find AI application runs
            logger.info("📋 Getting constructs info...")
            constructs = self.falcon_client.get_constructs_info(self.project_id)
            logger.info(f"✅ Found {len(constructs) if constructs else 0} constructs")
            
            recommendations = {
                "meta_recommendations": [],
                "baseline_recommendations": [],
                "other_recommendations": []
            }
            
            if not constructs:
                logger.warning("⚠️ No constructs found for project")
                return recommendations
            
            # Look through all specs to find ones with ai_run_id
            for construct_id, construct in constructs.items():
                if hasattr(construct, 'custom_specs') and construct.custom_specs:
                    for spec in construct.custom_specs:
                        if hasattr(spec, 'ai_run_id') and spec.ai_run_id:
                            # Get the AI application run details
                            try:
                                ai_run = self.falcon_client.get_ai_application(spec.ai_run_id)
                                
                                # Try to get the prompt to determine if it's meta-prompting
                                prompt_info = None
                                prompt_type = "other"
                                
                                if hasattr(ai_run, 'prompt_id') and ai_run.prompt_id:
                                    try:
                                        prompt_response = self.falcon_client.get_prompt(str(ai_run.prompt_id))
                                        prompt_name = getattr(prompt_response, 'name', 'Unknown')
                                        prompt_body = getattr(prompt_response, 'body', '')
                                        
                                        # Categorize based on prompt name
                                        if prompt_name.startswith("meta_prompt_"):
                                            prompt_type = "meta"
                                        elif prompt_name.startswith("baseline_prompt_"):
                                            prompt_type = "baseline"
                                        
                                        prompt_info = {
                                            "id": str(ai_run.prompt_id),
                                            "name": prompt_name,
                                            "body": prompt_body,
                                            "type": prompt_type
                                        }
                                    except Exception as prompt_error:
                                        logger.warning(f"Could not get prompt details for {ai_run.prompt_id}: {prompt_error}")
                                
                                recommendation_info = {
                                    "ai_run_id": str(spec.ai_run_id),
                                    "spec_id": str(spec.id),
                                    "construct_id": str(construct_id),
                                    "spec_name": getattr(spec, 'name', 'Unknown'),
                                    "construct_file": getattr(construct, 'file', 'Unknown'),
                                    "construct_lines": f"{getattr(construct, 'lineno', 0)}-{getattr(construct, 'end_lineno', 0)}",
                                    "status": str(getattr(ai_run, 'status', 'unknown')),
                                    "models": getattr(ai_run, 'models', []),
                                    "method": str(getattr(ai_run, 'method', 'unknown')),
                                    "created_at": str(getattr(ai_run, 'created_at', 'unknown')),
                                    "prompt_info": prompt_info
                                }
                                
                                # Categorize recommendation
                                if prompt_type == "meta":
                                    recommendations["meta_recommendations"].append(recommendation_info)
                                elif prompt_type == "baseline":
                                    recommendations["baseline_recommendations"].append(recommendation_info)
                                else:
                                    recommendations["other_recommendations"].append(recommendation_info)
                                    
                            except Exception as ai_run_error:
                                logger.warning(f"Could not get AI run details for {spec.ai_run_id}: {ai_run_error}")
                        else:
                            logger.debug(f"Spec {spec.id} has no ai_run_id")
                else:
                    logger.debug(f"Construct {construct_id} has no custom_specs")
            
            # Sort by creation date (newest first)
            for category in recommendations.values():
                if isinstance(category, list):
                    category.sort(key=lambda x: x["created_at"], reverse=True)
            
            logger.info(f"✅ Found {len(recommendations['meta_recommendations'])} meta-prompting recommendations")
            logger.info(f"✅ Found {len(recommendations['baseline_recommendations'])} baseline recommendations")
            logger.info(f"✅ Found {len(recommendations['other_recommendations'])} other recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting existing recommendations: {str(e)}")
            logger.exception("Full exception details:")
            return {
                "meta_recommendations": [],
                "baseline_recommendations": [],
                "other_recommendations": []
            }

    async def reuse_existing_recommendation(self, recommendation_info: Dict[str, Any]) -> RecommendationResult:
        """Reuse an existing recommendation instead of creating a new one"""
        try:
            spec_id = recommendation_info["spec_id"]
            construct_id = recommendation_info["construct_id"]
            
            # Get the original spec content
            original_spec = self.falcon_client.get_spec(
                spec_id=spec_id,
                sources="sources",
                construct=True
            )
            
            # Get the optimized spec content (the recommendation result)
            optimized_spec = self.falcon_client.get_spec(
                spec_id=spec_id,
                sources="sources",
                construct=False
            )
            
            # Determine which template was used based on prompt info
            template_id = "standard"  # default
            if recommendation_info.get("prompt_info"):
                prompt_name = recommendation_info["prompt_info"]["name"]
                if "simplified" in prompt_name:
                    template_id = "simplified"
            
            return RecommendationResult(
                spec_id=spec_id,
                construct_id=construct_id,
                original_code=original_spec.concrete.content if original_spec.concrete else "",
                recommended_code=optimized_spec.content,
                meta_prompt_used=recommendation_info.get("prompt_info", {}).get("body", ""),
                generated_prompt=recommendation_info.get("prompt_info", {}).get("body", ""),
                recommendation_success=True,
                error_message=None,
                new_spec_id=spec_id  # The spec is already the optimized version
            )
            
        except Exception as e:
            logger.error(f"Error reusing existing recommendation: {str(e)}")
            return RecommendationResult(
                spec_id=recommendation_info.get("spec_id", ""),
                construct_id=recommendation_info.get("construct_id", ""),
                original_code="",
                recommended_code="",
                meta_prompt_used="",
                generated_prompt="",
                recommendation_success=False,
                error_message=str(e),
                new_spec_id=None
            )

def save_evaluation_results(results: Dict[str, Any], output_dir: str = "results") -> str:
    """Save evaluation results to a JSON file"""
    logger.info(f"💾 Starting to save evaluation results to directory: {output_dir}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 Output directory created/verified: {output_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = results["project_info"]["name"].replace(" ", "_")
        filename = f"meta_artemis_{project_name}_{timestamp}_evaluation.json"
        filepath = os.path.join(output_dir, filename)
        
        logger.info(f"📄 Generated filename: {filename}")
        logger.info(f"📍 Full filepath: {filepath}")
        
        # Log results structure
        logger.info(f"📊 Results structure:")
        logger.info(f"   - Type: {type(results)}")
        if isinstance(results, dict):
            logger.info(f"   - Keys: {list(results.keys())}")
            logger.info(f"   - Project info: {bool(results.get('project_info'))}")
            logger.info(f"   - Summary: {results.get('summary', {})}")
        
        # Custom JSON encoder
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (RecommendationResult, SolutionResult)):
                    logger.debug(f"🔄 Encoding {type(obj).__name__} object")
                    return obj.__dict__
                return super().default(obj)
        
        logger.info("💾 Writing results to file...")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, cls=CustomEncoder)
        
        # Verify file was written
        file_size = os.path.getsize(filepath)
        logger.info(f"✅ Results saved successfully!")
        logger.info(f"   - File: {filepath}")
        logger.info(f"   - Size: {file_size} bytes")
        
        return filepath
        
    except Exception as e:
        logger.error(f"❌ Error saving evaluation results: {str(e)}")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error args: {e.args}")
        raise

def load_evaluation_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from a JSON file"""
    logger.info(f"📖 Starting to load evaluation results from: {filepath}")
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"❌ File does not exist: {filepath}")
            return None
        
        # Check file size
        file_size = os.path.getsize(filepath)
        logger.info(f"📊 File size: {file_size} bytes")
        
        # Load the file
        logger.info("📄 Opening file for reading...")
        with open(filepath, 'r') as f:
            logger.info("🔄 Parsing JSON content...")
            results = json.load(f)
        
        # Log loaded results structure
        logger.info(f"✅ JSON loaded successfully!")
        logger.info(f"📊 Loaded results structure:")
        logger.info(f"   - Type: {type(results)}")
        if isinstance(results, dict):
            logger.info(f"   - Keys: {list(results.keys())}")
            logger.info(f"   - Project info: {bool(results.get('project_info'))}")
            logger.info(f"   - Summary: {results.get('summary', {})}")
        
        logger.info(f"✅ Successfully loaded results from: {filepath}")
        return results
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON decode error loading {filepath}: {str(e)}")
        logger.error(f"❌ Error at line {e.lineno}, column {e.colno}")
        return None
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {filepath}: {str(e)}")
        return None
    except PermissionError as e:
        logger.error(f"❌ Permission error loading {filepath}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error loading results from {filepath}: {str(e)}")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error args: {e.args}")
        return None

async def main():
    """Example usage"""
    evaluator = MetaArtemisEvaluator(
        task_name="runtime_performance",
        meta_prompt_llm_type=LLMType("gpt-4-o"),
        code_optimization_llm_type=LLMType("gpt-4-o"),
        project_id="your-project-id-here",
    )
    
    results = await evaluator.evaluate_project_recommendations()
    
    if results:
        save_evaluation_results(results)

if __name__ == "__main__":
    asyncio.run(main()) 