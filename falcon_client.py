"""
Falcon API client wrapper for accessing and validating optimization data.

This module implements a wrapper pattern around the Falcon API client these reasons:

1. Data Validation and Type Safety:
   - Ensures all data from the external API meets the requirements of the tools that use it
   - Provides clear type hints and validation through Pydantic models

2. Interface Stability:
   - Creates a stable interface for our internal code even if the external API changes
   - Allows us to evolve our internal data models independently of the API
   - Reduces coupling between our codebase and external dependencies

3. Dependency Management:
   - Centralizes our interaction with the external API
   - Makes it easier to switch API clients or modify API interaction patterns
   - Reduces the number of places that need to be updated if the API changes

Required Environment Variables:
    Thanos Authentication:
        - THANOS_USERNAME: Username for authentication
        - THANOS_PASSWORD: Password for authentication
        - THANOS_CLIENT_ID: OAuth client ID
        - THANOS_CLIENT_SECRET: OAuth client secret
        - THANOS_GRANT_TYPE: OAuth grant type (default: "password")
        - THANOS_HOST: Thanos service hostname
        - THANOS_PORT: Thanos service port
        - THANOS_POSTFIX: API endpoint path
        - THANOS_HTTPS: Boolean flag for HTTPS

    Falcon Service:
        - FALCON_HTTPS: Boolean flag for HTTPS
        - FALCON_HOST: Falcon service hostname
        - FALCON_PORT: Falcon service port
        - FALCON_POSTFIX: API endpoint path

    These variables can be provided either through environment variables or via a .env file.
"""

from functools import lru_cache
from typing import Dict, Optional, List, Any, Set, Tuple, Union
from uuid import UUID

from artemis_client.falcon.client import (FalconClient, FalconSettings,
                                          ThanosSettings)
from falcon_models.rest_api.code_models import (
    ConcreteConstructResponse as RawConstructResponse,
    ConcreteConstructResponse as ConstructResponse,
    CustomSpecResponse
)
from falcon_models.rest_api.ai_models import (
    ProjectPromptRequest,
    ProjectPromptResponse,
    PromptRequestResponseBase
)
from falcon_models import (
    CodeAIMultiOptimiseRequest,
    ScoringTaskResponse
)


class FalconClientWrapper:
    """
    A wrapper around the FalconClient that provides specific access to optimization data.

    This class restricts interaction with the Falcon API by only exposing the required data
    and validating the structure of the response.

    The wrapper pattern serves three purposes:
    1. Data Validation: Ensures all data meets our internal requirements
    2. Interface Stability: Provides a stable interface even if the API changes
    3. Dependency Management: Centralizes API interaction

    Example usage:
        client = FalconClientWrapper()
        construct = client.get_construct(construct_id)
        # construct is now a ConstructResponse with validated data
    """

    def __init__(self, env_file_path: Optional[str] = None):
        """
        Initialize the Falcon API client.

        Args:
            env_file_path: Optional path to .env file with credentials.
                           If not provided, environment variables will be used.
                           Example file format available at optimisation-analysis/.env.prod.example
        """
        self.client = self._setup_client(env_file_path)

    def _setup_client(self, env_file_path: Optional[str] = None) -> FalconClient:
        """
        Set up and authenticate the Falcon client.

        Args:
            env_file_path: Optional path to .env file with credentials

        Returns:
            Authenticated FalconClient instance
        """
        falcon_settings = FalconSettings.with_env_prefix(
            "falcon",
            _env_file=env_file_path,
        )
        thanos_settings = ThanosSettings.with_env_prefix(
            "thanos",
            _env_file=env_file_path,
        )
        client = FalconClient(
            falcon_settings=falcon_settings,
            thanos_settings=thanos_settings,
        )
        client.authenticate()
        return client

    @lru_cache(maxsize=128)
    def get_construct(self, construct_id: str) -> ConstructResponse:
        """
        Get information about a construct.

        This method:
        1. Fetches data from the external API
        2. Converts to our internal model
        3. Validates the data structure
        4. Returns a validated response

        The results are cached using LRU (Least Recently Used) caching with a maximum
        of 128 entries. This helps reduce API calls for frequently accessed constructs.

        Args:
            construct_id: The UUID of the construct to get

        Returns:
            Validated construct details

        Raises:
            ValidationError: If the response data doesn't match the expected structure
        """
        raw_response = self.client.get_construct(construct_id=construct_id)
        
        # Convert raw response to our internal model
        construct_data = raw_response.dict() if hasattr(raw_response, 'dict') else raw_response
        return ConstructResponse(**construct_data)

    def get_global_prompts(
        self, task: str = "optimise", page: int = 1, per_page: int = 10
    ) -> List[ProjectPromptResponse]:
        """
        Get a list of available global prompts.

        Args:
            task: Type of prompts to retrieve (default: "optimise")
            page: Page number for pagination (default: 1)
            per_page: Number of prompts per page (default: 10)

        Returns:
            List of available prompts with their details

        Raises:
            ValidationError: If the response data doesn't match the expected structure
        """
        raw_response = self.client.get_global_prompts(
            task=task,
            page=page,
            per_page=per_page
        )
        return raw_response.prompts

    def get_prompt(self, prompt_id: str) -> ProjectPromptResponse:
        """
        Get details of a specific prompt.

        This method:
        1. Fetches data from the external API
        2. Converts to our internal model
        3. Validates the data structure
        4. Returns a validated response

        Args:
            prompt_id: ID of the prompt to retrieve
            
        Returns:
            Dictionary containing the prompt details

        Raises:
            ValidationError: If the response data doesn't match the expected structure
        """
        raw_response = self.client.get_prompt(prompt_id=prompt_id)
        return ProjectPromptResponse.model_validate(raw_response)

    def generate_code_with_prompt(
        self,
        project_id: str,
        construct_id: str,
        prompt_id: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate code improvements using a prompt.

        Args:
            project_id: The UUID of the project
            construct_id: The UUID of the construct to improve
            prompt_id: Optional ID of the prompt to use
            model_type: Optional specific model to use for generation

        Returns:
            Dictionary containing the task results
        """
        # Convert string UUIDs to UUID objects
        project_uuid = UUID(project_id)
        construct_uuid = UUID(construct_id)
        prompt_uuid = UUID(prompt_id) if prompt_id else None

        request = CodeAIMultiOptimiseRequest(
            project_id=project_uuid,
            spec_ids=[construct_uuid],
            models=[model_type] if model_type else ["gpt-4-o-mini"],
            prompt_id=prompt_uuid,
            align=False,
            raw_output=False,
            method="zero_shot"
        )
        return self.client.execute_recommendation_task(request)

    def get_constructs_info(
        self,
        project_id: str,
        construct_ids: Optional[List[str]] = None
    ) -> Dict[UUID, ConstructResponse]:
        """
        Get information about multiple constructs in a project.

        Args:
            project_id: The UUID of the project
            construct_ids: Optional list of construct IDs to filter by

        Returns:
            Dictionary mapping construct IDs to their details
        """
        raw_response = self.client.get_constructs_info(project_id, construct_ids)
        return {
            UUID(str(construct_id)): ConstructResponse(**construct_data.dict())
            for construct_id, construct_data in raw_response.items()
        }

    def get_scores(
        self,
        spec_ids: List[str],
        models: List[str],
        prompt_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieve scores for specific specs, models, and prompts.

        Args:
            spec_ids: List of UUIDs of the specs to get scores for
            models: List of model names to get scores for
            prompt_ids: List of UUIDs of the prompts to get scores for

        Returns:
            Dictionary containing the scores for the specified inputs
        """
        return self.client.get_scores(
            spec_ids=spec_ids,
            models=models,
            prompt_ids=prompt_ids
        )

    def execute_recommendation_task(
        self,
        request: CodeAIMultiOptimiseRequest,
        create_process: bool = True
    ) -> Dict:
        """
        Execute a code recommendation task.

        Args:
            request: The recommendation request parameters
            create_process: Whether to create a process for tracking

        Returns:
            Dictionary containing the task results
        """
        return self.client.execute_recommendation_task(request, create_process=create_process)

    def execute_scoring_task(
        self,
        project_id: str,
        spec_ids: List[str],
        models: List[str],
        prompt_ids: List[str]
    ) -> ScoringTaskResponse:
        """Execute a scoring task for multiple specs using multiple models and prompts.
        
        Args:
            project_id: The UUID of the project
            spec_ids: List of UUIDs of the specs to score
            models: List of model names to use for scoring
            prompt_ids: List of UUIDs of the prompts to use
            
        Returns:
            ScoringTaskResponse containing the details of the created scoring task
        """
        return self.client.execute_scoring_task(
            project_id=project_id,
            spec_ids=spec_ids,
            models=models,
            prompt_ids=prompt_ids
        )

    def get_scores(
        self,
        spec_ids: List[str],
        models: List[str],
        prompt_ids: List[str],
    ) -> Dict[str, Any]:
        """Retrieve scores for specific specs, models, and prompts.

        Args:
            spec_ids: List of UUIDs of the specs to get scores for
            models: List of model names to get scores for
            prompt_ids: List of UUIDs of the prompts to get scores for

        Returns:
            Dictionary containing the scores for the specified specs, models, and prompts
        """
        return self.client.get_scores(spec_ids, models, prompt_ids)

    def get_construct_to_specs(self, project_id: str) -> Dict[str, List[str]]:
        """
        Get a mapping from construct_id to a list of its spec IDs.

        Args:
            project_id: The UUID of the project to analyze

        Returns:
            Dictionary mapping construct_id to list of spec_ids
        """
        # First get all constructs in the project
        constructs = self.get_constructs_info(project_id)
        
        # Build mapping from construct ID to spec IDs
        construct_to_specs: Dict[str, List[str]] = {}
        for construct_id, construct in constructs.items():
            spec_ids = [str(spec.id) for spec in construct.custom_specs]
            construct_to_specs[str(construct_id)] = spec_ids
            
        return construct_to_specs

    def add_prompt(self, prompt_request: ProjectPromptRequest, project_id: str) -> ProjectPromptResponse:
        """
        Add a prompt to a project.

        Args:
            prompt_request: The prompt request containing name, body, and task
            project_id: The UUID of the project to add the prompt to

        Returns:
            ProjectPromptResponse containing the created prompt details

        Raises:
            ValidationError: If the response data doesn't match the expected structure
        """
        raw_response = self.client.add_prompt(prompt_request, project_id)
        return ProjectPromptResponse.model_validate(raw_response)

    def get_spec(
        self,
        spec_id: str,
        sources: str = "sources",
        construct: bool = False
    ) -> CustomSpecResponse:
        """Get information about a spec.

        This method:
        1. Fetches data from the external API
        2. Validates the response using CustomSpecResponse model
        3. Returns the validated spec details including source code if requested

        Args:
            spec_id: The UUID of the spec to get
            sources: Type of sources to include ("none", "root", or "sources")
            construct: Whether to include construct information

        Returns:
            Validated CustomSpecResponse containing spec details

        Raises:
            ValidationError: If the response data doesn't match the expected structure
            ValueError: If the response cannot be properly validated
        """
        try:
            # Get raw response from Falcon API - this will return a JSON dict
            raw_response = self.client.get_spec(
                spec_id=spec_id,
                sources=sources,
                construct=construct
            )

            # If raw_response is already a CustomSpecResponse, return it directly
            if isinstance(raw_response, CustomSpecResponse):
                return raw_response

            # Handle different response types
            if isinstance(raw_response, dict):
                response_data = raw_response
            elif hasattr(raw_response, 'dict'):
                response_data = raw_response.dict()
            elif hasattr(raw_response, '__dict__'):
                # Handle SQLAlchemy model
                response_data = {
                    key: getattr(raw_response, key)
                    for key in dir(raw_response)
                    if not key.startswith('_') and not callable(getattr(raw_response, key))
                }
            else:
                raise ValueError(f"Unexpected response type: {type(raw_response)}")

            # First try normal validation
            try:
                return CustomSpecResponse.model_validate(response_data)
            except Exception as validation_error:
                # If normal validation fails, try with from_attributes=True
                try:
                    return CustomSpecResponse.model_validate(response_data, from_attributes=True)
                except Exception as e:
                    raise ValueError(f"Failed to validate spec response: {str(e)}")

        except Exception as e:
            raise ValueError(f"Failed to get or validate spec {spec_id}: {str(e)}")

    def get_spec_source_code(
        self,
        spec_id: str,
    ) -> Union[str, Dict[str, str]]:
        """Get the source code of a spec by its ID.

        Args:
            spec_id: The UUID of the spec to retrieve

        Returns:
            Either:
                - The source code content string if successful
                - Dictionary with error message if the retrieval failed

        Note:
            Returns the spec's source code if available through the content attribute.
        """
        try:
            # Get the spec with its sources
            spec_response = self.get_spec(
                spec_id=spec_id,
                sources="sources",
                construct=False
            )
            
            return spec_response.content
            
        except Exception as e:
            return {
                "error": str(e)
            }
        

    def get_spec_details(
        self,
        spec_id: str,
    ) -> Dict[str, Any]:
        """Get detailed information about a spec by its ID.

        Args:
            spec_id: The UUID of the spec to retrieve

        Returns:
            Dictionary containing the spec details
        """
        try:
            # Get the spec with its sources
            spec_response = self.get_spec(
                spec_id=spec_id,
                sources="sources",
                construct=True
            )
            
            return spec_response
            
        except Exception as e:
            return {
                "error": str(e)
            }
