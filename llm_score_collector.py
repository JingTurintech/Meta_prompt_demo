#!/usr/bin/env python3
"""Script to collect LLM scores from Artemis platform for runtime correlation analysis."""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from uuid import UUID

from artemis_client.falcon.client import FalconSettings, ThanosSettings, FalconClient
from falcon_models import (
    ProjectInfoResponse,
    SpecScoreInfoResponse,
    CustomSpecResponse,
    CodeSpecScoresResponse,
    CodeTaskScoresResponse,
    CodeTaskModelScoreResponse
)
from meta_artemis_modules.shared_templates import DEFAULT_PROJECT_OPTIMISATION_IDS

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize clients
falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
falcon_client = FalconClient(falcon_settings, thanos_settings)
falcon_client.authenticate()

def get_construct_identifier(construct) -> str:
    """Create a meaningful identifier for a construct using file path and line numbers."""
    file_name = os.path.basename(construct.file)
    return f"{file_name}:{construct.lineno}-{construct.end_lineno}"

def collect_llm_scores():
    """Collect LLM scores for all specs in a project"""
    try:
        # Print available default projects
        print("\nAvailable benchmark projects:")
        projects_info = []
        for project_id in DEFAULT_PROJECT_OPTIMISATION_IDS.keys():
            try:
                project = falcon_client.get_project(project_id)
                projects_info.append((project_id, project.name))
                print(f"{len(projects_info)}. {project.name} ({project_id})")
            except Exception as e:
                logger.warning(f"Could not fetch info for project {project_id}: {str(e)}")
        
        if not projects_info:
            logger.error("No projects found")
            return
        
        # Get user selection
        selection = int(input("\nSelect a project number: ")) - 1
        if selection < 0 or selection >= len(projects_info):
            logger.error("Invalid selection")
            return
        
        project_id, project_name = projects_info[selection]
        
        # Get all constructs for the project
        constructs = falcon_client.get_constructs_info(project_id)
        if not constructs:
            logger.error("No constructs found for project")
            return
            
        # Collect scores for each spec
        scores_data = []
        for construct_id, construct in constructs.items():
            if hasattr(construct, 'custom_specs') and construct.custom_specs:
                construct_identifier = get_construct_identifier(construct)
                for spec in construct.custom_specs:
                    # Get scores for the spec
                    scores_response = falcon_client.get_scores([str(spec.id)])
                    
                    # The response is a dictionary mapping UUID to CodeSpecScoresResponse
                    spec_uuid = UUID(str(spec.id))
                    if spec_uuid in scores_response.root:
                        spec_scores = scores_response.root[spec_uuid]
                        if hasattr(spec_scores, 'scores'):
                            for task_scores in spec_scores.scores:
                                # task_scores is CodeTaskScoresResponse
                                task = task_scores.task
                                for model_score in task_scores.models:
                                    # model_score is CodeTaskModelScoreResponse
                                    scores_data.append({
                                        'project_id': project_id,
                                        'project_name': project_name,
                                        'construct_id': str(construct_id),
                                                'construct_identifier': construct_identifier,
                                        'spec_id': str(spec.id),
                                                'spec_name': spec.name if hasattr(spec, 'name') else f"spec-{spec.id[:8]}",
                                                'task': task,
                                                'model': model_score.name,
                                                'score': model_score.score,
                                                'prompt_id': str(model_score.prompt_id) if model_score.prompt_id else None,
                                                'metric_name': 'llm_score',
                                                'metric_measurements': str(model_score.score),
                                                'metric_count': 1
                                    })
        
        if not scores_data:
            logger.error("No scores found")
            return
            
        # Convert to DataFrame and save
        df = pd.DataFrame(scores_data)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'results/llm_scores_{project_name}_{timestamp}.csv'
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"\nScores saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error collecting scores: {str(e)}")

if __name__ == '__main__':
    collect_llm_scores() 