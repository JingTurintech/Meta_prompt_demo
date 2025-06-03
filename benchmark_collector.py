import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List
from loguru import logger
import sys
from artemis_client.falcon.client import FalconClient, FalconSettings
from artemis_client.falcon.client import ThanosSettings
from dotenv import load_dotenv
from uuid import UUID

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime and UUID objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

class BenchmarkCollector:
    def __init__(self):
        """Initialize the benchmark collector"""
        self.falcon_client = None
        self.setup_falcon_client()
        
    def setup_falcon_client(self):
        """Setup Falcon client with authentication"""
        try:
            falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
            thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
            self.falcon_client = FalconClient(falcon_settings, thanos_settings)
            self.falcon_client.authenticate()
            logger.info("Successfully authenticated with Falcon API")
        except Exception as e:
            logger.error(f"Error setting up Falcon client: {str(e)}")
            raise

    def get_project_info(self, project_id: str) -> Dict[str, Any]:
        """Get project information and detected language"""
        try:
            project_info = self.falcon_client.get_project(project_id)
            
            # Get construct details for language detection
            construct_details = self.falcon_client.get_constructs_info(project_id)
            
            detected_language = "unknown"
            
            if construct_details:
                # Get the first construct's details to determine language
                first_construct = next(iter(construct_details.values()))
                detected_language = first_construct.language if hasattr(first_construct, 'language') else "unknown"
            
            return {
                "project_id": project_id,
                "name": getattr(project_info, 'name', 'Unknown'),
                "description": getattr(project_info, 'description', 'No description available'),
                "language": detected_language
            }
        except Exception as e:
            logger.error(f"Error getting project info for {project_id}: {str(e)}")
            return None

    def get_project_specs(self, project_id: str) -> List[Dict[str, Any]]:
        """Get the first 10 original specs from constructs that have the 'RANK 1-10' tag"""
        try:
            constructs = self.falcon_client.get_constructs_info(project_id)
            logger.info(f"Found {len(constructs)} constructs in project {project_id}")
            
            specs = []
            for construct_id, construct in constructs.items():
                # Get construct tags
                construct_tags = [tag.name for tag in construct.tags] if hasattr(construct, 'tags') else []
                
                # Only process constructs that have the "RANK 1-10" tag
                if "RANK 1-10" in construct_tags:
                    if hasattr(construct, 'custom_specs'):
                        for spec in construct.custom_specs:
                            # Only get original specs (those without source_ids)
                            if not hasattr(spec, 'source_ids') or not spec.source_ids:
                                try:
                                    # Get detailed spec information
                                    spec_details = self.falcon_client.get_spec(
                                        str(spec.id), 
                                        sources="sources",
                                        construct=True
                                    )
                                    
                                    # Get tag names from TagResponse objects
                                    tag_names = [tag for tag in spec.tags] if hasattr(spec, 'tags') else []
                                    
                                    # Get any scores if available
                                    scores = {}
                                    if hasattr(spec, 'scores'):
                                        scores = spec.scores
                                    
                                    # Collect all available information
                                    spec_info = {
                                        'id': str(spec.id),
                                        'content': spec_details.content,
                                        'construct_id': str(construct_id),
                                        'tags': tag_names,
                                        'scores': scores,
                                        'metadata': {
                                            'created_at': getattr(spec, 'created_at', None),
                                            'updated_at': getattr(spec, 'updated_at', None),
                                            'language': getattr(construct, 'language', None),
                                            'file': getattr(construct, 'file', None),
                                            'name': getattr(spec, 'name', None),
                                            'imports': getattr(spec, 'imports', []),
                                            'file_operation': getattr(spec, 'file_operation', None),
                                            'enabled': getattr(spec, 'enabled', True),
                                            'line_numbers': {
                                                'start': getattr(construct, 'lineno', None),
                                                'end': getattr(construct, 'end_lineno', None)
                                            },
                                            'tokens': getattr(construct, 'tokens', None),
                                            'construct_tags': construct_tags
                                        }
                                    }
                                    
                                    specs.append(spec_info)
                                    logger.info(f"Added original spec {spec.id} from construct with tags: {construct_tags}")
                                    
                                    # Stop after collecting 10 specs
                                    if len(specs) >= 10:
                                        logger.info("Collected 10 specs, stopping collection")
                                        return specs
                                    
                                except Exception as e:
                                    logger.error(f"Error getting spec {spec.id}: {e}")
                                    continue
                else:
                    logger.info(f"Skipping construct {construct_id} - does not have 'RANK 1-10' tag. Current tags: {construct_tags}")
            
            logger.info(f"Found {len(specs)} original specs from constructs with 'RANK 1-10' tag")
            return specs
        except Exception as e:
            logger.error(f"Error getting specs for project {project_id}: {str(e)}")
            return []

    def collect_benchmark(self, project_ids: List[str], output_dir: str = "benchmarks") -> None:
        """
        Collect benchmark data from multiple projects and save to JSON files
        
        Args:
            project_ids: List of project IDs to collect data from
            output_dir: Directory to save benchmark files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get current timestamp for the benchmark collection
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for project_id in project_ids:
            try:
                logger.info(f"Processing project {project_id}")
                
                # Get project information
                project_info = self.get_project_info(project_id)
                if not project_info:
                    logger.error(f"Failed to get project info for {project_id}, skipping...")
                    continue
                
                # Get project specs
                specs = self.get_project_specs(project_id)
                if not specs:
                    logger.error(f"No specs found for project {project_id}, skipping...")
                    continue
                
                # Create benchmark data structure
                benchmark_data = {
                    "metadata": {
                        "collected_at": timestamp,
                        "project_info": project_info
                    },
                    "code_snippets": specs  # Include all spec information
                }
                
                # Create filename using project name and timestamp
                project_name = project_info["name"].replace(" ", "_")
                filename = f"{project_name}_{timestamp}.json"
                filepath = os.path.join(output_dir, filename)
                
                # Save benchmark data with custom encoder
                with open(filepath, 'w') as f:
                    json.dump(benchmark_data, f, indent=2, cls=CustomEncoder)
                    
                logger.info(f"Successfully saved benchmark data to {filepath}")
                
            except Exception as e:
                logger.error(f"Error processing project {project_id}: {str(e)}")
                continue

def main():
    # Example usage
    collector = BenchmarkCollector()
    
    # List of project IDs to collect benchmarks from
    project_ids = [
        "1cf9f904-d506-4a27-969f-ae6db943eb55",
        # Add more project IDs here
    ]
    
    # Collect benchmarks
    collector.collect_benchmark(project_ids)

if __name__ == "__main__":
    main() 