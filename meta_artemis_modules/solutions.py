"""
Solution creation and management functions for the Meta Artemis application.
Handles creating solutions from recommendations and managing solution lifecycles.
"""

import streamlit as st
from loguru import logger
from typing import Dict, Any, List, Optional
from benchmark_evaluator_meta_artemis import MetaArtemisEvaluator, LLMType
from .utils import get_session_state, update_session_state
import pandas as pd
import json
from datetime import datetime
from meta_artemis_modules.shared_templates import META_PROMPT_TEMPLATES


def create_solutions_from_recommendations(meta_artemis_state: dict, optimization_id: str = None) -> List[Dict[str, Any]]:
    """Create solutions from selected recommendations"""
    logger.info("🔄 Starting solution creation from recommendations")
    
    try:
        # Get the selected recommendations
        selected_recommendations = meta_artemis_state.get('selected_recommendations', [])
        if not selected_recommendations:
            logger.warning("No recommendations selected for solution creation")
            return []
        
        # Get the evaluator results
        generated_recommendations = meta_artemis_state.get('generated_recommendations')
        if not generated_recommendations:
            logger.error("No generated recommendations found in state")
            return []
        
        # Group recommendations by template for solution creation
        solutions_to_create = []
        
        for selected_rec in selected_recommendations:
            spec_id = selected_rec['spec_id']
            construct_id = selected_rec['construct_id']
            template_id = selected_rec['template_id']
            
            # Find the corresponding recommendation result
            recommendation_result = None
            for spec_result in generated_recommendations['spec_results']:
                if (spec_result['spec_info']['spec_id'] == spec_id and 
                    template_id in spec_result['template_results']):
                    recommendation_result = spec_result['template_results'][template_id]['recommendation']
                    break
            
            if recommendation_result and recommendation_result.recommendation_success:
                solution_info = {
                    'spec_id': spec_id,
                    'construct_id': construct_id,
                    'template_id': template_id,
                    'recommendation_result': recommendation_result,
                    'spec_info': next((s['spec_info'] for s in generated_recommendations['spec_results'] 
                                     if s['spec_info']['spec_id'] == spec_id), None)
                }
                solutions_to_create.append(solution_info)
        
        logger.info(f"📊 Creating {len(solutions_to_create)} solutions from recommendations")
        return solutions_to_create
        
    except Exception as e:
        logger.error(f"❌ Error creating solutions from recommendations: {str(e)}")
        st.error(f"Error creating solutions: {str(e)}")
        return []


def get_solution_status_summary(solutions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get summary of solution statuses"""
    status_counts = {}
    
    for solution in solutions:
        status = solution.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return status_counts


def validate_solution_data(solution_data: Dict[str, Any]) -> bool:
    """Validate solution data structure"""
    required_fields = ['spec_id', 'construct_id', 'template_id', 'recommendation_result']
    
    for field in required_fields:
        if field not in solution_data:
            logger.error(f"Missing required field: {field}")
            return False
    
    return True


def format_solution_display_name(solution_data: Dict[str, Any]) -> str:
    """Format a display name for a solution"""
    construct_id = solution_data.get('construct_id', 'unknown')
    template_id = solution_data.get('template_id', 'unknown')
    
    # Try to get a more human-readable template name
    template_name = template_id
    if template_id in META_PROMPT_TEMPLATES:
        template_name = META_PROMPT_TEMPLATES[template_id]['name']
    elif template_id == 'baseline':
        template_name = 'Baseline'
    
    return f"{construct_id} ({template_name})"


def get_solution_metrics_summary(solution_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics summary from solution data"""
    if not solution_data.get('has_results'):
        return {"has_metrics": False, "summary": "No results available"}
    
    results_summary = solution_data.get('results_summary', {})
    
    metrics_summary = {
        "has_metrics": True,
        "runtime_count": len(results_summary.get('runtime_metrics', {})),
        "memory_count": len(results_summary.get('memory_metrics', {})),
        "cpu_count": len(results_summary.get('cpu_metrics', {})),
        "total_metrics": results_summary.get('total_metrics', 0)
    }
    
    # Add key performance indicators
    runtime_metrics = results_summary.get('runtime_metrics', {})
    if runtime_metrics:
        # Get the first runtime metric as primary indicator
        primary_metric = next(iter(runtime_metrics.values()))
        metrics_summary["primary_runtime"] = primary_metric.get('avg', 0)
    
    return metrics_summary 


async def execute_batch_solutions_async(evaluator: MetaArtemisEvaluator, batch_config: dict) -> List[Dict[str, Any]]:
    """Execute batch solution creation asynchronously"""
    try:
        batch_results = []
        
        if batch_config["source_type"] == "recommendations":
            # Create solutions from recommendations
            selected_recommendations = batch_config.get("selected_recommendations", [])
            
            for rec in selected_recommendations:
                try:
                    # Check the source type to determine how to handle the recommendation
                    source = rec.get("source", "unknown")
                    
                    if source == "placeholder":
                        # Skip placeholder entries - they don't have actual recommendations
                        result = {
                            "construct_id": rec["construct_id"],
                            "template_id": rec["template_id"],
                            "spec_name": rec["spec_name"],
                            "solution_id": None,
                            "success": False,
                            "error": "No recommendation available - placeholder entry",
                            "timestamp": datetime.now().isoformat()
                        }
                        batch_results.append(result)
                        continue
                    
                    elif source == "current_batch":
                        # This is from current batch process - has RecommendationResult object
                        recommendation = rec["recommendation"]
                        
                        spec_info = {
                            "spec_id": rec["spec_id"],
                            "construct_id": rec["construct_id"],
                            "name": rec["spec_name"]
                        }
                        
                        # Use the correct method name and parameters
                        solution = await evaluator.create_and_execute_solution_from_spec(
                            spec_info,
                            recommendation
                        )
                        
                        result = {
                            "construct_id": rec["construct_id"],
                            "template_id": rec["template_id"],
                            "spec_name": rec["spec_name"],
                            "solution_id": solution.solution_id if solution else None,
                            "success": solution.success if solution else False,
                            "error": solution.error_log if solution and hasattr(solution, 'error_log') else "",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                    elif source == "existing_artemis":
                        # This is an existing recommendation from Artemis - create solution using spec_id directly
                        try:
                            # For existing recommendations, the spec_id already contains the optimized code
                            # We just need to create a solution using that spec via Falcon client
                            from falcon_models import FullSolutionInfoRequest, SolutionSpecResponseBase
                            from falcon_models.service.code import SolutionStatusEnum
                            from uuid import UUID
                            
                            spec_id = rec.get("spec_id")
                            if not spec_id:
                                raise ValueError("No spec_id found in recommendation")
                            
                            # Create solution request
                            solution_request = FullSolutionInfoRequest(
                                specs=[SolutionSpecResponseBase(spec_id=UUID(spec_id))],
                                status=SolutionStatusEnum.created
                            )
                            
                            # Get optimization run (optional)
                            optimisation_id = await evaluator._get_or_create_optimisation()
                            
                            # Add solution using Falcon client directly
                            solution_response = evaluator.falcon_client.add_solution(
                                project_id=evaluator.project_id,
                                optimisation_id=optimisation_id,
                                solution=solution_request
                            )
                            
                            # Extract solution ID (same pattern as benchmark app)
                            if isinstance(solution_response, dict):
                                solution_id = solution_response.get('solution_id') or solution_response.get('id') or solution_response.get('solutionId')
                            else:
                                solution_id = str(solution_response)
                            
                            if solution_id:
                                result = {
                                    "construct_id": rec["construct_id"],
                                    "template_id": rec["template_id"],
                                    "spec_name": rec["spec_name"],
                                    "solution_id": solution_id,
                                    "success": True,
                                    "error": "",
                                    "timestamp": datetime.now().isoformat()
                                }
                            else:
                                raise ValueError("Could not extract solution ID from response")
                                
                        except Exception as create_error:
                            result = {
                                "construct_id": rec["construct_id"],
                                "template_id": rec["template_id"],
                                "spec_name": rec["spec_name"],
                                "solution_id": None,
                                "success": False,
                                "error": f"Failed to create solution: {str(create_error)}",
                                "timestamp": datetime.now().isoformat()
                            }
                    
                    else:
                        # Unknown source type
                        result = {
                            "construct_id": rec["construct_id"],
                            "template_id": rec["template_id"],
                            "spec_name": rec["spec_name"],
                            "solution_id": None,
                            "success": False,
                            "error": f"Unknown recommendation source: {source}",
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error creating solution for {rec.get('spec_name', 'unknown')}: {str(e)}")
                    batch_results.append({
                        "construct_id": rec.get("construct_id", "unknown"),
                        "template_id": rec.get("template_id", "unknown"),
                        "spec_name": rec.get("spec_name", "unknown"),
                        "solution_id": None,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        elif batch_config["source_type"] == "prompt_versions":
            # Create solutions from all recommendations of specific prompt versions
            from falcon_models import FullSolutionInfoRequest, SolutionSpecResponseBase
            from falcon_models.service.code import SolutionStatusEnum
            from uuid import UUID
            
            selected_recommendations = batch_config.get("selected_recommendations", [])
            selected_templates = batch_config.get("selected_templates", [])  # Now supports multiple templates
            
            # For backward compatibility, also check for single template
            if not selected_templates:
                single_template = batch_config.get("selected_template")
                if single_template:
                    selected_templates = [single_template]
            
            if not selected_recommendations:
                logger.warning("No recommendations provided for prompt versions solution creation")
                return batch_results
            
            if not selected_templates:
                logger.warning("No templates selected for prompt versions solution creation")
                return batch_results
            
            # Group recommendations by project and template
            project_template_recommendations = {}
            for rec in selected_recommendations:
                # Determine project ID from recommendation
                project_id = rec.get("project_id")
                if not project_id:
                    # Try to get project_id from evaluator if not in recommendation
                    project_id = evaluator.project_id
                
                template_name = rec.get("template_name", "unknown")
                
                # Only process recommendations for selected templates
                if template_name in selected_templates:
                    if project_id not in project_template_recommendations:
                        project_template_recommendations[project_id] = {}
                    if template_name not in project_template_recommendations[project_id]:
                        project_template_recommendations[project_id][template_name] = []
                    
                    project_template_recommendations[project_id][template_name].append(rec)
            
            logger.info(f"📊 Creating prompt version solutions for {len(project_template_recommendations)} projects with {len(selected_templates)} templates: {', '.join(selected_templates)}")
            
            # Create one solution per project-template combination
            for project_id, template_recs in project_template_recommendations.items():
                for template_name, project_recs in template_recs.items():
                    try:
                        # Collect all spec IDs from this project-template combination
                        spec_ids = []
                        construct_count = 0
                        
                        for rec in project_recs:
                            spec_id = rec.get("spec_id")
                            if spec_id and rec.get("source") != "placeholder":
                                spec_ids.append(spec_id)
                                construct_count += 1
                        
                        if not spec_ids:
                            logger.warning(f"No valid spec IDs found for project {project_id} with template '{template_name}'")
                            result = {
                                "project_id": project_id,
                                "template_id": template_name.lower().replace(" ", "_"),
                                "spec_name": f"{template_name}_{project_id[:8]}",
                                "solution_id": None,
                                "success": False,
                                "error": "No valid spec IDs found",
                                "specs_count": 0,
                                "template_name": template_name,
                                "timestamp": datetime.now().isoformat()
                            }
                            batch_results.append(result)
                            continue
                        
                        # Create solution request with all specs for this template
                        solution_specs = [SolutionSpecResponseBase(spec_id=UUID(spec_id)) for spec_id in spec_ids]
                        solution_request = FullSolutionInfoRequest(
                            specs=solution_specs,
                            status=SolutionStatusEnum.created
                        )
                        
                        # Get optimization run (optional)
                        optimisation_id = await evaluator._get_or_create_optimisation()
                        
                        # Add solution using Falcon client directly
                        solution_response = evaluator.falcon_client.add_solution(
                            project_id=project_id,
                            optimisation_id=optimisation_id,
                            solution=solution_request
                        )
                        
                        # Extract solution ID
                        if isinstance(solution_response, dict):
                            solution_id = solution_response.get('solution_id') or solution_response.get('id') or solution_response.get('solutionId')
                        else:
                            solution_id = str(solution_response)
                        
                        if solution_id:
                            result = {
                                "project_id": project_id,
                                "template_id": template_name.lower().replace(" ", "_"),
                                "spec_name": f"{template_name}_{project_id[:8]}",
                                "solution_id": solution_id,
                                "success": True,
                                "error": "",
                                "specs_count": len(spec_ids),
                                "constructs_count": construct_count,
                                "template_name": template_name,
                                "timestamp": datetime.now().isoformat()
                            }
                            logger.info(f"✅ Created prompt version solution {solution_id} for project {project_id} with {len(spec_ids)} specs from '{template_name}' template")
                        else:
                            raise ValueError("Could not extract solution ID from response")
                        
                        batch_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error creating prompt version solution for project {project_id} with template '{template_name}': {str(e)}")
                        batch_results.append({
                            "project_id": project_id,
                            "template_id": template_name.lower().replace(" ", "_"),
                            "spec_name": f"{template_name}_{project_id[:8]}",
                            "solution_id": None,
                            "success": False,
                            "error": str(e),
                            "specs_count": 0,
                            "template_name": template_name,
                            "timestamp": datetime.now().isoformat()
                        })
        
        elif batch_config["source_type"] == "original_code":
            # Create baseline solutions from original code (no spec IDs = all constructs use original code)
            from falcon_models import FullSolutionInfoRequest
            from falcon_models.service.code import SolutionStatusEnum
            
            selected_projects = batch_config.get("selected_projects", [])
            solution_name_prefix = batch_config.get("solution_name_prefix", "baseline_original")
            
            for project_id in selected_projects:
                try:
                    # Create a solution with NO specs (empty specs list = use all original constructs)
                    solution_request = FullSolutionInfoRequest(
                        specs=[],  # Empty specs = use all original constructs
                        status=SolutionStatusEnum.created
                    )
                    
                    # Get optimization run (optional)
                    optimisation_id = await evaluator._get_or_create_optimisation()
                    
                    # Add solution using Falcon client directly
                    solution_response = evaluator.falcon_client.add_solution(
                        project_id=project_id,  # Use the specific project ID
                        optimisation_id=optimisation_id,
                        solution=solution_request
                    )
                    
                    # Extract solution ID
                    if isinstance(solution_response, dict):
                        solution_id = solution_response.get('solution_id') or solution_response.get('id') or solution_response.get('solutionId')
                    else:
                        solution_id = str(solution_response)
                    
                    if solution_id:
                        result = {
                            "project_id": project_id,
                            "construct_id": "all_constructs",
                            "template_id": "original_code",
                            "spec_name": f"{solution_name_prefix}_{project_id[:8]}",
                            "solution_id": solution_id,
                            "success": True,
                            "error": "",
                            "specs_count": "all_original",
                            "timestamp": datetime.now().isoformat()
                        }
                        logger.info(f"✅ Created baseline solution {solution_id} for project {project_id} using all original constructs")
                    else:
                        raise ValueError("Could not extract solution ID from response")
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error creating baseline solution for project {project_id}: {str(e)}")
                    batch_results.append({
                        "project_id": project_id,
                        "construct_id": "all_constructs",
                        "template_id": "original_code",
                        "spec_name": f"{solution_name_prefix}_{project_id[:8]}",
                        "solution_id": None,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        else:  # existing solutions
            # Process existing solutions
            selected_solutions = batch_config.get("selected_existing_solutions", [])
            
            for solution in selected_solutions:
                try:
                    # Process existing solution (e.g., update, validate, etc.)
                    result = await evaluator.process_existing_solution(solution["solution_id"])
                    
                    batch_results.append({
                        "solution_id": solution["solution_id"],
                        "status": result.get("status") if result else "unknown",
                        "success": bool(result),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing solution {solution['solution_id']}: {str(e)}")
                    batch_results.append({
                        "solution_id": solution["solution_id"],
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        return batch_results
        
    except Exception as e:
        logger.error(f"Error in batch solution processing: {str(e)}")
        raise e


def display_batch_solution_results(batch_results: List[Dict[str, Any]], batch_config: dict = None, container=None):
    """Display batch solution results in a Streamlit container"""
    if container is None:
        container = st
        
    container.markdown("### 🔧 Batch Solution Results")
    
    if not batch_results:
        container.warning("No results to display")
        return
    
    # Summary statistics
    total_operations = len(batch_results)
    successful_operations = sum(1 for result in batch_results if result["success"])
    failed_operations = total_operations - successful_operations
    
    col1, col2, col3, col4 = container.columns(4)
    with col1:
        container.metric("Total Operations", total_operations)
    with col2:
        container.metric("Successful", successful_operations)
    with col3:
        container.metric("Failed", failed_operations)
    with col4:
        success_rate = (successful_operations / total_operations) * 100 if total_operations > 0 else 0
        container.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Results table
    container.markdown("#### 📋 Detailed Results")
    
    results_data = []
    for result in batch_results:
        # Determine the type and display name
        template_id = result.get("template_id", "unknown")
        source_type = batch_config.get("source_type", "unknown") if batch_config else "unknown"
        
        if template_id == "original_code":
            solution_type = "📋 Original Code"
            specs_info = f"({result.get('specs_count', 'Unknown')} specs)" if result.get('specs_count') else ""
        elif source_type == "prompt_versions":
            solution_type = f"🎯 {template_id.replace('_', ' ').title()} Version"
            specs_count = result.get('specs_count', 0)
            constructs_count = result.get('constructs_count', 0)
            specs_info = f"({constructs_count} constructs, {specs_count} specs)" if specs_count and constructs_count else ""
        else:
            solution_type = f"📊 {template_id.replace('_', ' ').title()}"
            specs_info = ""
        
        results_data.append({
            "Solution ID": result.get("solution_id", "N/A")[:12] + "..." if result.get("solution_id") else "N/A",
            "Type": solution_type,
            "Construct": result.get("construct_id", "N/A")[:12] + "..." if result.get("construct_id") else "N/A",
            "Spec Name": result.get("spec_name", "N/A"),
            "Specs": specs_info,
            "Success": "✅" if result["success"] else "❌",
            "Error": result.get("error", "")[:50] + "..." if result.get("error") and len(result.get("error", "")) > 50 else result.get("error", ""),
            "Timestamp": result["timestamp"][:19] if result.get("timestamp") else "N/A"
        })
    
    results_df = pd.DataFrame(results_data)
    container.dataframe(results_df, use_container_width=True) 
    
    # Show additional info for original code solutions
    original_code_results = [r for r in batch_results if r.get("template_id") == "original_code"]
    if original_code_results:
        with container.expander("📋 Original Code Solutions Details", expanded=False):
            container.markdown("**Baseline solutions created using original, unmodified code:**")
            for result in original_code_results:
                if result["success"]:
                    container.markdown(f"✅ **{result.get('spec_name', 'Unknown')}**")
                    container.markdown(f"   - Solution ID: `{result.get('solution_id', 'N/A')}`")
                    container.markdown(f"   - Construct: `{result.get('construct_id', 'N/A')}`")
                    container.markdown(f"   - Specs included: {result.get('specs_count', 'Unknown')}")
                else:
                    container.markdown(f"❌ **{result.get('spec_name', 'Unknown')}** - {result.get('error', 'Unknown error')}")
                container.markdown("---")
    
    # Show additional info for prompt version solutions
    prompt_version_results = [r for r in batch_results if batch_config and batch_config.get("source_type") == "prompt_versions"]
    if prompt_version_results:
        with container.expander("🎯 Prompt Version Solutions Details", expanded=False):
            if batch_config:
                selected_templates = batch_config.get("selected_templates", ["Unknown"])
                container.markdown(f"**Solutions created using {len(selected_templates)} templates: {', '.join(selected_templates)}:**")
            
            for result in prompt_version_results:
                if result["success"]:
                    container.markdown(f"✅ **{result.get('spec_name', 'Unknown')}**")
                    container.markdown(f"   - Solution ID: `{result.get('solution_id', 'N/A')}`")
                    container.markdown(f"   - Project ID: `{result.get('project_id', 'N/A')}`")
                    container.markdown(f"   - Template: {result.get('template_id', 'N/A').replace('_', ' ').title()}")
                    container.markdown(f"   - Constructs included: {result.get('constructs_count', 'Unknown')}")
                    container.markdown(f"   - Total specs: {result.get('specs_count', 'Unknown')}")
                else:
                    container.markdown(f"❌ **{result.get('spec_name', 'Unknown')}** - {result.get('error', 'Unknown error')}")
                container.markdown("---") 