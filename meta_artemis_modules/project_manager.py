"""
Project management functions for the Meta Artemis application.
Handles project info retrieval, existing solutions, and project-level operations.
"""

import streamlit as st
import numpy as np
from loguru import logger
from typing import Dict, Any, List, Optional
from meta_artemis_modules.evaluator import MetaArtemisEvaluator
from vision_models.service.llm import LLMType
from meta_artemis_modules.shared_templates import (
    DEFAULT_PROJECT_OPTIMISATION_IDS,
    DEFAULT_META_PROMPT_LLM,
    DEFAULT_CODE_OPTIMIZATION_LLM
)


async def get_project_info_async(project_id: str):
    """Get project information asynchronously"""
    logger.info(f"🔄 Starting async project info retrieval for project: {project_id}")
    try:
        logger.info("🤖 Creating MetaArtemisEvaluator instance")
        evaluator = MetaArtemisEvaluator(
            task_name="runtime_performance",
            meta_prompt_llm_type=LLMType(DEFAULT_META_PROMPT_LLM),
            code_optimization_llm_type=LLMType(DEFAULT_CODE_OPTIMIZATION_LLM),
            project_id=project_id
        )
        
        logger.info("🔗 Setting up API clients")
        await evaluator.setup_clients()
        
        logger.info("📊 Getting project information")
        project_info = await evaluator.get_project_info()
        logger.info(f"✅ Project info retrieved: {bool(project_info)}")
        if project_info:
            logger.info(f"📋 Project name: {project_info.get('name', 'Unknown')}")
        
        logger.info("📄 Getting project specifications")
        project_specs = await evaluator.get_project_specs()
        logger.info(f"✅ Project specs retrieved: {len(project_specs) if project_specs else 0} specs")
        
        # Get solutions from the default optimization ID
        logger.info("🔍 Getting solutions from default optimization")
        optimization_ids = DEFAULT_PROJECT_OPTIMISATION_IDS.get(project_id, [])
        optimization_id = optimization_ids[0] if optimization_ids else None
        if optimization_id:
            logger.info(f"✅ Found default optimization ID: {optimization_id}")
            try:
                # Get optimization details
                optimization = evaluator.falcon_client.get_optimisation(optimization_id)
                
                if str(optimization.project_id) == project_id:
                    # Get solutions for this optimization
                    logger.info(f"🔍 Getting solutions for optimization {optimization_id}")
                    solutions_response = evaluator.falcon_client.get_solutions(optimization_id, page=1, per_page=50)
                    logger.info(f"📊 Found {len(solutions_response.docs)} solutions")
                    
                    existing_solutions = []
                    for solution in solutions_response.docs:
                        solution_info = {
                            "solution_id": str(solution.id),
                            "optimization_id": optimization_id,
                            "optimization_name": optimization.name or "Default Optimization",
                            "status": str(solution.status),
                            "created_at": str(solution.created_at) if hasattr(solution, 'created_at') else "Unknown",
                            "specs": [{"spec_id": str(spec.spec_id), "construct_id": str(spec.construct_id)} for spec in solution.specs] if solution.specs else [],
                            "has_results": bool(solution.results),
                            "results_summary": {}
                        }
                        
                        # Extract results summary if available
                        if solution.results and hasattr(solution.results, 'values'):
                            runtime_metrics = {}
                            memory_metrics = {}
                            cpu_metrics = {}
                            
                            for metric_name, values in solution.results.values.items():
                                if values:  # Only process non-empty value lists
                                    metric_data = {
                                        "avg": np.mean(values),
                                        "min": np.min(values),
                                        "max": np.max(values),
                                        "std": np.std(values),
                                        "count": len(values),
                                        "values": values  # Store raw values for detailed analysis
                                    }
                                    
                                    if 'runtime' in metric_name.lower() or 'time' in metric_name.lower():
                                        runtime_metrics[metric_name] = metric_data
                                    elif 'memory' in metric_name.lower() or 'mem' in metric_name.lower():
                                        memory_metrics[metric_name] = metric_data
                                    elif 'cpu' in metric_name.lower():
                                        cpu_metrics[metric_name] = metric_data
                            
                            solution_info["results_summary"] = {
                                "runtime_metrics": runtime_metrics,
                                "memory_metrics": memory_metrics,
                                "cpu_metrics": cpu_metrics,
                                "total_metrics": len(runtime_metrics) + len(memory_metrics) + len(cpu_metrics)
                            }
                            
                            # Add summary statistics for quick display
                            summary_stats = []
                            for metric_type, metrics in [("Runtime", runtime_metrics), ("Memory", memory_metrics), ("CPU", cpu_metrics)]:
                                for metric_name, data in metrics.items():
                                    if metric_type == "Runtime":
                                        summary_stats.append(f"{metric_name}: {data['avg']:.3f}s (avg)")
                                    elif metric_type == "Memory":
                                        summary_stats.append(f"{metric_name}: {data['avg']:.0f} bytes (avg)")
                                    else:
                                        summary_stats.append(f"{metric_name}: {data['avg']:.3f} (avg)")
                            
                            solution_info["metrics_summary"] = summary_stats
                            solution_info["detailed_metrics"] = {
                                "runtime_count": len(runtime_metrics),
                                "memory_count": len(memory_metrics),
                                "cpu_count": len(cpu_metrics),
                                "total_measurements": sum(len(data['values']) for data in list(runtime_metrics.values()) + list(memory_metrics.values()) + list(cpu_metrics.values()))
                            }
                        else:
                            solution_info["results_summary"] = {
                                "runtime_metrics": {},
                                "memory_metrics": {},
                                "cpu_metrics": {},
                                "total_metrics": 0
                            }
                            solution_info["metrics_summary"] = []
                            solution_info["detailed_metrics"] = {
                                "runtime_count": 0,
                                "memory_count": 0,
                                "cpu_count": 0,
                                "total_measurements": 0
                            }
                        
                        existing_solutions.append(solution_info)
                    
                    logger.info(f"✅ Successfully retrieved {len(existing_solutions)} solutions")
                else:
                    logger.warning(f"❌ Optimization {optimization_id} belongs to different project: {optimization.project_id} (expected: {project_id})")
                    existing_solutions = []
            except Exception as e:
                logger.error(f"❌ Error getting solutions from optimization {optimization_id}: {str(e)}")
                existing_solutions = []
        else:
            logger.warning(f"❌ No default optimization ID found for project {project_id}")
            existing_solutions = []
        
        logger.info("✅ Project info retrieval completed successfully")
        return project_info, project_specs, existing_solutions
    except Exception as e:
        logger.error(f"❌ Error getting project info: {str(e)}")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error args: {e.args}")
        st.error(f"Error getting project info: {str(e)}")
        return None, None, None


async def get_existing_solutions_async(project_id: str, selected_optimization_ids: Optional[List[str]] = None):
    """Get existing solutions and optimizations from Artemis"""
    logger.info(f"🔍 Getting existing solutions for project: {project_id}")
    
    try:
        # Setup evaluator to access Falcon client
        evaluator = MetaArtemisEvaluator(
            task_name="runtime_performance",
            meta_prompt_llm_type=LLMType(DEFAULT_META_PROMPT_LLM),
            code_optimization_llm_type=LLMType(DEFAULT_CODE_OPTIMIZATION_LLM),
            project_id=project_id
        )
        await evaluator.setup_clients()
        
        # Get project info
        project_info = await evaluator.get_project_info()
        
        # Get existing recommendations and solutions from the project
        existing_recs = await evaluator.get_existing_recommendations()
        
        # Extract optimization IDs from recommendations
        optimization_ids = set()
        all_recommendations = (
            existing_recs.get("meta_recommendations", []) +
            existing_recs.get("baseline_recommendations", []) +
            existing_recs.get("other_recommendations", [])
        )
        
        # Get optimization IDs from AI runs in recommendations
        for rec in all_recommendations:
            if rec.get("ai_run_id"):
                try:
                    ai_run = evaluator.falcon_client.get_ai_application(rec["ai_run_id"])
                    if hasattr(ai_run, 'project_id') and str(ai_run.project_id) == project_id:
                        # Try to find associated optimizations
                        # For now, we'll use common optimization IDs and any found in the system
                        pass
                except Exception as e:
                    logger.warning(f"Could not get AI run {rec['ai_run_id']}: {str(e)}")
        
        # Use selected optimization IDs if provided, otherwise use project defaults
        if selected_optimization_ids:
            optimization_ids.update(selected_optimization_ids)
            logger.info(f"🎯 Using selected optimization IDs: {selected_optimization_ids}")
        else:
            # Add project-specific optimization IDs first, then fallbacks
            from meta_artemis_modules.shared_templates import DEFAULT_PROJECT_OPTIMISATION_IDS
            
            common_optimization_ids = []
            
            # Add project-specific optimization first if available
            if project_id in DEFAULT_PROJECT_OPTIMISATION_IDS:
                project_optimization_ids = DEFAULT_PROJECT_OPTIMISATION_IDS[project_id]
                common_optimization_ids.extend(project_optimization_ids)
                logger.info(f"🎯 Using default optimization IDs for project {project_id}: {project_optimization_ids}")
            else:
                logger.warning(f"⚠️ No default optimization ID found for project {project_id}. Add it to DEFAULT_PROJECT_OPTIMISATION_IDS in meta_artemis_modules/shared_templates.py")
            
            optimization_ids.update(common_optimization_ids)
        
        logger.info(f"🔍 Will search for solutions in optimization IDs: {list(optimization_ids)}")
        
        existing_optimizations = []
        all_solutions = []
        
        for opt_id in optimization_ids:
            try:
                # Get optimization details
                optimization = evaluator.falcon_client.get_optimisation(opt_id)
                
                if str(optimization.project_id) == project_id:
                    existing_optimizations.append({
                        "id": opt_id,
                        "name": optimization.name or "Unnamed Optimization",
                        "project_id": str(optimization.project_id),
                        "status": str(optimization.status),
                        "num_solutions": optimization.num_of_solutions or 0,
                        "created_at": str(optimization.created_at) if optimization.created_at else "Unknown",
                        "best_solutions": optimization.best_solutions
                    })
                    
                    # Get solutions for this optimization
                    try:
                        logger.info(f"🔍 Getting solutions for optimization {opt_id}")
                        solutions_response = evaluator.falcon_client.get_solutions(opt_id, page=1, per_page=50)
                        logger.info(f"📊 Found {len(solutions_response.docs)} solutions for optimization {opt_id}")
                        
                        for solution in solutions_response.docs:
                            solution_info = {
                                "solution_id": str(solution.id),
                                "optimization_id": opt_id,
                                "optimization_name": optimization.name or "Unnamed",
                                "status": str(solution.status),
                                "created_at": str(solution.created_at) if hasattr(solution, 'created_at') else "Unknown",
                                "specs": [{"spec_id": str(spec.spec_id), "construct_id": str(spec.construct_id)} for spec in solution.specs] if solution.specs else [],
                                "has_results": bool(solution.results),
                                "results_summary": {}
                            }
                            
                            # Extract results summary if available
                            if solution.results and hasattr(solution.results, 'values'):
                                runtime_metrics = {}
                                memory_metrics = {}
                                cpu_metrics = {}
                                
                                for metric_name, values in solution.results.values.items():
                                    if values:  # Only process non-empty value lists
                                        metric_data = {
                                            "avg": np.mean(values),
                                            "min": np.min(values),
                                            "max": np.max(values),
                                            "std": np.std(values),
                                            "count": len(values),
                                            "values": values  # Store raw values for detailed analysis
                                        }
                                        
                                        if 'runtime' in metric_name.lower() or 'time' in metric_name.lower():
                                            runtime_metrics[metric_name] = metric_data
                                        elif 'memory' in metric_name.lower() or 'mem' in metric_name.lower():
                                            memory_metrics[metric_name] = metric_data
                                        elif 'cpu' in metric_name.lower():
                                            cpu_metrics[metric_name] = metric_data
                                
                                solution_info["results_summary"] = {
                                    "runtime_metrics": runtime_metrics,
                                    "memory_metrics": memory_metrics,
                                    "cpu_metrics": cpu_metrics,
                                    "total_metrics": len(runtime_metrics) + len(memory_metrics) + len(cpu_metrics)
                                }
                                
                                # Add summary statistics for quick display
                                summary_stats = []
                                for metric_type, metrics in [("Runtime", runtime_metrics), ("Memory", memory_metrics), ("CPU", cpu_metrics)]:
                                    for metric_name, data in metrics.items():
                                        if metric_type == "Runtime":
                                            summary_stats.append(f"{metric_name}: {data['avg']:.3f}s (avg)")
                                        elif metric_type == "Memory":
                                            summary_stats.append(f"{metric_name}: {data['avg']:.0f} bytes (avg)")
                                        else:
                                            summary_stats.append(f"{metric_name}: {data['avg']:.3f} (avg)")
                                
                                solution_info["metrics_summary"] = summary_stats
                                solution_info["detailed_metrics"] = {
                                    "runtime_count": len(runtime_metrics),
                                    "memory_count": len(memory_metrics),
                                    "cpu_count": len(cpu_metrics),
                                    "total_measurements": sum(len(data['values']) for data in list(runtime_metrics.values()) + list(memory_metrics.values()) + list(cpu_metrics.values()))
                                }
                            else:
                                solution_info["results_summary"] = {
                                    "runtime_metrics": {},
                                    "memory_metrics": {},
                                    "cpu_metrics": {},
                                    "total_metrics": 0
                                }
                                solution_info["metrics_summary"] = []
                                solution_info["detailed_metrics"] = {
                                    "runtime_count": 0,
                                    "memory_count": 0,
                                    "cpu_count": 0,
                                    "total_measurements": 0
                                }
                            
                            all_solutions.append(solution_info)
                            
                    except Exception as e:
                        logger.warning(f"Could not get solutions for optimization {opt_id}: {str(e)}")
                else:
                    logger.warning(f"❌ Optimization {opt_id} belongs to different project: {optimization.project_id} (expected: {project_id})")
                        
            except Exception as e:
                logger.warning(f"❌ Could not get optimization {opt_id}: {str(e)}")
        
        # If no optimizations found, also check if there are any solutions directly
        if not existing_optimizations:
            logger.info("🔍 No optimizations found, checking for existing recommendations as fallback...")
            if len(all_recommendations) > 0:
                logger.info(f"📋 Found {len(all_recommendations)} existing recommendations that can be used")
                # Create a virtual optimization entry for display purposes
                existing_optimizations.append({
                    "id": "recommendations",
                    "name": f"Existing Recommendations ({len(all_recommendations)} available)",
                    "project_id": project_id,
                    "status": "available",
                    "num_solutions": 0,
                    "created_at": "Various",
                    "best_solutions": None
                })
        
        logger.info(f"✅ Found {len(existing_optimizations)} optimizations and {len(all_solutions)} solutions")
        return project_info, existing_optimizations, all_solutions
        
    except Exception as e:
        logger.error(f"❌ Error getting existing solutions: {str(e)}")
        return None, [], []


def get_optimization_configurations():
    """Get optimization configurations with project context"""
    from meta_artemis_modules.shared_templates import DEFAULT_PROJECT_OPTIMISATION_IDS
    
    # Define optimization names and descriptions - these would normally come from API
    optimization_descriptions = {
        "eef157cf-c8d4-4e7a-a2e5-79cf2f07be88": {
            "name": "Default Optimization",
            "description": "Default optimization for testing and development"
        },
        "1ef5f3e1-6138-4236-b010-79f6cdb6c2be": {
            "name": "Chess Board Optimization",
            "description": "Optimization for chess board processing"
        },
        "ab9e1675-e787-443c-8108-f7b5ca564912": {
            "name": "Big Chess Board Optimization", 
            "description": "Optimization for large chess board processing"
        },
        "05abf1c8-8ff7-457e-b7cb-25cd89130ff3": {
            "name": "BitNet Claude3.7 Optimization",
            "description": "BitNet optimization using Claude 3.7 Sonnet"
        },
        "9afc41b2-17f5-4799-90f1-1f1eb3625c42": {
            "name": "BitNet GPT-4o Optimization",
            "description": "BitNet optimization using GPT-4o"
        },
        "d91557b7-6a75-4523-a2eb-b2ff6b6e3d91": {
            "name": "llama.cpp Optimization",
            "description": "Performance optimization for llama.cpp"
        },
        "af6c8049-cf3d-4379-975f-7f4247580188": {
            "name": "faster-whisper Optimization",
            "description": "Speed optimization for faster-whisper"
        },
        "24f078c2-2c71-42ec-82e8-049edca0fa20": {
            "name": "Langflow Optimization",
            "description": "Flow optimization for Langflow"
        },
        "3f9da777-66e4-4b71-958f-abdb7456fadb": {
            "name": "Whisper GPU Optimization",
            "description": "GPU acceleration optimization for Whisper"
        },
        "f6eccc1a-6b81-4b40-bd52-5d6464e53e58": {
            "name": "QuantLib 2.0 Optimization",
            "description": "Performance optimization for QuantLib 2.0"
        },
        "a46ff34d-7037-4d79-81b1-4d7ab680cd4f": {
            "name": "QuantLib Optimization",
            "description": "Performance optimization for QuantLib"
        },
        "c4e3ef1f-c571-4de3-b474-a435e721a5f2": {
            "name": "csv-parser Optimization",
            "description": "CSV parsing performance optimization"
        },
        "e8f76f2e-5329-4cba-a122-1992fba209c2": {
            "name": "BitmapPlusPlus Optimization",
            "description": "Bitmap processing optimization"
        },
        "f2474897-bbee-43df-a7cf-c862034233aa": {
            "name": "rpcs3 Optimization",
            "description": "PlayStation 3 emulator optimization"
        },
        "25f1f709-0b46-4654-b9dd-1ed187b7a349": {
            "name": "BitNet-file Optimization",
            "description": "BitNet file processing optimization"
        },
        "787e1843-6a34-4266-a3c2-de6a82bf6793": {
            "name": "AABitNet Optimization",
            "description": "AABitNet neural network optimization"
        }
    }
    
    # Get project configurations
    project_configs = get_project_configurations()
    
    # Create optimization configurations with project context
    optimization_configs = {}
    for project_id, optimization_ids in DEFAULT_PROJECT_OPTIMISATION_IDS.items():
        if project_id in project_configs:
            project_info = project_configs[project_id]
            for optimization_id in optimization_ids:
                optimization_configs[optimization_id] = {
                    "project_id": project_id,
                    "project_name": project_info["name"],
                    "project_description": project_info["description"],
                    "optimization_name": optimization_descriptions.get(optimization_id, {}).get("name", f"Optimization {optimization_id[:8]}..."),
                    "optimization_description": optimization_descriptions.get(optimization_id, {}).get("description", f"Optimization with ID {optimization_id}")
                }
    
    return optimization_configs


def get_project_configurations():
    """Get predefined project configurations"""
    from meta_artemis_modules.shared_templates import DEFAULT_PROJECT_OPTIMISATION_IDS
    
    # Define project names and descriptions for the predefined projects
    project_descriptions = {
        "6c47d53e-7384-44d8-be9d-c186a7af480a": {
            "name": "Default Project 1",
            "description": "Default project for testing and development"
        },
        "114ba2fa-8bae-4e19-8f46-3fbef23b4a98": {
            "name": "BitNet",
            "description": "BitNet project - Neural network optimization and performance analysis"
        },
        "28334995-7488-4414-876a-fbbdd1d990f9": {
            "name": "llama.cpp",
            "description": "llama.cpp - C++ implementation of LLaMA for performance optimization"
        },
        "9f8f7777-f359-4f39-bfa8-6a0f4ebe473c": {
            "name": "faster-whisper",
            "description": "faster-whisper - Optimized Whisper implementation for speech recognition"
        },
        "0126bc6f-57c0-4148-bd3e-3d30ea7c6099": {
            "name": "Langflow",
            "description": "Langflow - Flow-based programming for LLM applications"
        },
        "1cf9f904-d506-4a27-969f-ae6db943eb55": {
            "name": "Whisper GPU",
            "description": "Whisper GPU - GPU-accelerated speech recognition optimization"
        },
        "17789b06-49be-4dec-b2bc-2d741a350328": {
            "name": "QuantLib 2.0",
            "description": "QuantLib 2.0 - Quantitative finance library version 2.0"
        },
        "f28e9994-4b44-446c-8973-7ab2037f1f55": {
            "name": "QuantLib",
            "description": "QuantLib - Quantitative finance library for financial calculations"
        },
        "a732b310-6ec1-44b5-bf4d-ac4b3618a62d": {
            "name": "csv-parser",
            "description": "csv-parser - High-performance CSV parsing and processing"
        },
        "26ecc1a2-2b9c-4733-9d5d-07d0a6608686": {
            "name": "BitmapPlusPlus", 
            "description": "BitmapPlusPlus - High-performance bitmap processing library"
        },
        "074babc9-86c9-48c5-ac96-4d350a36c9ad": {
            "name": "rpcs3",
            "description": "rpcs3 - PlayStation 3 emulator performance optimization"
        }
    }
    
    # Create configurations from DEFAULT_PROJECT_OPTIMISATION_IDS
    configurations = {}
    for project_id in DEFAULT_PROJECT_OPTIMISATION_IDS.keys():
        if project_id in project_descriptions:
            configurations[project_id] = project_descriptions[project_id]
        else:
            # Fallback for any project IDs not in our descriptions
            configurations[project_id] = {
                "name": f"Project {project_id[:8]}...",
                "description": f"Predefined project with ID {project_id}"
            }
    
    return configurations


def validate_project_id(project_id: str) -> bool:
    """Validate if project ID is in correct UUID format"""
    import re
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    return bool(uuid_pattern.match(project_id)) 
    return bool(uuid_pattern.match(project_id)) 