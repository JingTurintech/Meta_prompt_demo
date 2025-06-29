import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import os
import time
import random
from uuid import UUID
from benchmark_evaluator_meta_artemis import (
    MetaArtemisEvaluator, LLMType, save_evaluation_results, 
    load_evaluation_results, RecommendationResult, SolutionResult
)
from shared_templates import (
    OPTIMIZATION_TASKS, META_PROMPT_TEMPLATES, AVAILABLE_LLMS, 
    DEFAULT_PROJECT_OPTIMISATION_IDS
)
from loguru import logger
import sys
import colorsys
import numpy as np
from typing import Dict, Any, List
from artemis_client.falcon.client import FalconSettings, ThanosSettings, FalconClient

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def generate_colors(n):
    """Generate n distinct colors using HSV color space."""
    colors = []
    for i in range(n):
        hue = i * 0.618033988749895 % 1
        saturation = 0.7
        value = 0.95
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

def initialize_session_state():
    """Initialize all session state variables"""
    if "meta_artemis_state" not in st.session_state:
        logger.info("📝 Creating new meta_artemis_state")
        st.session_state.meta_artemis_state = {
            # Project info
            "project_id": list(DEFAULT_PROJECT_OPTIMISATION_IDS.keys())[0],  # Use first project from shared mapping
            "project_info": None,
            "project_specs": None,
            "existing_solutions": None,
            "existing_optimizations": None,
            
            # Workflow state
            "current_step": 1,
            "workflow_choice": None,  # "view_existing", "create_new", or "execute_existing"
            
            # Configuration
            "meta_prompt_llm": "gpt-4-o",
            "code_optimization_llm": "gpt-4-o",
            "selected_task": "runtime_performance",
            "selected_templates": ["standard"],
            "custom_baseline_prompt": OPTIMIZATION_TASKS["runtime_performance"]["default_prompt"],
            "custom_task_description": OPTIMIZATION_TASKS["runtime_performance"]["description"],
            "custom_worker_name": None,
            "custom_command": None,
            "evaluation_repetitions": 3,
            
            # Selected data for viewing
            "selected_solutions": [],
            "selected_optimization_id": None,
            
            # Creation state (for new workflow)
            "recommendations_generated": False,
            "generated_recommendations": None,
            "solutions_created": False,
            "execution_in_progress": False,
            
            # Results
            "current_results": None  # Current results being viewed
        }
        logger.info("✅ Session state initialized successfully")
    else:
        pass  # Session state already exists



async def get_project_info_async(project_id: str):
    """Get project information asynchronously"""
    logger.info(f"🔄 Starting async project info retrieval for project: {project_id}")
    try:
        logger.info("🤖 Creating MetaArtemisEvaluator instance")
        evaluator = MetaArtemisEvaluator(
            task_name="runtime_performance",
            meta_prompt_llm_type=LLMType("gpt-4-o"),
            code_optimization_llm_type=LLMType("gpt-4-o"),
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
        
        logger.info("🔍 Getting existing recommendations")
        existing_recs = await evaluator.get_existing_recommendations()
        logger.info(f"✅ Existing recommendations retrieved: {bool(existing_recs)}")
        if existing_recs:
            total_recs = sum(len(recs) for recs in existing_recs.values() if isinstance(recs, list))
            logger.info(f"📈 Total existing recommendations: {total_recs}")
        
        logger.info("✅ Project info retrieval completed successfully")
        return project_info, project_specs, existing_recs
    except Exception as e:
        logger.error(f"❌ Error getting project info: {str(e)}")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error args: {e.args}")
        st.error(f"Error getting project info: {str(e)}")
        return None, None, None

async def generate_recommendations_async(progress_callback=None):
    """Generate recommendations asynchronously"""
    try:
        state = st.session_state.meta_artemis_state
        
        if progress_callback:
            progress_callback({"progress": 0.1, "message": "Setting up evaluator...", "status": "setup"})
        
        evaluator = MetaArtemisEvaluator(
            task_name=state["selected_task"],
            meta_prompt_llm_type=LLMType(state["meta_prompt_llm"]),
            code_optimization_llm_type=LLMType(state["code_optimization_llm"]),
            project_id=state["project_id"],
            current_prompt=state["custom_baseline_prompt"],
            custom_task_description=state["custom_task_description"],
            selected_templates=state.get("selected_templates", []),
            custom_worker_name=state.get("custom_worker_name"),
            custom_command=state.get("custom_command"),
            evaluation_repetitions=3,
            reuse_existing_recommendations=False,  # Always generate new in this function
            selected_existing_recommendations=[]
        )
        
        # Create a custom progress callback for the evaluator that forwards structured data
        def evaluator_progress_callback(data):
            if progress_callback:
                if isinstance(data, dict):
                    progress_callback(data)
                else:
                    # Handle legacy string/dict format from evaluator
                    if isinstance(data, str):
                        progress_callback({"message": data, "status": "evaluator_update"})
                    else:
                        progress_callback(data)
        
        evaluator.progress_callback = evaluator_progress_callback
        
        await evaluator.setup_clients()
        
        if progress_callback:
            progress_callback({"progress": 0.2, "message": "Getting project information...", "status": "project_info"})
        
        # Get project info and specs
        project_info = await evaluator.get_project_info()
        project_specs = await evaluator.get_project_specs()
        
        if not project_specs:
            raise ValueError("No specifications found for project")
        
        # Derive generation modes from new configuration
        generation_modes = []
        if state.get("selected_templates"):
            generation_modes.append("meta")
        if state.get("include_baseline", False):
            generation_modes.append("baseline")
        
        # Initialize prompts storage
        evaluator.generated_prompts = {}
        evaluator.meta_prompts = {}
        
        if "meta" in generation_modes:
            if progress_callback:
                progress_callback({"progress": 0.3, "message": "Generating meta-prompts...", "status": "generating_meta_prompts"})
            
            # Generate optimization prompts using meta-prompting
            await evaluator.generate_optimization_prompts(project_info)
            
            # Send meta-prompts to UI
            if progress_callback:
                for template_id, meta_prompt_data in evaluator.meta_prompts.items():
                    if template_id != "baseline":  # Skip baseline
                        progress_callback({
                            "status": "meta_prompt_ready",
                            "template_id": template_id,
                            "filled_meta_prompt": meta_prompt_data.get("filled_template", "")
                        })
                
                for template_id, generated_prompt in evaluator.generated_prompts.items():
                    if template_id != "baseline":  # Skip baseline
                        progress_callback({
                            "status": "prompt_ready",
                            "template_id": template_id,
                            "generated_prompt": generated_prompt
                        })
        
        if "baseline" in generation_modes:
            if progress_callback:
                progress_callback({"progress": 0.35, "message": "Adding baseline prompt...", "status": "adding_baseline"})
            
            # Add baseline prompt directly
            evaluator.generated_prompts["baseline"] = state["custom_baseline_prompt"]
            evaluator.meta_prompts["baseline"] = {"name": "Baseline Prompt", "filled_template": "Direct baseline prompt usage"}
        
        if progress_callback:
            progress_callback({"progress": 0.4, "message": "Starting recommendation generation...", "status": "starting_recommendations"})
        
        # Generate recommendations for each spec
        results = {
            "project_info": project_info,
            "meta_prompts": evaluator.meta_prompts,
            "generated_prompts": evaluator.generated_prompts,
            "generation_modes": generation_modes,
            "spec_results": [],
            "summary": {
                "total_specs": len(project_specs),
                "successful_recommendations": 0,
                "failed_recommendations": 0
            }
        }
        
        # Build list of templates to use based on generation modes
        templates_to_use = []
        if "meta" in generation_modes:
            templates_to_use.extend(state["selected_templates"])
        if "baseline" in generation_modes:
            templates_to_use.append("baseline")
        
        # For both meta and baseline modes, group specs by construct and only process selected constructs
        construct_to_specs = {}
        selected_constructs = state.get("selected_constructs", [])
        
        for spec_info in project_specs:
            construct_id = spec_info["construct_id"]
            # Only include specs from selected constructs
            if construct_id in selected_constructs:
                if construct_id not in construct_to_specs:
                    construct_to_specs[construct_id] = []
                construct_to_specs[construct_id].append(spec_info)
        
        # Store recommendations per construct to reuse
        baseline_recommendations_by_construct = {}
        meta_recommendations_by_construct = {}  # Only one per construct, not per template
        
        # Calculate total work: one recommendation per construct per mode (not per template)
        num_constructs = len(construct_to_specs)
        modes_count = len([m for m in generation_modes if m in ["meta", "baseline"]])
        total_construct_work = num_constructs * modes_count
        current_work = 0
        
        logger.info(f"📊 Processing {num_constructs} selected constructs out of {len(selected_constructs)} total selected")
        logger.info(f"📊 Generation modes: {generation_modes}")
        logger.info(f"📊 Total work items: {total_construct_work} (constructs × modes)")
        
        # Process each construct once for each mode
        for construct_id, specs in construct_to_specs.items():
            # Use the first spec as representative for this construct
            representative_spec = specs[0]
            
            # Process meta-prompting mode if selected
            if "meta" in generation_modes and state.get("selected_templates"):
                current_work += 1
                
                if progress_callback:
                    progress = 0.4 + (current_work / total_construct_work) * 0.5
                    progress_callback({
                        "progress": progress,
                        "message": f"Creating meta-prompting recommendation for construct {construct_id} ({current_work}/{total_construct_work})",
                        "status": "processing_spec",
                        "spec_name": representative_spec['name'],
                        "template_name": "Meta-Prompting"
                    })
                
                # Create meta-prompting recommendations for ALL selected templates
                meta_recommendations_by_construct[construct_id] = {}
                
                for template_id in state["selected_templates"]:
                    template_name = META_PROMPT_TEMPLATES[template_id]["name"]
                    
                    recommendation_result = await evaluator.execute_recommendation_for_spec(representative_spec, template_id)
                    meta_recommendations_by_construct[construct_id][template_id] = recommendation_result
                    
                    if recommendation_result.recommendation_success:
                        results["summary"]["successful_recommendations"] += 1
                    else:
                        results["summary"]["failed_recommendations"] += 1
                    
                    # Send recommendation completion to UI
                    if progress_callback:
                        progress_callback({
                            "status": "recommendation_complete",
                            "recommendation_data": {
                                "spec_name": representative_spec['name'],
                                "template_name": template_name,
                                "success": recommendation_result.recommendation_success,
                                "is_meta": True,
                                "error": recommendation_result.error_message if not recommendation_result.recommendation_success else None
                            }
                        })
            
            # Process baseline mode if selected
            if "baseline" in generation_modes:
                current_work += 1
                
                if progress_callback:
                    progress = 0.4 + (current_work / total_construct_work) * 0.5
                    progress_callback({
                        "progress": progress,
                        "message": f"Creating baseline recommendation for construct {construct_id} ({current_work}/{total_construct_work})",
                        "status": "processing_spec",
                        "spec_name": representative_spec['name'],
                        "template_name": "Baseline"
                    })
                
                recommendation_result = await evaluator.execute_baseline_recommendation_for_spec(representative_spec)
                baseline_recommendations_by_construct[construct_id] = recommendation_result
                
                if recommendation_result.recommendation_success:
                    results["summary"]["successful_recommendations"] += 1
                else:
                    results["summary"]["failed_recommendations"] += 1
                
                # Send recommendation completion to UI
                if progress_callback:
                    progress_callback({
                        "status": "recommendation_complete",
                        "recommendation_data": {
                            "spec_name": representative_spec['name'],
                            "template_name": "Baseline",
                            "success": recommendation_result.recommendation_success,
                            "is_meta": False,
                            "error": recommendation_result.error_message if not recommendation_result.recommendation_success else None
                        }
                    })
        
        # Now create spec results by reusing the construct-level recommendations
        for spec_info in project_specs:
            construct_id = spec_info["construct_id"]
            
            # Only include specs from selected constructs
            if construct_id not in selected_constructs:
                continue
                
            spec_results = {
                "spec_info": spec_info,
                "template_results": {}
            }
            
            # Add meta-prompting results if available
            if "meta" in generation_modes and construct_id in meta_recommendations_by_construct and state.get("selected_templates"):
                for template_id in state["selected_templates"]:
                    if template_id in meta_recommendations_by_construct[construct_id]:
                        base_recommendation = meta_recommendations_by_construct[construct_id][template_id]
                        
                        # Create a recommendation result for this spec, reusing the construct-level optimization
                        if spec_info["spec_id"] == base_recommendation.spec_id:
                            # This is the original spec used for the recommendation
                            recommendation_result = base_recommendation
                        else:
                            # This is another spec in the same construct, reuse the optimization
                            recommendation_result = RecommendationResult(
                                spec_id=spec_info["spec_id"],
                                construct_id=spec_info["construct_id"],
                                original_code=spec_info["content"],
                                recommended_code=base_recommendation.recommended_code,
                                meta_prompt_used=base_recommendation.meta_prompt_used,
                                generated_prompt=base_recommendation.generated_prompt,
                                recommendation_success=base_recommendation.recommendation_success,
                                error_message=f"Reused meta-prompting recommendation from construct {construct_id}" if base_recommendation.recommendation_success else base_recommendation.error_message,
                                new_spec_id=base_recommendation.new_spec_id
                            )
                        
                        spec_results["template_results"][template_id] = {
                            "recommendation": recommendation_result,
                            "solution": None  # Will be filled in Step 3
                        }
            
            # Add baseline results if available
            if "baseline" in generation_modes and construct_id in baseline_recommendations_by_construct:
                base_recommendation = baseline_recommendations_by_construct[construct_id]
                
                # Create a recommendation result for this spec, reusing the construct-level optimization
                if spec_info["spec_id"] == base_recommendation.spec_id:
                    # This is the original spec used for the recommendation
                    recommendation_result = base_recommendation
                else:
                    # This is another spec in the same construct, reuse the optimization
                    recommendation_result = RecommendationResult(
                        spec_id=spec_info["spec_id"],
                        construct_id=spec_info["construct_id"],
                        original_code=spec_info["content"],
                        recommended_code=base_recommendation.recommended_code,
                        meta_prompt_used=base_recommendation.meta_prompt_used,
                        generated_prompt=base_recommendation.generated_prompt,
                        recommendation_success=base_recommendation.recommendation_success,
                        error_message=f"Reused baseline recommendation from construct {construct_id}" if base_recommendation.recommendation_success else base_recommendation.error_message,
                        new_spec_id=base_recommendation.new_spec_id
                    )
                
                spec_results["template_results"]["baseline"] = {
                    "recommendation": recommendation_result,
                    "solution": None  # Will be filled in Step 3
                }
            
            results["spec_results"].append(spec_results)
        
        if progress_callback:
            progress_callback({"progress": 1.0, "message": "Recommendations generated successfully!", "status": "complete"})
        
        # Log summary of recommendations
        if "baseline" in generation_modes:
            num_constructs = len(construct_to_specs)
            num_baseline_created = len(baseline_recommendations_by_construct)
            logger.info(f"Baseline recommendation summary: {num_baseline_created} created for {num_constructs} constructs")
        
        if "meta" in generation_modes:
            num_constructs = len(construct_to_specs)
            num_meta_created = len(meta_recommendations_by_construct)
            total_template_recommendations = sum(len(templates) for templates in meta_recommendations_by_construct.values())
            selected_template_names = [META_PROMPT_TEMPLATES[tid]["name"] for tid in state['selected_templates']]
            logger.info(f"Meta-prompting recommendation summary: {total_template_recommendations} recommendations created for {num_constructs} constructs using {len(state['selected_templates'])} templates: {', '.join(selected_template_names)}")
        
        for construct_id, specs in construct_to_specs.items():
            logger.info(f"Construct {construct_id}: {len(specs)} specs")
        
        return results
        
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None

def generate_recommendations_step2():
    """Generate recommendations in Step 2 with dynamic progress display"""
    
    # Check if we have completed meta-prompts to display
    state = st.session_state.meta_artemis_state
    final_meta_prompts = state.get("final_meta_prompts", {})
    
    # Static display of completed meta-prompts
    if final_meta_prompts:
        st.markdown("### 🧠 Generated Meta-Prompts")
        st.markdown("*These are the filled meta-prompt templates used to generate optimization prompts*")
        
        for template_id, prompt_data in final_meta_prompts.items():
            template_info = META_PROMPT_TEMPLATES.get(template_id, {})
            template_description = template_info.get("description", "")
            
            with st.expander(f"📝 {prompt_data['name']}", expanded=True):
                if template_description:
                    st.markdown(f"*{template_description}*")
                
                st.markdown("**🔧 Filled Meta-Prompt Template:**")
                st.markdown("*This is the actual prompt sent to the meta-prompting LLM*")
                
                content = prompt_data.get("content", "")
                if content:
                    unique_key = f"static_meta_prompt_{template_id}_{hash(content[:50])}"
                    st.text_area(
                        "Filled Meta-Prompt:",
                        value=content,
                        height=200,
                        key=unique_key,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                
                if "generated_prompt" in prompt_data and prompt_data["generated_prompt"]:
                    st.markdown("**✨ Generated Optimization Prompt:**")
                    st.markdown("*This is the optimization prompt generated by the meta-prompting LLM*")
                    
                    generated_content = prompt_data["generated_prompt"]
                    unique_gen_key = f"static_generated_prompt_{template_id}_{hash(generated_content[:50])}"
                    st.text_area(
                        "Generated Optimization Prompt:",
                        value=generated_content,
                        height=150,
                        key=unique_gen_key,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                
                st.divider()
        
        st.markdown("---")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(progress_data):
        """Simplified progress callback - only shows basic progress and status"""
        if isinstance(progress_data, dict):
            # Handle structured progress data
            message = progress_data.get("message", "")
            progress = progress_data.get("progress", None)
            
            # Update progress bar and status
            if progress is not None:
                progress_bar.progress(progress)
            if message:
                status_text.text(message)
        
        elif isinstance(progress_data, tuple) and len(progress_data) == 2:
            # Handle old format: (progress, message)
            progress, message = progress_data
            progress_bar.progress(progress)
            status_text.text(message)
        
        elif isinstance(progress_data, (int, float)):
            # Handle simple progress updates
            progress_bar.progress(progress_data)
        
        elif isinstance(progress_data, str):
            # Handle simple message updates
            status_text.text(progress_data)
    
    def display_meta_prompts_progress(container, meta_prompts):
        """Display generated meta-prompts in real-time - DISABLED for cleaner UI"""
        pass  # Remove dynamic display during generation
    
    def display_recommendations_progress(container, recommendations):
        """Display generated recommendations in real-time - DISABLED for cleaner UI"""
        pass  # Remove dynamic display during generation
    
    try:
        # Clear old meta-prompts from session state when starting new generation
        if "final_meta_prompts" in st.session_state.meta_artemis_state:
            del st.session_state.meta_artemis_state["final_meta_prompts"]
        
        # Generate recommendations with simplified progress tracking
        results = asyncio.run(generate_recommendations_async(progress_callback))
        
        if results:
            # Store results
            st.session_state.meta_artemis_state["generated_recommendations"] = results
            st.session_state.meta_artemis_state["recommendations_generated"] = True
            
            # Store the final meta-prompts for static display
            if results.get("meta_prompts"):
                final_meta_prompts = {
                    template_id: {
                        "name": meta_prompt_data.get("name", template_id),
                        "content": meta_prompt_data.get("filled_template", ""),
                        "generated_prompt": results.get("generated_prompts", {}).get(template_id, "")
                    }
                    for template_id, meta_prompt_data in results["meta_prompts"].items()
                    if template_id != "baseline"
                }
                
                # Store in session state for static display
                st.session_state.meta_artemis_state["final_meta_prompts"] = final_meta_prompts
            
            st.success("🎉 Recommendations generated successfully!")
            st.info("🎯 Now you can select which recommendations to use for solution creation.")
            
            # Add Available Recommendations button
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("📋 Available Recommendations", type="primary", key="view_available_recommendations"):
                    # Set workflow choice to view existing to show the recommendations
                    st.session_state.meta_artemis_state["workflow_choice"] = "view_existing"
                    st.rerun()
            
            # Instead of going to step 3, reload the recommendation selection interface
            st.rerun()
        else:
            status_text.text("❌ Failed to generate recommendations")
            st.error("Failed to generate recommendations. Check logs for details.")
            
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Error generating recommendations: {str(e)}")
        progress_bar.progress(0)

async def get_existing_solutions_async(project_id: str):
    """Get existing solutions and optimizations from Artemis"""
    logger.info(f"🔍 Getting existing solutions for project: {project_id}")
    
    try:
        # Setup evaluator to access Falcon client
        evaluator = MetaArtemisEvaluator(
            task_name="runtime_performance",
            meta_prompt_llm_type=LLMType("gpt-4-o"),
            code_optimization_llm_type=LLMType("gpt-4-o"),
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
        
        # Add project-specific optimization IDs first, then fallbacks
        # Use optimization IDs from shared templates
        project_specific_optimizations = DEFAULT_PROJECT_OPTIMISATION_IDS
        
        common_optimization_ids = []
        
        # Add project-specific optimization first if available
        if project_id in project_specific_optimizations:
            common_optimization_ids.append(project_specific_optimizations[project_id])
        
        # Add project ID as fallback (sometimes project ID can be used as optimization ID)
        common_optimization_ids.append(project_id)
        
        optimization_ids.update(common_optimization_ids)
        
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

def step_1_project_analysis():
    """Step 1: Project Analysis and Solution Discovery"""
    st.header("📊 Step 1: Project Analysis & Solution Discovery")
    
    # Project ID Configuration (moved from sidebar)
    st.markdown("### 🔧 Project Configuration")
    
    # Project ID selection with default options
    project_id_option = st.radio(
        "Project ID Option:",
        ["Use default project", "Enter custom project ID"],
        help="Select a default project or enter your own project ID"
    )
    
    if project_id_option == "Use default project":
        # Create display names for projects from shared mapping
        default_projects = {
            project_id: f"{project_id.split('-')[0]} ({project_id[:8]}...)" 
            for project_id in DEFAULT_PROJECT_OPTIMISATION_IDS.keys()
        }
        
        selected_project = st.selectbox(
            "Select default project:",
            options=list(default_projects.keys()),
            format_func=lambda x: default_projects[x],
            help="Select from available default projects (benchmark projects from various open-source repositories)"
        )
        project_id = selected_project
    else:
        project_id = st.text_input(
            "Project ID:",
            value=st.session_state.meta_artemis_state["project_id"],
            help="Enter the Artemis project ID to evaluate"
        )
    
    # Update session state
    st.session_state.meta_artemis_state["project_id"] = project_id
    
    if not project_id:
        st.warning("⚠️ Please select or enter a project ID to continue.")
        logger.warning("No project ID provided")
        return
    
    st.markdown(f"**Selected Project ID:** `{project_id}`")
    
    # Show Artemis link for reference
    artemis_link = f"https://artemis.turintech.ai/projects/{project_id}"
    st.markdown(f"**Artemis Link:** [View in Artemis]({artemis_link})")
    
    st.divider()
    
    # Project analysis button
    if st.button("🔍 Discover Project & Solutions", key="analyze_project_button"):
        logger.info(f"🔍 Discovering project: {project_id}")
        
        with st.spinner("🔄 Discovering project information and existing solutions..."):
            try:
                # Run async function
                project_info, existing_optimizations, existing_solutions = asyncio.run(get_existing_solutions_async(project_id))
                
                if project_info:
                    logger.info("✅ Project discovery completed successfully")
                    
                    # Store results in session state
                    st.session_state.meta_artemis_state["project_info"] = project_info
                    st.session_state.meta_artemis_state["existing_optimizations"] = existing_optimizations
                    st.session_state.meta_artemis_state["existing_solutions"] = existing_solutions
                    
                    st.success("✅ Project discovery completed successfully!")
                    st.rerun()
                else:
                    logger.error("❌ Project discovery failed")
                    st.error("❌ Failed to discover project. Please check the project ID and try again.")
                    
            except Exception as e:
                logger.error(f"❌ Error during project discovery: {str(e)}")
                st.error(f"❌ Error during project discovery: {str(e)}")
    
    # Show results if available
    project_info = st.session_state.meta_artemis_state.get("project_info")
    existing_optimizations = st.session_state.meta_artemis_state.get("existing_optimizations", [])
    existing_solutions = st.session_state.meta_artemis_state.get("existing_solutions", [])
    
    if project_info:
        
        # Project Information
        st.markdown("### 📋 Project Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {project_info['name']}")
            st.markdown(f"**Language:** {project_info['language']}")
            st.markdown(f"**Runner:** {project_info.get('runner_name', 'Not specified')}")
        
        with col2:
            st.markdown(f"**Setup Command:** `{project_info.get('setup_command', 'None')}`")
            st.markdown(f"**Compile Command:** `{project_info.get('compile_command', 'None')}`")
            st.markdown(f"**Performance Command:** `{project_info.get('perf_command', 'None')}`")
        
        st.markdown(f"**Description:** {project_info['description']}")
        
        # Existing Solutions Overview
        st.markdown("### 🎯 Existing Solutions Overview")
        
        if existing_optimizations:
            st.markdown(f"Found **{len(existing_optimizations)}** optimization runs with **{len(existing_solutions)}** total solutions:")
            
            # Show optimization summary
            for opt in existing_optimizations:
                with st.expander(f"🎯 {opt['name']} - {opt['num_solutions']} solutions"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**ID:** `{opt['id']}`")
                        st.markdown(f"**Status:** {opt['status']}")
                        st.markdown(f"**Created:** {opt['created_at']}")
                    with col2:
                        st.markdown(f"**Solutions:** {opt['num_solutions']}")
                        if opt['best_solutions']:
                            st.markdown("**Has Best Solutions:** ✅")
                        else:
                            st.markdown("**Has Best Solutions:** ❌")
            
            # Solutions with results
            solutions_with_results = [s for s in existing_solutions if s['has_results']]
            st.markdown(f"**Solutions with execution results:** {len(solutions_with_results)}/{len(existing_solutions)}")
            
            if solutions_with_results:
                # Show solutions with results and allow selection for detailed analysis
                st.markdown("#### 📊 Solutions with Results - Select for Detailed Analysis")
                
                # Create selection options
                solution_options = []
                for solution in solutions_with_results:
                    # Create a summary for each solution
                    metrics_summary = []
                    total_measurements = 0
                    
                    if solution['results_summary']:
                        # Handle new nested structure
                        for metric_type, metrics in solution['results_summary'].items():
                            if isinstance(metrics, dict):
                                for metric_name, stats in metrics.items():
                                    if isinstance(stats, dict) and 'avg' in stats:
                                        # Track total measurements
                                        if 'count' in stats:
                                            total_measurements = max(total_measurements, stats['count'])
                                        
                                        if 'runtime' in metric_name.lower():
                                            metrics_summary.append(f"Runtime: {stats['avg']:.3f}s")
                                        elif 'memory' in metric_name.lower():
                                            metrics_summary.append(f"Memory: {stats['avg']:.0f} bytes")
                                        elif 'cpu' in metric_name.lower():
                                            metrics_summary.append(f"CPU: {stats['avg']:.1f}%")
                    
                    # Add measurement count to the summary
                    if total_measurements > 0:
                        metrics_summary.append(f"{total_measurements} measurements")
                    
                    metrics_text = " | ".join(metrics_summary) if metrics_summary else "No metrics"
                    option_text = f"{solution['solution_id'][:12]}... - {solution['optimization_name']} ({metrics_text})"
                    solution_options.append((option_text, solution))
                
                # Solution selection
                if solution_options:
                    selected_option = st.selectbox(
                        "Choose a solution to analyze in detail:",
                        options=["Select a solution..."] + [opt[0] for opt in solution_options],
                        key="solution_selection"
                    )
                    
                    if selected_option != "Select a solution...":
                        # Find the selected solution
                        selected_solution = None
                        for opt_text, solution in solution_options:
                            if opt_text == selected_option:
                                selected_solution = solution
                                break
                        
                        if selected_solution:
                            # Show detailed analysis button
                            if st.button("🔍 Analyze Selected Solution", type="primary", key="analyze_solution_btn"):
                                with st.spinner("🔄 Loading detailed solution analysis..."):
                                    try:
                                        # Get detailed solution data from Artemis
                                        solution_details = asyncio.run(get_solution_details_from_artemis([selected_solution['solution_id']]))
                                        
                                        if 'error' in solution_details:
                                            st.error(f"❌ Error loading solution details: {solution_details['error']}")
                                        else:
                                            # Store in session state for detailed display
                                            st.session_state.meta_artemis_state["selected_solution_analysis"] = {
                                                'solution_details': solution_details['solution_details'],
                                                'project_id': solution_details['project_id'],
                                                'selected_solution_info': selected_solution
                                            }
                                            st.success("✅ Solution analysis loaded!")
                                            st.rerun()
                                            
                                    except Exception as e:
                                        st.error(f"❌ Error analyzing solution: {str(e)}")
                            
                            # Show basic solution info
                            with st.expander(f"📋 Basic Info: {selected_solution['solution_id'][:12]}..."):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"**Full ID:** `{selected_solution['solution_id']}`")
                                    st.markdown(f"**Status:** {selected_solution['status']}")
                                    st.markdown(f"**Created:** {selected_solution['created_at']}")
                                    st.markdown(f"**Specs:** {len(selected_solution['specs'])}")
                                with col2:
                                    st.markdown(f"**Optimization:** {selected_solution['optimization_name']}")
                                    if selected_solution['results_summary']:
                                        st.markdown("**Performance Metrics:**")
                                        # Handle new nested structure
                                        for metric_type, metrics in selected_solution['results_summary'].items():
                                            if isinstance(metrics, dict):
                                                for metric_name, stats in metrics.items():
                                                    if isinstance(stats, dict) and 'avg' in stats:
                                                        if 'runtime' in metric_name.lower():
                                                            st.markdown(f"  - {metric_name}: {stats['avg']:.3f}s (avg)")
                                                        elif 'memory' in metric_name.lower():
                                                            st.markdown(f"  - {metric_name}: {stats['avg']:.0f} bytes (avg)")
                
                # Show detailed analysis results if available
                selected_analysis = st.session_state.meta_artemis_state.get("selected_solution_analysis")
                if selected_analysis:
                    st.markdown("---")
                    st.markdown("### 🔬 Detailed Solution Analysis")
                    display_single_solution_analysis(selected_analysis)
        else:
            st.info("No existing optimization runs found for this project")
        
        # Workflow choice selection
        st.markdown("---")
        st.markdown("### 🔄 Choose Your Workflow")
        
        # Check if user has already analyzed a solution in detail
        has_detailed_analysis = st.session_state.meta_artemis_state.get("selected_solution_analysis") is not None
        
        if has_detailed_analysis:
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Analyze Another Solution", key="analyze_another_btn"):
                    # Clear current analysis to allow selecting another
                    if "selected_solution_analysis" in st.session_state.meta_artemis_state:
                        del st.session_state.meta_artemis_state["selected_solution_analysis"]
                    st.rerun()
            with col2:
                if st.button("⚡ Execute This Solution", key="execute_selected_solution_btn", type="secondary"):
                    # Get the selected solution from the analysis
                    selected_analysis = st.session_state.meta_artemis_state.get("selected_solution_analysis")
                    if selected_analysis:
                        selected_solution_info = selected_analysis.get("selected_solution_info")
                        if selected_solution_info:
                            # Store the solution for execution and set default execution config
                            st.session_state.meta_artemis_state["selected_solutions"] = [selected_solution_info]
                            st.session_state.meta_artemis_state["workflow_choice"] = "execute_existing"
                            # Set default execution configuration
                            st.session_state.meta_artemis_state["execution_config"] = {
                                "repetition_count": 3,
                                "worker_name": "jing_runner", 
                                "custom_command": None
                            }
                            # Mark that we came directly from Step 1
                            st.session_state.meta_artemis_state["came_from_step1"] = True
                            # Skip Step 2 and go directly to Step 3 (execution)
                            st.session_state.meta_artemis_state["current_step"] = 3
                            st.rerun()
            with col3:
                if st.button("🚀 Start New Evaluation", key="create_new_solutions_btn", type="primary"):
                    st.session_state.meta_artemis_state["workflow_choice"] = "create_new"
                    st.session_state.meta_artemis_state["current_step"] = 2
                    st.rerun()
        else:
            if st.button("🚀 Create New Solutions (Meta-Prompting Evaluation)", type="primary", key="create_new_solutions_direct"):
                st.session_state.meta_artemis_state["workflow_choice"] = "create_new"
                st.session_state.meta_artemis_state["current_step"] = 2
                st.rerun()
    else:
        st.info("👆 Click 'Discover Project & Solutions' to get started")

def step_2_workflow_handler():
    """Step 2: Handle selected workflow (view existing or create new)"""
    workflow_choice = st.session_state.meta_artemis_state.get("workflow_choice")
    
    if not workflow_choice:
        st.warning("⚠️ Please complete Step 1: Project Analysis and choose a workflow first")
        return
    
    if workflow_choice == "view_existing":
        step_2_view_existing_solutions()
    elif workflow_choice == "create_new":
        step_2_create_new_solutions()
    elif workflow_choice == "execute_existing":
        step_2_execute_existing_solutions()

def step_2_view_existing_solutions():
    """Step 2: View and analyze existing solutions"""
    st.header("👁️ Step 2: View Existing Solutions")
    
    project_id = st.session_state.meta_artemis_state.get("project_id")
    if not project_id:
        st.warning("⚠️ Please complete Step 1 first and select a project ID.")
        return
    
    # Use existing data from Step 1 - no need to reload!
    existing_solutions = st.session_state.meta_artemis_state.get("existing_solutions", [])
    existing_optimizations = st.session_state.meta_artemis_state.get("existing_optimizations", [])
    project_info = st.session_state.meta_artemis_state.get("project_info")
    
    # Check if Step 1 was completed properly
    if not project_info:
        st.warning("⚠️ Please complete Step 1: Project Analysis first by clicking 'Discover Project & Solutions'.")
        return
    
    if not existing_solutions:
        st.warning("⚠️ No existing solutions found for this project.")
        st.info("""
        This means Step 1 successfully connected to the project but found no existing optimization runs or solutions.
        
        **This could mean:**
        - This is a new project without any optimization runs yet
        - All existing solutions may have been deleted
        - The project exists but no optimization experiments have been run
        """)
        
        st.markdown("### 🚀 What would you like to do?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Create New Solutions", type="primary", key="create_new_from_view"):
                st.session_state.meta_artemis_state["workflow_choice"] = "create_new"
                st.rerun()
        
        with col2:
            if st.button("← Back to Step 1", key="back_to_step1_view"):
                st.session_state.meta_artemis_state["current_step"] = 1
                st.rerun()
        
        return
    
    # Filter and selection options
    st.markdown("### 🔍 Filter & Select Solutions")
    
    # Filter by optimization
    optimization_options = ["All Optimizations"] + [f"{opt['name']} ({opt['id'][:8]}...)" for opt in existing_optimizations]
    selected_optimization = st.selectbox("Filter by Optimization:", optimization_options)
    
    # Filter by status
    status_options = ["All Status"] + list(set(sol['status'] for sol in existing_solutions))
    selected_status = st.selectbox("Filter by Status:", status_options)
    
    # Filter by results availability
    results_filter = st.selectbox("Filter by Results:", ["All Solutions", "With Results Only", "Without Results"])
    
    # Apply filters
    filtered_solutions = existing_solutions
    
    if selected_optimization != "All Optimizations":
        opt_id = selected_optimization.split("(")[1].split(")")[0].replace("...", "")
        # Find the full optimization ID
        for opt in existing_optimizations:
            if opt['id'].startswith(opt_id):
                filtered_solutions = [s for s in filtered_solutions if s['optimization_id'] == opt['id']]
                break
    
    if selected_status != "All Status":
        filtered_solutions = [s for s in filtered_solutions if s['status'] == selected_status]
    
    if results_filter == "With Results Only":
        filtered_solutions = [s for s in filtered_solutions if s['has_results']]
    elif results_filter == "Without Results":
        filtered_solutions = [s for s in filtered_solutions if not s['has_results']]
    
    st.markdown(f"**Showing {len(filtered_solutions)} of {len(existing_solutions)} solutions**")
    
    # Solution selection
    if filtered_solutions:
        st.markdown("### 📊 Select Solutions to Analyze")
        
        # Create a table for solution selection
        solution_data = []
        for sol in filtered_solutions:
            # Initialize metrics
            runtime_stats = "N/A"
            memory_stats = "N/A"
            cpu_stats = "N/A"
            total_runs = 0
            
            if sol.get('has_results') and sol.get('results_summary'):
                results = sol['results_summary']
                
                # Get runtime metrics
                if results['runtime_metrics']:
                    runtime_values = []
                    for metric_data in results['runtime_metrics'].values():
                        if 'values' in metric_data:
                            runtime_values.extend(metric_data['values'])
                    if runtime_values:
                        avg_runtime = np.mean(runtime_values)
                        runtime_stats = f"{avg_runtime:.3f}s"
                        total_runs = max(total_runs, len(runtime_values))
                
                # Get memory metrics
                if results['memory_metrics']:
                    memory_values = []
                    for metric_data in results['memory_metrics'].values():
                        if 'values' in metric_data:
                            memory_values.extend(metric_data['values'])
                    if memory_values:
                        avg_memory = np.mean(memory_values)
                        memory_stats = f"{avg_memory:.0f}B"
                        total_runs = max(total_runs, len(memory_values))
                
                # Get CPU metrics
                if results['cpu_metrics']:
                    cpu_values = []
                    for metric_data in results['cpu_metrics'].values():
                        if 'values' in metric_data:
                            cpu_values.extend(metric_data['values'])
                    if cpu_values:
                        avg_cpu = np.mean(cpu_values)
                        cpu_stats = f"{avg_cpu:.1f}%"
                        total_runs = max(total_runs, len(cpu_values))
            
            solution_data.append({
                "Select": False,
                "Solution ID": sol['solution_id'][:12] + "...",
                "Optimization": sol['optimization_name'],
                "Status": sol['status'],
                "Has Results": "✅" if sol['has_results'] else "❌",
                "Avg Runtime": runtime_stats,
                "Avg Memory": memory_stats,
                "Avg CPU": cpu_stats,
                "Runs": total_runs if total_runs > 0 else "N/A",
                "Created": sol['created_at'][:19] if sol['created_at'] != "Unknown" else "Unknown",
                "Specs": len(sol['specs'])
            })
        
        if solution_data:
            # Use st.data_editor for selection
            edited_df = st.data_editor(
                pd.DataFrame(solution_data),
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=False),
                    "Solution ID": st.column_config.TextColumn("Solution ID", disabled=True),
                    "Optimization": st.column_config.TextColumn("Optimization", disabled=True),
                    "Status": st.column_config.TextColumn("Status", disabled=True),
                    "Has Results": st.column_config.TextColumn("Has Results", disabled=True),
                    "Avg Runtime": st.column_config.TextColumn("⏱️ Avg Runtime", disabled=True),
                    "Avg Memory": st.column_config.TextColumn("💾 Avg Memory", disabled=True),
                    "Avg CPU": st.column_config.TextColumn("🖥️ Avg CPU", disabled=True),
                    "Runs": st.column_config.TextColumn("🔄 Runs", disabled=True),
                    "Created": st.column_config.TextColumn("Created", disabled=True),
                    "Specs": st.column_config.NumberColumn("Specs", disabled=True)
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Get selected solutions
            selected_indices = edited_df[edited_df["Select"] == True].index.tolist()
            selected_solutions = [filtered_solutions[i] for i in selected_indices]
            
            if selected_solutions:
                st.session_state.meta_artemis_state["selected_solutions"] = selected_solutions
                st.success(f"✅ Selected {len(selected_solutions)} solutions for analysis")
                
                # Show analysis button
                if st.button("📊 Analyze Selected Solutions", key="analyze_solutions"):
                    logger.info(f"📊 Analyzing {len(selected_solutions)} selected solutions")
                    st.session_state.meta_artemis_state["current_step"] = 4  # Skip to results
                    st.rerun()
                    
                # Show detailed view of selected solutions
                st.markdown("#### 🔍 Selected Solutions Details")
                for sol in selected_solutions[:3]:  # Show first 3
                    with st.expander(f"📊 {sol['solution_id'][:12]}... - {sol['optimization_name']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Full ID:** `{sol['solution_id']}`")
                            st.markdown(f"**Status:** {sol['status']}")
                            st.markdown(f"**Created:** {sol['created_at']}")
                            st.markdown(f"**Specs:** {len(sol['specs'])}")
                        with col2:
                            st.markdown(f"**Optimization ID:** `{sol['optimization_id']}`")
                            st.markdown(f"**Has Results:** {'✅' if sol['has_results'] else '❌'}")
                            
                            # Show detailed metrics if available
                            if sol.get('detailed_metrics'):
                                metrics = sol['detailed_metrics']
                                st.markdown("**📊 Detailed Metrics:**")
                                if metrics['runtime_count'] > 0:
                                    st.markdown(f"  - Runtime metrics: {metrics['runtime_count']}")
                                if metrics['memory_count'] > 0:
                                    st.markdown(f"  - Memory metrics: {metrics['memory_count']}")
                                if metrics['cpu_count'] > 0:
                                    st.markdown(f"  - CPU metrics: {metrics['cpu_count']}")
                                if metrics['total_measurements'] > 0:
                                    st.markdown(f"  - Total measurements: {metrics['total_measurements']}")
                        
                        # Show performance summary if available
                        if sol.get('metrics_summary') and sol['metrics_summary']:
                            st.markdown("**📈 Performance Summary:**")
                            for metric_summary in sol['metrics_summary'][:5]:  # Show first 5 metrics
                                st.markdown(f"  - {metric_summary}")
                            
                            if len(sol['metrics_summary']) > 5:
                                st.markdown(f"  - ... and {len(sol['metrics_summary']) - 5} more metrics")
                        
                        # Show detailed statistics if available
                        if sol.get('results_summary') and sol['results_summary']['total_metrics'] > 0:
                            st.markdown("**📊 Detailed Statistics:**")
                            
                            # Runtime metrics
                            if sol['results_summary']['runtime_metrics']:
                                st.markdown("**⏱️ Runtime Metrics:**")
                                for metric_name, data in sol['results_summary']['runtime_metrics'].items():
                                    st.markdown(f"  - **{metric_name}:** {data['avg']:.3f}s avg (±{data['std']:.3f}s std, {data['count']} measurements)")
                            
                            # Memory metrics
                            if sol['results_summary']['memory_metrics']:
                                st.markdown("**💾 Memory Metrics:**")
                                for metric_name, data in sol['results_summary']['memory_metrics'].items():
                                    st.markdown(f"  - **{metric_name}:** {data['avg']:.0f} bytes avg (±{data['std']:.0f} std, {data['count']} measurements)")
                            
                            # CPU metrics
                            if sol['results_summary']['cpu_metrics']:
                                st.markdown("**🖥️ CPU Metrics:**")
                                for metric_name, data in sol['results_summary']['cpu_metrics'].items():
                                    st.markdown(f"  - **{metric_name}:** {data['avg']:.3f} avg (±{data['std']:.3f} std, {data['count']} measurements)")
            else:
                st.info("👆 Select solutions to analyze using the checkboxes above")
    else:
        st.warning("No solutions match the current filters")

def step_2_execute_existing_solutions():
    """Step 2: Select and configure existing solutions for execution"""
    st.header("⚡ Step 2: Execute Existing Solutions")
    
    project_id = st.session_state.meta_artemis_state.get("project_id")
    if not project_id:
        st.warning("⚠️ Please complete Step 1 first and select a project ID.")
        return
    
    # Use existing data from Step 1 - no need to reload!
    existing_solutions = st.session_state.meta_artemis_state.get("existing_solutions", [])
    existing_optimizations = st.session_state.meta_artemis_state.get("existing_optimizations", [])
    project_info = st.session_state.meta_artemis_state.get("project_info")
    
    # Check if Step 1 was completed properly
    if not project_info:
        st.warning("⚠️ Please complete Step 1: Project Analysis first by clicking 'Discover Project & Solutions'.")
        return
    
    if not existing_solutions:
        st.warning("⚠️ No existing solutions found for this project.")
        st.info("""
        This means Step 1 successfully connected to the project but found no existing optimization runs or solutions.
        
        **This could mean:**
        - This is a new project without any optimization runs yet
        - All existing solutions may have been deleted
        - The project exists but no optimization experiments have been run
        """)
        
        st.markdown("### 🚀 What would you like to do?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Create New Solutions", type="primary", key="create_new_from_execute"):
                st.session_state.meta_artemis_state["workflow_choice"] = "create_new"
                st.rerun()
        
        with col2:
            if st.button("⚡ Manual Entry", type="secondary", key="skip_to_step3"):
                st.session_state.meta_artemis_state["current_step"] = 3
                st.session_state.meta_artemis_state["manual_execution_mode"] = True
                st.rerun()
        
        with col3:
            if st.button("← Back to Step 1", key="back_to_step1_execute"):
                st.session_state.meta_artemis_state["current_step"] = 1
                st.rerun()
        
        return
    
    # Filter and selection options
    st.markdown("### 🔍 Filter & Select Solutions for Execution")
    
    # Filter by optimization
    optimization_options = ["All Optimizations"] + [f"{opt['name']} ({opt['id'][:8]}...)" for opt in existing_optimizations]
    selected_optimization = st.selectbox("Filter by Optimization:", optimization_options)
    
    # Filter by status - for execution, we want created/completed solutions
    execution_status_options = ["All Status", "created", "completed", "failed"]
    selected_status = st.selectbox("Filter by Status:", execution_status_options)
    
    # Apply filters
    filtered_solutions = existing_solutions
    
    if selected_optimization != "All Optimizations":
        opt_id = selected_optimization.split("(")[1].split(")")[0].replace("...", "")
        # Find the full optimization ID
        for opt in existing_optimizations:
            if opt['id'].startswith(opt_id):
                filtered_solutions = [s for s in filtered_solutions if s['optimization_id'] == opt['id']]
                break
    
    if selected_status != "All Status":
        filtered_solutions = [s for s in filtered_solutions if s['status'] == selected_status]
    
    st.markdown(f"**Showing {len(filtered_solutions)} of {len(existing_solutions)} solutions**")
    
    # Solution selection for execution
    if filtered_solutions:
        st.markdown("### ⚡ Select Solutions to Execute")
        
        # Create a table for solution selection
        solution_data = []
        for sol in filtered_solutions:
            solution_data.append({
                "Select": False,
                "Solution ID": sol['solution_id'][:12] + "...",
                "Optimization": sol['optimization_name'],
                "Status": sol['status'],
                "Has Results": "✅" if sol['has_results'] else "❌",
                "Created": sol['created_at'][:19] if sol['created_at'] != "Unknown" else "Unknown",
                "Specs": len(sol['specs'])
            })
        
        if solution_data:
            # Use st.data_editor for selection
            edited_df = st.data_editor(
                pd.DataFrame(solution_data),
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=False),
                    "Solution ID": st.column_config.TextColumn("Solution ID", disabled=True),
                    "Optimization": st.column_config.TextColumn("Optimization", disabled=True),
                    "Status": st.column_config.TextColumn("Status", disabled=True),
                    "Has Results": st.column_config.TextColumn("Has Results", disabled=True),
                    "Created": st.column_config.TextColumn("Created", disabled=True),
                    "Specs": st.column_config.NumberColumn("Specs", disabled=True)
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Get selected solutions
            selected_indices = edited_df[edited_df["Select"] == True].index.tolist()
            selected_solutions = [filtered_solutions[i] for i in selected_indices]
            
            if selected_solutions:
                st.session_state.meta_artemis_state["selected_solutions"] = selected_solutions
                st.success(f"✅ Selected {len(selected_solutions)} solutions for execution")
                
                # Show execution configuration
                st.markdown("### ⚙️ Execution Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    repetition_count = st.number_input("Number of repetitions", min_value=1, max_value=10, value=3,
                                                     help="Number of times to execute each solution for reliable metrics")
                    
                with col2:
                    worker_name = st.text_input("Worker name", value="jing_runner", 
                                              help="Name of the worker to use for execution")
                    custom_command = st.text_input("Custom command (optional)", 
                                                 help="Custom command to run instead of default")
                
                # Store execution settings
                st.session_state.meta_artemis_state["execution_config"] = {
                    "repetition_count": repetition_count,
                    "worker_name": worker_name,
                    "custom_command": custom_command
                }
                # Clear the came_from_step1 flag since we're coming from Step 2
                st.session_state.meta_artemis_state["came_from_step1"] = False
                
                # Show execution button
                if st.button("⚡ Execute Selected Solutions", type="primary", key="execute_solutions"):
                    logger.info(f"⚡ Executing {len(selected_solutions)} selected solutions")
                    st.session_state.meta_artemis_state["current_step"] = 3
                    st.rerun()
                    
                # Show detailed view of selected solutions
                st.markdown("#### 🔍 Selected Solutions Details")
                for sol in selected_solutions[:3]:  # Show first 3
                    with st.expander(f"📊 {sol['solution_id'][:12]}... - {sol['optimization_name']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Full ID:** `{sol['solution_id']}`")
                            st.markdown(f"**Status:** {sol['status']}")
                            st.markdown(f"**Created:** {sol['created_at']}")
                            st.markdown(f"**Specs:** {len(sol['specs'])}")
                        with col2:
                            st.markdown(f"**Optimization ID:** `{sol['optimization_id']}`")
                            st.markdown(f"**Has Results:** {'✅' if sol['has_results'] else '❌'}")
                            
                            # Show detailed metrics if available
                            if sol.get('detailed_metrics'):
                                metrics = sol['detailed_metrics']
                                st.markdown("**📊 Detailed Metrics:**")
                                if metrics['runtime_count'] > 0:
                                    st.markdown(f"  - Runtime metrics: {metrics['runtime_count']}")
                                if metrics['memory_count'] > 0:
                                    st.markdown(f"  - Memory metrics: {metrics['memory_count']}")
                                if metrics['cpu_count'] > 0:
                                    st.markdown(f"  - CPU metrics: {metrics['cpu_count']}")
                                if metrics['total_measurements'] > 0:
                                    st.markdown(f"  - Total measurements: {metrics['total_measurements']}")
                        
                        # Show performance summary if available
                        if sol.get('metrics_summary') and sol['metrics_summary']:
                            st.markdown("**📈 Performance Summary:**")
                            for metric_summary in sol['metrics_summary'][:5]:  # Show first 5 metrics
                                st.markdown(f"  - {metric_summary}")
                            
                            if len(sol['metrics_summary']) > 5:
                                st.markdown(f"  - ... and {len(sol['metrics_summary']) - 5} more metrics")
                        
                        # Show detailed statistics if available
                        if sol.get('results_summary') and sol['results_summary']['total_metrics'] > 0:
                            st.markdown("**📊 Detailed Statistics:**")
                            
                            # Runtime metrics
                            if sol['results_summary']['runtime_metrics']:
                                st.markdown("**⏱️ Runtime Metrics:**")
                                for metric_name, data in sol['results_summary']['runtime_metrics'].items():
                                    st.markdown(f"  - **{metric_name}:** {data['avg']:.3f}s avg (±{data['std']:.3f}s std, {data['count']} measurements)")
                            
                            # Memory metrics
                            if sol['results_summary']['memory_metrics']:
                                st.markdown("**💾 Memory Metrics:**")
                                for metric_name, data in sol['results_summary']['memory_metrics'].items():
                                    st.markdown(f"  - **{metric_name}:** {data['avg']:.0f} bytes avg (±{data['std']:.0f} std, {data['count']} measurements)")
                            
                            # CPU metrics
                            if sol['results_summary']['cpu_metrics']:
                                st.markdown("**🖥️ CPU Metrics:**")
                                for metric_name, data in sol['results_summary']['cpu_metrics'].items():
                                    st.markdown(f"  - **{metric_name}:** {data['avg']:.3f} avg (±{data['std']:.3f} std, {data['count']} measurements)")
            else:
                st.info("👆 Select solutions to execute using the checkboxes above")
    else:
        st.warning("No solutions match the current filters")

def step_2_create_new_solutions():
    """Step 2: Select recommendations and optionally create new ones"""
    st.header("🎯 Step 2: Select Recommendations")
    
    project_id = st.session_state.meta_artemis_state["project_id"]
    state = st.session_state.meta_artemis_state
    
    if not project_id:
        st.warning("⚠️ Please complete Step 1 first.")
        return
    
    # Auto-load existing recommendations if not already loaded
    existing_recs = state.get("existing_recommendations")
    generated_recs = state.get("generated_recommendations")
    
    if existing_recs is None:
        with st.spinner("🔄 Loading existing recommendations..."):
            try:
                evaluator = MetaArtemisEvaluator(
                    task_name="runtime_performance",
                    meta_prompt_llm_type=LLMType("gpt-4-o"),
                    code_optimization_llm_type=LLMType("gpt-4-o"),
                    project_id=project_id
                )
                
                async def load_recommendations():
                    await evaluator.setup_clients()
                    return await evaluator.get_existing_recommendations()
                
                existing_recs = asyncio.run(load_recommendations())
                state["existing_recommendations"] = existing_recs
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error loading existing recommendations: {str(e)}")
                existing_recs = {}
    
    # Section 1: Show available recommendations for selection
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📋 Available Recommendations")
    with col2:
        if st.button("🔄 Refresh Recommendations", key="refresh_recommendations", help="Reload recommendations from Artemis"):
            # Clear cached recommendations to force reload
            state["existing_recommendations"] = None
            st.rerun()
    
    # Combine existing and generated recommendations
    all_recommendations = []
    
    # Add existing recommendations
    if existing_recs:
        for rec in existing_recs.get("meta_recommendations", []):
            rec_copy = rec.copy()
            rec_copy["source"] = "existing"
            rec_copy["type"] = "Meta-Prompting"
            rec_copy["type_emoji"] = "🧠"
            all_recommendations.append(rec_copy)
        
        for rec in existing_recs.get("baseline_recommendations", []):
            rec_copy = rec.copy()
            rec_copy["source"] = "existing"
            rec_copy["type"] = "Baseline"
            rec_copy["type_emoji"] = "📝"
            all_recommendations.append(rec_copy)
        
        for rec in existing_recs.get("other_recommendations", []):
            rec_copy = rec.copy()
            rec_copy["source"] = "existing"
            rec_copy["type"] = "Other"
            rec_copy["type_emoji"] = "🔧"
            all_recommendations.append(rec_copy)
    
    # Add newly generated recommendations
    if generated_recs and generated_recs.get("spec_results"):
        for spec_result in generated_recs["spec_results"]:
            spec_info = spec_result["spec_info"]
            for template_id, template_result in spec_result["template_results"].items():
                rec = template_result["recommendation"]
                if rec.recommendation_success:
                    template_name = META_PROMPT_TEMPLATES.get(template_id, {}).get("name", template_id)
                    new_rec = {
                        "source": "generated",
                        "type": template_name,
                        "type_emoji": "🆕",
                        "spec_name": spec_info["name"],
                        "construct_file": spec_info["file"],
                        "construct_lines": f"{spec_info['lineno']}-{spec_info['end_lineno']}",
                        "spec_id": rec.spec_id,
                        "construct_id": rec.construct_id,
                        "template_id": template_id,
                        "status": "completed",
                        "created_at": "Just generated",
                        "models": [state.get("code_optimization_llm", "unknown")]
                    }
                    all_recommendations.append(new_rec)
    
    # Show summary metrics
    if all_recommendations:
        existing_count = len([r for r in all_recommendations if r["source"] == "existing"])
        generated_count = len([r for r in all_recommendations if r["source"] == "generated"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Available", len(all_recommendations))
        with col2:
            st.metric("Existing", existing_count)
        with col3:
            st.metric("Just Generated", generated_count)
        
        # Create a cleaner, table-like selection interface
        st.markdown("#### 🔍 Select Recommendations for Solution Creation")
        
        # Track selected recommendations
        selected_recommendations = []
        selected_template_results = []  # For generated recommendations
        
        # Group recommendations by construct (construct_id or construct_file)
        construct_groups = {}
        for rec in all_recommendations:
            construct_key = rec.get('construct_id') or rec['construct_file']
            if construct_key not in construct_groups:
                # Use the correct construct name field based on source
                if rec["source"] == "existing":
                    # For existing recommendations, use spec_name if available and valid (not a model name)
                    spec_name = rec.get('spec_name', '')
                    # Check if spec_name contains model names (indicates it's not a real construct name)
                    if spec_name and not any(model in spec_name.lower() for model in ['gpt', 'claude', 'llama', 'mistral', 'gemini']):
                        construct_name = spec_name
                    else:
                        # Fallback to using filename without extension as construct name
                        construct_name = rec['construct_file'].split('/')[-1].replace('.cpp', '').replace('.py', '').replace('.c', '').replace('.h', '')
                else:
                    # For generated recommendations, use spec_name from spec_info
                    construct_name = rec['spec_name']
                
                construct_groups[construct_key] = {
                    'construct_name': construct_name,
                    'construct_file': rec['construct_file'],
                    'construct_lines': rec['construct_lines'],
                    'recommendations': []
                }
            construct_groups[construct_key]['recommendations'].append(rec)
        
        # Get unique constructs for color coding
        unique_constructs = list(construct_groups.keys())
        
        # Create selection table with grouped data
        recommendation_data = []
        for construct_key, group in construct_groups.items():
            # Sort recommendations within each construct to show baseline first, then others
            sorted_recs = sorted(group['recommendations'], key=lambda x: (
                0 if x.get('template_id') == 'baseline' or x.get('type') == 'Baseline' else 1,
                x.get('template_id', ''),
                x.get('type', '')
            ))
            
            for rec in sorted_recs:
                if rec["source"] == "existing":
                    prompt_name = rec.get('prompt_info', {}).get('name', 'Unknown')
                    models_text = ', '.join(rec['models'][:2])
                    created_date = rec['created_at'][:10] if rec['created_at'] != "Unknown" else "Unknown"
                    
                    # Simplified construct info: just file and lines with color indicator
                    construct_info = f"{rec['construct_file'].split('/')[-1]}:{rec['construct_lines']}"
                    # Use color-coded circle emojis for different constructs
                    color_indicators = ["🔴", "🟠", "🟡", "🟢", "🔵", "🟣", "🟤", "⚫", "⚪", "🔘"]
                    construct_index = unique_constructs.index(construct_key)
                    color_indicator = color_indicators[construct_index % len(color_indicators)]
                    colored_construct_info = f"{color_indicator} {construct_info}"
                    
                    recommendation_data.append({
                        "Select": False,
                        "Construct": colored_construct_info,
                        "Prompt": prompt_name,
                        "Source": "Existing",
                        "Model": models_text,
                        "Created": created_date,
                        "_id": rec.get("ai_run_id"),
                        "_template_id": None,
                        "_spec_id": None,
                        "_construct_key": construct_key
                    })
                else:  # generated
                    # Simplified construct info: just file and lines with color indicator
                    construct_info = f"{rec['construct_file'].split('/')[-1]}:{rec['construct_lines']}"
                    # Use color-coded circle emojis for different constructs
                    color_indicators = ["🔴", "🟠", "🟡", "🟢", "🔵", "🟣", "🟤", "⚫", "⚪", "🔘"]
                    construct_index = unique_constructs.index(construct_key)
                    color_indicator = color_indicators[construct_index % len(color_indicators)]
                    colored_construct_info = f"{color_indicator} {construct_info}"
                    
                    # For generated recommendations, show template name as prompt
                    if rec['template_id'] == 'baseline':
                        prompt_name = "Baseline"
                    else:
                        prompt_name = rec['type']
                    
                    recommendation_data.append({
                        "Select": False,
                        "Construct": colored_construct_info,
                        "Prompt": prompt_name,
                        "Source": "Generated",
                        "Model": rec['models'][0],
                        "Created": "Just now",
                        "_id": None,
                        "_template_id": rec['template_id'],
                        "_spec_id": rec['spec_id'],
                        "_construct_key": construct_key
                    })
        
        if recommendation_data:
            # Use st.data_editor for better selection
            edited_df = st.data_editor(
                pd.DataFrame(recommendation_data),
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=False),
                    "Construct": st.column_config.TextColumn("Construct", disabled=True, width="large"),
                    "Prompt": st.column_config.TextColumn("Prompt", disabled=True, width="medium"),
                    "Source": st.column_config.TextColumn("Source", disabled=True, width="small"),
                    "Model": st.column_config.TextColumn("Model", disabled=True, width="medium"),
                    "Created": st.column_config.TextColumn("Created", disabled=True, width="small")
                },
                hide_index=True,
                use_container_width=True,
                column_order=["Select", "Construct", "Prompt", "Source", "Model", "Created"]
            )
            
            # Process selections
            selected_indices = edited_df[edited_df["Select"] == True].index.tolist()
            
            for idx in selected_indices:
                row = edited_df.iloc[idx]
                if row["Source"] == "Existing":
                    selected_recommendations.append(row["_id"])
                else:  # Generated
                    selected_template_results.append({
                        "spec_id": row["_spec_id"],
                        "template_id": row["_template_id"]
                    })
            
            total_selected = len(selected_recommendations) + len(selected_template_results)
            
            if total_selected > 0:
                st.success(f"✅ Selected {total_selected} recommendations for solution creation")
                
                # Store selections
                state["selected_existing_recommendations"] = selected_recommendations
                state["selected_template_results"] = selected_template_results
                
                # Show proceed button
                if st.button("🚀 Create Solutions from Selected Recommendations", type="primary", key="proceed_selected"):
                    logger.info(f"🚀 Creating solutions from {total_selected} selected recommendations")
                    
                    # Set recommendation mode based on what was selected
                    if selected_recommendations and selected_template_results:
                        # Both existing and generated recommendations selected
                        state["recommendation_mode"] = "mixed"
                    elif selected_recommendations:
                        # Only existing recommendations were selected
                        state["recommendation_mode"] = "existing"
                    elif selected_template_results:
                        # Only generated recommendations were selected
                        state["recommendation_mode"] = "new"
                    
                    state["current_step"] = 3
                    st.rerun()
            else:
                st.info("👆 Select recommendations above to proceed with solution creation")
    else:
        st.info("No recommendations available. Generate new ones below.")
    
    st.markdown("---")
    
    # Section 2: Optional new recommendation generation
    generate_new = st.checkbox(
        "🆕 Generate New Recommendations", 
        value=False,
        help="Check this box to configure and generate new recommendations"
    )
    
    if generate_new:
        st.markdown("### ⚙️ New Recommendation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            task_name = st.selectbox(
                "Optimization Task:",
                list(OPTIMIZATION_TASKS.keys()),
                help="Choose the type of optimization to perform"
            )
            
            meta_prompt_llm = st.selectbox(
                "Meta-Prompt LLM:",
                [llm.value for llm in AVAILABLE_LLMS],
                index=0,  # Default to gpt-4-o
                help="LLM for generating meta-prompts"
            )
            
            code_optimization_llm = st.selectbox(
                "Code Optimization LLM:",
                [llm.value for llm in AVAILABLE_LLMS],
                index=0,  # Default to gpt-4-o
                help="LLM for code optimization via Artemis"
            )
        
        with col2:
            # Meta-prompt templates as checkboxes
            st.markdown("**Meta-Prompt Templates:**")
            selected_templates = []
            
            for template_id, template_info in META_PROMPT_TEMPLATES.items():
                # Default to selecting "standard" template
                default_selected = template_id == "standard"
                is_selected = st.checkbox(
                    f"{template_info['name']}", 
                    value=state.get(f"template_{template_id}_selected", default_selected),
                    help=template_info["description"],
                    key=f"template_{template_id}_checkbox"
                )
                
                if is_selected:
                    selected_templates.append(template_id)
                    # Store the template selection state
                    state[f"template_{template_id}_selected"] = True
                else:
                    # Store the template selection state
                    state[f"template_{template_id}_selected"] = False
            
            include_baseline = st.checkbox(
                "Include Baseline Evaluation", 
                value=state.get("include_baseline", False),
                help="Also evaluate using the current baseline prompt"
            )
        
        # Advanced configuration (shown directly, no expander)
        st.markdown("### 🔧 Advanced Configuration")
        
        default_task_desc = OPTIMIZATION_TASKS[task_name]["description"]
        default_prompt = OPTIMIZATION_TASKS[task_name]["default_prompt"]
        
        custom_task_description = st.text_area(
            "Custom Task Description (optional):",
            value=state.get("custom_task_description", default_task_desc),
            help="Override the default task description"
        )
        
        # Show baseline prompt editor if baseline is selected
        if include_baseline:
            current_prompt = st.text_area(
                "Current/Baseline Prompt:",
                value=state.get("custom_baseline_prompt", default_prompt),
                help="The baseline prompt to improve upon",
                height=150
            )
        else:
            current_prompt = state.get("custom_baseline_prompt", default_prompt)
        
        # Show template editors for selected meta-prompt templates
        custom_templates = {}
        if selected_templates:
            st.markdown("**Edit Selected Meta-Prompt Templates:**")
            for template_id in selected_templates:
                if template_id in META_PROMPT_TEMPLATES:
                    template_info = META_PROMPT_TEMPLATES[template_id]
                    custom_template = st.text_area(
                        f"{template_info['name']}:",
                        value=state.get(f"custom_template_{template_id}", template_info["template"]),
                        height=200,
                        help="Edit the meta-prompt template. Available placeholders: {objective}, {project_name}, {project_description}, {project_languages}, {task_description}, {current_prompt}, {target_llm}",
                        key=f"template_{template_id}_content"
                    )
                    custom_templates[template_id] = custom_template
                    # Store the custom template content
                    state[f"custom_template_{template_id}"] = custom_template
        
        # Store configuration
        state.update({
            "selected_task": task_name,
            "meta_prompt_llm": meta_prompt_llm,
            "code_optimization_llm": code_optimization_llm,
            "selected_templates": selected_templates,
            "custom_templates": custom_templates,  # Store custom template contents
            "include_baseline": include_baseline,
            "custom_task_description": custom_task_description,
            "custom_baseline_prompt": current_prompt
        })
        
        # Check if we have templates or baseline selected
        has_templates = selected_templates or include_baseline
        
        # Show construct selection
        if has_templates:
            st.markdown("### 🎯 Select Constructs for Optimization")
            
            # Get project specs to build construct list
            project_specs = state.get("project_specs", [])
            if not project_specs:
                # Try to get project specs if not already loaded
                async def load_project_specs():
                    project_info, project_specs, existing_recs = await get_project_info_async(state["project_id"])
                    if project_specs:
                        state["project_specs"] = project_specs
                        state["project_info"] = project_info
                        state["existing_recommendations"] = existing_recs
                        return project_specs
                    return []
                
                if st.button("📋 Load Project Constructs", key="load_constructs"):
                    project_specs = asyncio.run(load_project_specs())
                    st.rerun()
            
            if project_specs:
                # Group specs by construct
                construct_groups = {}
                for spec in project_specs:
                    construct_id = spec["construct_id"]
                    if construct_id not in construct_groups:
                        construct_groups[construct_id] = {
                            "specs": [],
                            "file": spec["file"],
                            "name": spec["name"]
                        }
                    construct_groups[construct_id]["specs"].append(spec)
                
                # Create construct data for table
                construct_data = []
                
                # Initialize construct selection state if not exists
                if "construct_selections" not in st.session_state:
                    st.session_state["construct_selections"] = {}
                
                for construct_id, info in construct_groups.items():
                    filename = info["file"].split("/")[-1] if info["file"] else "unknown"
                    # Use stored selection state, default to True for new constructs
                    is_selected = st.session_state["construct_selections"].get(construct_id, True)
                    construct_data.append({
                        "Select": is_selected,
                        "Construct": f"{filename}",
                        "Spec Name": info["name"],
                        "File Path": info["file"],
                        "Specs Count": len(info["specs"]),
                        "_construct_id": construct_id  # Hidden field for reference
                    })
                
                # Add Select All / Deselect All buttons
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("✅ Select All", key="select_all_constructs"):
                        # Update all construct selections to True
                        for construct_id in construct_groups.keys():
                            st.session_state["construct_selections"][construct_id] = True
                        st.rerun()
                with col2:
                    if st.button("❌ Deselect All", key="deselect_all_constructs"):
                        # Update all construct selections to False
                        for construct_id in construct_groups.keys():
                            st.session_state["construct_selections"][construct_id] = False
                        st.rerun()
                
                # Use st.data_editor for construct selection
                if construct_data:
                    # Ensure pandas DataFrame is available
                    construct_df = pd.DataFrame(construct_data)
                    edited_df = st.data_editor(
                        construct_df,
                        column_config={
                            "Select": st.column_config.CheckboxColumn("Select", default=True),
                            "Construct": st.column_config.TextColumn("Construct", disabled=True, width="medium"),
                            "Spec Name": st.column_config.TextColumn("Spec Name", disabled=True, width="large"),
                            "File Path": st.column_config.TextColumn("File Path", disabled=True, width="large"),
                            "Specs Count": st.column_config.NumberColumn("Specs Count", disabled=True, width="small"),
                            "_construct_id": st.column_config.TextColumn("Construct ID", disabled=True, width="medium")
                        },
                        hide_index=True,
                        use_container_width=True,
                        column_order=["Select", "Construct", "Spec Name", "File Path", "Specs Count", "_construct_id"]
                    )
                    
                    # Process selections and save to session state
                    selected_indices = edited_df[edited_df["Select"] == True].index.tolist()
                    selected_constructs = [edited_df.iloc[i]["_construct_id"] for i in selected_indices]
                    
                    # Update session state with current selections
                    for i, row in edited_df.iterrows():
                        construct_id = row["_construct_id"]
                        st.session_state["construct_selections"][construct_id] = row["Select"]
                    
                    # Store selected constructs
                    state["selected_constructs"] = selected_constructs
                    
                    # Show selection summary
                    if selected_constructs:
                        st.success(f"✅ Selected {len(selected_constructs)} out of {len(construct_groups)} constructs for optimization")
                        
                        # Show generate button
                        if st.button("🚀 Generate New Recommendations", type="primary", key="generate_new"):
                            logger.info(f"🚀 Starting new recommendation generation for {len(selected_constructs)} constructs")
                            generate_recommendations_step2()
                            return
                        
                        # Show filled meta-prompts if they exist
                        final_meta_prompts = state.get("final_meta_prompts", {})
                        if final_meta_prompts:
                            st.markdown("---")
                            st.markdown("### 🧠 **Generated Meta-Prompts**")
                            st.markdown("*These are the filled meta-prompt templates used to generate optimization prompts*")
                            
                            for template_id, prompt_data in final_meta_prompts.items():
                                with st.expander(f"📝 {prompt_data['name']}", expanded=False):
                                    st.markdown("**🔧 Filled Meta-Prompt Template:**")
                                    st.text_area(
                                        "Meta-Prompt:",
                                        value=prompt_data.get("content", ""),
                                        height=200,
                                        key=f"display_meta_{template_id}",
                                        disabled=True
                                    )
                                    
                                    if prompt_data.get("generated_prompt"):
                                        st.markdown("**✨ Generated Optimization Prompt:**")
                                        st.text_area(
                                            "Optimization Prompt:",
                                            value=prompt_data["generated_prompt"],
                                            height=150,
                                            key=f"display_opt_{template_id}",
                                            disabled=True
                                        )
                    else:
                        st.info("👆 Please select at least one construct to generate recommendations for.")
                else:
                    st.warning("⚠️ No construct data available.")
            else:
                st.warning("⚠️ No project constructs found. Please complete Step 1: Project Analysis first.")
        else:
            st.warning("⚠️ Please select at least one meta-prompt template or enable baseline evaluation")
def step_2_recommendation_selection():
    """Legacy function - now redirects to workflow handler"""
    step_2_workflow_handler()



def create_solutions_from_recommendations(meta_artemis_state: dict, optimization_id: str = None) -> List[Dict[str, Any]]:
    """Create solutions from successful recommendations"""
    error_details = []
    
    try:
        logger.info("🚀 Starting solution creation from recommendations")
        logger.info(f"📋 Recommendation mode: {meta_artemis_state.get('recommendation_mode', 'Unknown')}")
        logger.info(f"🎯 Project ID: {meta_artemis_state.get('project_id', 'Unknown')}")
        logger.info(f"🔧 Optimization ID: {optimization_id}")
        
        # Setup Falcon client
        logger.info("🔌 Setting up Falcon client...")
        falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
        thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
        falcon_client = FalconClient(falcon_settings, thanos_settings)
        
        logger.info("🔐 Authenticating with Falcon...")
        falcon_client.authenticate()
        logger.info("✅ Falcon client setup and authentication completed")
        
        project_id = meta_artemis_state["project_id"]
        created_solutions = []
        
        recommendation_mode = meta_artemis_state["recommendation_mode"]
        logger.info(f"📊 Processing recommendations in mode: {recommendation_mode}")
        
        if recommendation_mode == "new":
            # Create solutions from generated recommendations
            logger.info("📋 Processing generated recommendations...")
            generated_recs = meta_artemis_state.get("generated_recommendations")
            
            if not generated_recs:
                error_msg = "❌ No generated recommendations found in session state"
                error_details.append(error_msg)
                logger.error(error_msg)
                logger.error(f"🔍 Available keys in meta_artemis_state: {list(meta_artemis_state.keys())}")
                return []
            
            logger.info(f"📊 Generated recommendations structure: {list(generated_recs.keys())}")
            
            if "spec_results" not in generated_recs:
                error_msg = "❌ No spec results found in generated recommendations"
                error_details.append(error_msg)
                logger.error(error_msg)
                logger.error(f"🔍 Available keys in generated_recs: {list(generated_recs.keys())}")
                return []
            
            spec_results_count = len(generated_recs["spec_results"])
            logger.info(f"📊 Found {spec_results_count} spec results to process")
            
            successful_recommendations = 0
            total_recommendations = 0
            
            for spec_idx, spec_result in enumerate(generated_recs["spec_results"]):
                spec_info = spec_result["spec_info"]
                logger.info(f"🔄 Processing spec {spec_idx + 1}/{spec_results_count}: {spec_info['name']} ({spec_info['file']})")
                
                template_results = spec_result.get("template_results", {})
                logger.info(f"📋 Found {len(template_results)} template results for this spec")
                
                for template_id, template_result in template_results.items():
                    total_recommendations += 1
                    recommendation = template_result["recommendation"]
                    
                    logger.info(f"🔍 Processing template '{template_id}' for spec '{spec_info['name']}'")
                    logger.info(f"   - Recommendation success: {recommendation.recommendation_success}")
                    logger.info(f"   - New spec ID: {recommendation.new_spec_id}")
                    logger.info(f"   - Error message: {recommendation.error_message}")
                    
                    if recommendation.recommendation_success and recommendation.new_spec_id:
                        successful_recommendations += 1
                        logger.info(f"✅ Creating solution for successful recommendation: {spec_info['name']} (template: {template_id})")
                        try:
                            # Create solution using the new spec created by Artemis
                            from falcon_models import FullSolutionInfoRequest, SolutionSpecResponseBase
                            from falcon_models.service.code import SolutionStatusEnum
                            from uuid import UUID
                            
                            solution_request = FullSolutionInfoRequest(
                                specs=[SolutionSpecResponseBase(spec_id=UUID(recommendation.new_spec_id))],
                                status=SolutionStatusEnum.created
                            )
                            
                            # Add solution
                            logger.info(f"🔧 Adding solution to Artemis (project: {project_id}, optimization: {optimization_id})")
                            solution_response = falcon_client.add_solution(
                                project_id=project_id,
                                optimisation_id=optimization_id,
                                solution=solution_request
                            )
                            
                            logger.info(f"📋 Solution response type: {type(solution_response)}")
                            logger.info(f"📋 Solution response: {solution_response}")
                            
                            # Extract solution ID
                            if isinstance(solution_response, dict):
                                solution_id = solution_response.get('solution_id') or solution_response.get('id') or solution_response.get('solutionId')
                                logger.info(f"🔍 Extracted solution ID from dict: {solution_id}")
                                logger.info(f"🔍 Available keys in response: {list(solution_response.keys())}")
                            else:
                                solution_id = str(solution_response)
                                logger.info(f"🔍 Converted response to string: {solution_id}")
                            
                            if solution_id:
                                # Get template name for display
                                if template_id == "baseline":
                                    template_name = "Baseline Prompt"
                                else:
                                    template_name = META_PROMPT_TEMPLATES.get(template_id, {}).get("name", template_id)
                                
                                created_solutions.append({
                                    'solution_id': solution_id,
                                    'spec_id': recommendation.new_spec_id,
                                    'construct_id': recommendation.construct_id,
                                    'template_id': template_id,
                                    'template_name': template_name,
                                    'spec_name': spec_info['name'],
                                    'construct_file': spec_info['file'],
                                    'construct_lines': f"{spec_info['lineno']}-{spec_info['end_lineno']}",
                                    'name': f"{spec_info['name']}_{template_name.replace(' ', '_')}",
                                    'status': 'created',
                                    'created_at': 'just now',
                                    'optimization_id': optimization_id,
                                    'recommendation_type': 'meta' if template_id != 'baseline' else 'baseline'
                                })
                                
                                logger.info(f"✅ Created solution {solution_id} for spec {recommendation.new_spec_id} using {template_name}")
                            else:
                                error_msg = f"❌ Could not extract solution ID from response for spec {recommendation.new_spec_id} (template: {template_id})"
                                error_details.append(error_msg)
                                logger.error(error_msg)
                                logger.error(f"Response was: {solution_response}")
                                
                        except Exception as e:
                            error_msg = f"❌ Error creating solution for spec {recommendation.new_spec_id} (template: {template_id}): {str(e)}"
                            error_details.append(error_msg)
                            logger.error(error_msg)
                            logger.exception(f"💥 Full exception details for spec {recommendation.new_spec_id}:")
                            logger.error(f"🔍 Exception type: {type(e).__name__}")
                            logger.error(f"🔍 Exception args: {e.args}")
                            continue
                    else:
                        if not recommendation.recommendation_success:
                            error_msg = f"⚠️ Skipping failed recommendation for spec {spec_info['name']} (template: {template_id}): {recommendation.error_message}"
                            error_details.append(error_msg)
                            logger.warning(error_msg)
                        elif not recommendation.new_spec_id:
                            error_msg = f"⚠️ Skipping recommendation for spec {spec_info['name']} (template: {template_id}): No new spec ID available"
                            error_details.append(error_msg)
                            logger.warning(error_msg)
            
            # Log summary
            logger.info(f"📊 Solution creation summary:")
            logger.info(f"   Total recommendations: {total_recommendations}")
            logger.info(f"   Successful recommendations: {successful_recommendations}")
            logger.info(f"   Solutions created: {len(created_solutions)}")
            logger.info(f"   Errors encountered: {len(error_details)}")
            
            # Store error details for display
            if error_details:
                st.session_state.meta_artemis_state["solution_creation_errors"] = error_details
        
        elif recommendation_mode == "existing":
            # Create solutions from existing recommendations
            logger.info("📋 Processing existing recommendations...")
            selected_recs = meta_artemis_state.get("selected_existing_recommendations", [])
            existing_recs = meta_artemis_state.get("existing_recommendations", {})
            
            if not selected_recs:
                error_msg = "❌ No existing recommendations selected"
                error_details.append(error_msg)
                logger.error(error_msg)
                logger.error(f"🔍 Available keys in meta_artemis_state: {list(meta_artemis_state.keys())}")
                return []
            
            if not existing_recs:
                error_msg = "❌ No existing recommendations data found"
                error_details.append(error_msg)
                logger.error(error_msg)
                return []
            
            logger.info(f"📊 Found {len(selected_recs)} selected existing recommendations")
            
            # Create a lookup for all recommendations
            all_recs_lookup = {}
            
            for rec in existing_recs.get("meta_recommendations", []):
                all_recs_lookup[rec["ai_run_id"]] = {**rec, "type": "meta"}
            
            for rec in existing_recs.get("baseline_recommendations", []):
                all_recs_lookup[rec["ai_run_id"]] = {**rec, "type": "baseline"}
            
            for rec in existing_recs.get("other_recommendations", []):
                all_recs_lookup[rec["ai_run_id"]] = {**rec, "type": "other"}
            
            successful_solutions = 0
            total_selected = len(selected_recs)
            
            for rec_id in selected_recs:
                if rec_id not in all_recs_lookup:
                    error_msg = f"❌ Recommendation {rec_id} not found in lookup"
                    error_details.append(error_msg)
                    logger.error(error_msg)
                    continue
                
                rec_info = all_recs_lookup[rec_id]
                logger.info(f"🔄 Processing existing recommendation: {rec_info['spec_name']} (type: {rec_info['type']})")
                
                try:
                    # For existing recommendations, the spec_id already contains the optimized code
                    # We just need to create a solution using that spec
                    from falcon_models import FullSolutionInfoRequest, SolutionSpecResponseBase
                    from falcon_models.service.code import SolutionStatusEnum
                    from uuid import UUID
                    
                    solution_request = FullSolutionInfoRequest(
                        specs=[SolutionSpecResponseBase(spec_id=UUID(rec_info["spec_id"]))],
                        status=SolutionStatusEnum.created
                    )
                    
                    # Add solution
                    logger.info(f"🔧 Adding solution to Artemis for existing recommendation {rec_id}")
                    solution_response = falcon_client.add_solution(
                        project_id=project_id,
                        optimisation_id=optimization_id,
                        solution=solution_request
                    )
                    
                    logger.info(f"📋 Solution response type: {type(solution_response)}")
                    logger.info(f"📋 Solution response: {solution_response}")
                    
                    # Extract solution ID
                    if isinstance(solution_response, dict):
                        solution_id = solution_response.get('solution_id') or solution_response.get('id') or solution_response.get('solutionId')
                        logger.info(f"🔍 Extracted solution ID from dict: {solution_id}")
                        logger.info(f"🔍 Available keys in response: {list(solution_response.keys())}")
                    else:
                        solution_id = str(solution_response)
                        logger.info(f"🔍 Converted response to string: {solution_id}")
                    
                    if solution_id:
                        # Get template name for display
                        if rec_info["type"] == "baseline":
                            template_name = "Baseline Prompt"
                        elif rec_info["type"] == "meta":
                            template_name = "Meta-prompting"
                        else:
                            template_name = "Other"
                        
                        created_solutions.append({
                            'solution_id': solution_id,
                            'spec_id': rec_info["spec_id"],
                            'construct_id': rec_info["construct_id"],
                            'template_id': rec_info["type"],
                            'template_name': template_name,
                            'spec_name': rec_info["spec_name"],
                            'construct_file': rec_info["construct_file"],
                            'construct_lines': rec_info["construct_lines"],
                            'name': f"{rec_info['spec_name']}_{template_name.replace(' ', '_')}",
                            'status': 'created',
                            'created_at': 'just now',
                            'optimization_id': optimization_id,
                            'recommendation_type': rec_info["type"],
                            'ai_run_id': rec_info["ai_run_id"]
                        })
                        
                        successful_solutions += 1
                        logger.info(f"✅ Created solution {solution_id} for existing recommendation {rec_id}")
                    else:
                        error_msg = f"❌ Could not extract solution ID from response for existing recommendation {rec_id}"
                        error_details.append(error_msg)
                        logger.error(error_msg)
                        logger.error(f"Response was: {solution_response}")
                        
                except Exception as e:
                    error_msg = f"❌ Error creating solution for existing recommendation {rec_id}: {str(e)}"
                    error_details.append(error_msg)
                    logger.error(error_msg)
                    logger.exception(f"💥 Full exception details for recommendation {rec_id}:")
                    logger.error(f"🔍 Exception type: {type(e).__name__}")
                    logger.error(f"🔍 Exception args: {e.args}")
                    continue
            
            # Log summary
            logger.info(f"📊 Existing recommendation solution creation summary:")
            logger.info(f"   Total selected: {total_selected}")
            logger.info(f"   Solutions created: {successful_solutions}")
            logger.info(f"   Errors encountered: {len(error_details)}")
            
            # Store error details for display
            if error_details:
                st.session_state.meta_artemis_state["solution_creation_errors"] = error_details
        
        return created_solutions
        
    except Exception as e:
        error_msg = f"❌ Critical error in solution creation: {str(e)}"
        error_details.append(error_msg)
        logger.error(error_msg)
        logger.exception("💥 Full critical exception details:")
        logger.error(f"🔍 Exception type: {type(e).__name__}")
        logger.error(f"🔍 Exception args: {e.args}")
        logger.error(f"🔍 Current recommendation mode: {meta_artemis_state.get('recommendation_mode', 'Unknown')}")
        logger.error(f"🔍 Current project ID: {meta_artemis_state.get('project_id', 'Unknown')}")
        
        # Store error details for display
        st.session_state.meta_artemis_state["solution_creation_errors"] = error_details
        return []

def step_3_execute_existing_solutions():
    """Step 3: Execute Existing Solutions"""
    st.header("⚡ Execute Selected Solution")
    
    # Get selected solutions and execution config
    selected_solutions = st.session_state.meta_artemis_state.get("selected_solutions", [])
    execution_config = st.session_state.meta_artemis_state.get("execution_config", {})
    came_from_step1 = st.session_state.meta_artemis_state.get("came_from_step1", False)
    
    if not selected_solutions:
        st.error("❌ No solutions selected for execution")
        return
    
    # Show solution information in a table instead of dropdown
    st.markdown("### 📊 Solution Information")
    
    # Handle single solution (from Step 1) or multiple solutions (from Step 2)
    if len(selected_solutions) == 1:
        solution = selected_solutions[0]
        
        # Create solution info table for single solution
        solution_info_data = [
            {"Property": "Solution ID", "Value": str(solution['solution_id'])},
            {"Property": "Optimization", "Value": str(solution['optimization_name'])},
            {"Property": "Status", "Value": str(solution['status'])},
            {"Property": "Created", "Value": str(solution['created_at'][:19] if solution['created_at'] != "Unknown" else "Unknown")},
            {"Property": "Specs Count", "Value": str(len(solution['specs']))},
            {"Property": "Has Results", "Value": "✅ Yes" if solution['has_results'] else "❌ No"}
        ]
        
        # Add performance metrics if available
        if solution.get('results_summary'):
            summary = solution['results_summary']
            
            # Runtime metrics
            if summary.get('runtime_metrics'):
                runtime_values = []
                for metric_data in summary['runtime_metrics'].values():
                    if 'values' in metric_data:
                        runtime_values.extend(metric_data['values'])
                if runtime_values:
                    avg_runtime = np.mean(runtime_values)
                    solution_info_data.append({"Property": "⏱️ Avg Runtime", "Value": f"{avg_runtime:.3f}s"})
            
            # Memory metrics
            if summary.get('memory_metrics'):
                memory_values = []
                for metric_data in summary['memory_metrics'].values():
                    if 'values' in metric_data:
                        memory_values.extend(metric_data['values'])
                if memory_values:
                    avg_memory = np.mean(memory_values)
                    solution_info_data.append({"Property": "💾 Avg Memory", "Value": f"{avg_memory:.0f} bytes"})
            
            # CPU metrics
            if summary.get('cpu_metrics'):
                cpu_values = []
                for metric_data in summary['cpu_metrics'].values():
                    if 'values' in metric_data:
                        cpu_values.extend(metric_data['values'])
                if cpu_values:
                    avg_cpu = np.mean(cpu_values)
                    solution_info_data.append({"Property": "🖥️ Avg CPU", "Value": f"{avg_cpu:.1f}%"})
            
            # Total runs
            total_runs = 0
            for metric_type in ['runtime_metrics', 'memory_metrics', 'cpu_metrics']:
                if summary.get(metric_type):
                    for metric_data in summary[metric_type].values():
                        if 'values' in metric_data:
                            total_runs = max(total_runs, len(metric_data['values']))
            if total_runs > 0:
                solution_info_data.append({"Property": "🔄 Previous Runs", "Value": str(total_runs)})
        
        # Display solution info table
        st.dataframe(
            pd.DataFrame(solution_info_data),
            column_config={
                "Property": st.column_config.TextColumn("Property", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="large")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        # Multiple solutions - show summary table
        st.markdown(f"**Selected {len(selected_solutions)} solutions for execution:**")
        
        solution_summary_data = []
        for i, sol in enumerate(selected_solutions, 1):
            solution_summary_data.append({
                "#": i,
                "Solution ID": sol['solution_id'][:12] + "...",
                "Optimization": sol['optimization_name'],
                "Status": sol['status'],
                "Has Results": "✅" if sol['has_results'] else "❌",
                "Specs": len(sol['specs'])
            })
        
        st.dataframe(
            pd.DataFrame(solution_summary_data),
            hide_index=True,
            use_container_width=True
        )
    
    # Execution Configuration
    st.markdown("### ⚙️ Execution Configuration")
    
    # If coming from Step 1, allow configuration
    if came_from_step1:
        # Get project info for default values
        project_info = st.session_state.meta_artemis_state.get("project_info", {})
        
        col1, col2 = st.columns(2)
        with col1:
            repetition_count = st.number_input(
                "Number of repetitions", 
                min_value=1, max_value=10, 
                value=execution_config.get("repetition_count", 3),
                help="Number of times to execute the solution for reliable metrics"
            )
        with col2:
            # Use project's default worker as default value
            default_worker = project_info.get("runner_name", "jing_runner")
            worker_name = st.text_input(
                "Worker name", 
                value=execution_config.get("worker_name", default_worker),
                help=f"Name of the worker to use for execution (project default: {default_worker})"
            )
        
        # Use project's default command as default value
        default_command = project_info.get("perf_command", "")
        custom_command = st.text_input(
            "Custom command (optional)", 
            value=execution_config.get("custom_command", "") or default_command or "",
            help=f"Custom command to run instead of default (project default: {default_command or 'None'})"
        )
        
        # Update execution config
        execution_config = {
            "repetition_count": repetition_count,
            "worker_name": worker_name,
            "custom_command": custom_command if custom_command else None
        }
        st.session_state.meta_artemis_state["execution_config"] = execution_config
    else:
        # Just display the configuration
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Repetitions:** {execution_config.get('repetition_count', 1)}")
            st.markdown(f"**Worker:** {execution_config.get('worker_name', 'jing_runner')}")
        with col2:
            command = execution_config.get('custom_command', 'Project default')
            st.markdown(f"**Command:** `{command}`")
    
    # Execution Summary
    st.markdown("### 📋 Execution Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Solutions to Execute", len(selected_solutions))
    with col2:
        st.metric("Repetitions", execution_config.get("repetition_count", 1))
    with col3:
        st.metric("Worker", execution_config.get("worker_name", "jing_runner"))
    with col4:
        total_executions = len(selected_solutions) * execution_config.get("repetition_count", 1)
        st.metric("Total Executions", total_executions)

    
    # Execute button
    if st.button("🚀 Start Execution", type="primary", key="start_existing_execution"):
        # Store the solutions for execution
        st.session_state.meta_artemis_state["solutions_for_execution"] = [
            {
                "solution_id": sol["solution_id"],
                "solution_name": f"{sol['optimization_name']}_{sol['solution_id'][:8]}",
                "spec_id": sol["specs"][0]["spec_id"] if sol["specs"] else "unknown",
                "construct_id": sol["specs"][0]["construct_id"] if sol["specs"] else "unknown",
                "template_id": "existing_solution",
                "template_name": "Existing Solution",
                "spec_name": f"existing_spec_{sol['solution_id'][:8]}",
                "construct_file": "unknown",
                "construct_lines": "unknown",
                "name": f"Existing Solution {sol['solution_id'][:8]}",
                "optimization_id": sol["optimization_id"],
                "recommendation_type": "existing",
                "ai_run_id": "existing"
            }
            for sol in selected_solutions
        ]
        
        # Set execution configuration from Step 2
        st.session_state.meta_artemis_state["custom_worker_name"] = execution_config.get("worker_name")
        st.session_state.meta_artemis_state["custom_command"] = execution_config.get("custom_command")
        st.session_state.meta_artemis_state["evaluation_repetitions"] = execution_config.get("repetition_count", 1)
        
        # Start execution
        st.session_state.meta_artemis_state["execution_in_progress"] = True
        st.rerun()
    
    # Show execution if in progress
    if st.session_state.meta_artemis_state.get("execution_in_progress", False):
        execute_solutions_step3()

def step_3_manual_execution():
    """Step 3: Manual solution entry and execution when auto-discovery fails"""
    st.header("⚡ Step 3: Manual Solution Execution")
    
    st.info("""
    **Manual Execution Mode**: Since automatic solution discovery failed, you can manually enter solution IDs to execute.
    
    This is useful when:
    - You know the solution IDs but the API discovery is failing
    - You want to execute specific solutions directly
    - There are API connection issues but execution still works
    """)
    
    project_id = st.session_state.meta_artemis_state.get("project_id")
    if not project_id:
        st.error("❌ No project ID found. Please go back to Step 1.")
        return
    
    st.markdown(f"**Project ID:** `{project_id}`")
    
    # Manual solution entry
    st.markdown("### 📝 Enter Solution IDs")
    
    # Text area for multiple solution IDs
    solution_ids_text = st.text_area(
        "Solution IDs (one per line):",
        height=150,
        placeholder="""Enter solution IDs, one per line, for example:
a1b2c3d4-e5f6-7890-abcd-ef1234567890
b2c3d4e5-f6g7-8901-bcde-f12345678901
c3d4e5f6-g7h8-9012-cdef-123456789012""",
        help="Enter the full solution IDs you want to execute, one per line"
    )
    
    # Parse solution IDs
    solution_ids = []
    if solution_ids_text.strip():
        lines = solution_ids_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) >= 8:  # Basic validation
                solution_ids.append(line)
    
    if solution_ids:
        st.success(f"✅ Found {len(solution_ids)} solution IDs")
        
        # Show parsed solution IDs
        with st.expander("📋 Parsed Solution IDs"):
            for i, sol_id in enumerate(solution_ids, 1):
                st.code(f"{i}. {sol_id}")
    
    # Execution configuration
    st.markdown("### ⚙️ Execution Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        custom_worker = st.text_input(
            "Worker Name:",
            value="jing_runner",
            help="Worker name to use for execution"
        )
        
        repetition_count = st.slider(
            "Number of Repetitions:",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of times to execute each solution for more reliable metrics"
        )
    
    with col2:
        custom_command = st.text_input(
            "Custom Command (optional):",
            value="",
            help="Custom command to run during execution (leave empty for project default)"
        )
        
        # Optimization ID (optional)
        optimization_id = st.text_input(
            "Optimization ID (optional):",
            value="",
            help="If you know the optimization ID, enter it here"
        )
    
    # Execute button
    if solution_ids and st.button("🚀 Execute Solutions", type="primary", key="manual_execute"):
        # Convert to the format expected by the execution system
        solutions_for_execution = []
        for i, sol_id in enumerate(solution_ids):
            solutions_for_execution.append({
                "solution_id": sol_id,
                "solution_name": f"Manual_Solution_{i+1}",
                "spec_id": "manual_entry",
                "construct_id": "manual_entry", 
                "template_id": "manual",
                "template_name": "Manual Entry",
                "spec_name": f"manual_spec_{i+1}",
                "construct_file": "manual_entry",
                "construct_lines": "manual_entry",
                "name": f"Manual Solution {i+1}",
                "optimization_id": optimization_id if optimization_id else "manual",
                "recommendation_type": "manual",
                "ai_run_id": "manual"
            })
        
        # Store execution data
        st.session_state.meta_artemis_state["solutions_for_execution"] = solutions_for_execution
        st.session_state.meta_artemis_state["custom_worker_name"] = custom_worker if custom_worker else None
        st.session_state.meta_artemis_state["custom_command"] = custom_command if custom_command else None
        # Manual repetition is no longer needed - using evaluation_repetitions parameter
        
        # Start execution
        st.session_state.meta_artemis_state["execution_in_progress"] = True
        st.success(f"🚀 Starting execution of {len(solution_ids)} solutions...")
        st.rerun()
    
    # Show execution if in progress
    if st.session_state.meta_artemis_state.get("execution_in_progress", False):
        execute_solutions_step3()
    
    # Back button
    st.markdown("---")
    if st.button("← Back to Step 2", key="back_to_step2_manual"):
        st.session_state.meta_artemis_state["manual_execution_mode"] = False
        st.session_state.meta_artemis_state["current_step"] = 2
        st.rerun()

def step_3_solution_execution():
    """Step 3: Solution Creation and Execution"""
    st.header("🚀 Step 3: Solution Creation and Execution")
    
    # Check for manual execution mode
    manual_mode = st.session_state.meta_artemis_state.get("manual_execution_mode", False)
    if manual_mode:
        step_3_manual_execution()
        return
    
    # Check if we have the necessary data from step 2
    workflow_choice = st.session_state.meta_artemis_state.get("workflow_choice")
    project_id = st.session_state.meta_artemis_state.get("project_id")
    
    if not workflow_choice or not project_id:
        st.warning("⚠️ Please complete Step 1: Project Analysis first")
        return
    
    # Check if we have the necessary data based on workflow
    if workflow_choice == "execute_existing":
        # For execute existing workflow, check for selected solutions
        selected_solutions = st.session_state.meta_artemis_state.get("selected_solutions", [])
        if not selected_solutions:
            st.warning("⚠️ Please go back to Step 2 and select existing solutions to execute")
            return
    elif workflow_choice == "create_new":
        # For create new workflow, check for recommendation mode
        recommendation_mode = st.session_state.meta_artemis_state.get("recommendation_mode")
        if not recommendation_mode:
            st.warning("⚠️ Please complete Step 2: Recommendation Selection first")
            return
        
        # Check if we have selected recommendations
        if recommendation_mode == "existing":
            selected_recs = st.session_state.meta_artemis_state.get("selected_existing_recommendations", [])
            if not selected_recs:
                st.warning("⚠️ Please go back to Step 2 and select existing recommendations")
                return
        elif recommendation_mode == "new":
            # For new recommendations, we need either generated_recommendations or selected_template_results
            generated_recs = st.session_state.meta_artemis_state.get("generated_recommendations")
            selected_template_results = st.session_state.meta_artemis_state.get("selected_template_results", [])
            if not generated_recs and not selected_template_results:
                st.warning("⚠️ Please go back to Step 2 and generate new recommendations")
                return
        elif recommendation_mode == "mixed":
            # For mixed mode, check if we have both types
            selected_recs = st.session_state.meta_artemis_state.get("selected_existing_recommendations", [])
            selected_template_results = st.session_state.meta_artemis_state.get("selected_template_results", [])
            if not selected_recs and not selected_template_results:
                st.warning("⚠️ Please go back to Step 2 and select recommendations")
                return
    else:
        st.warning("⚠️ Invalid workflow. Please go back to Step 1.")
        return
    
    # Handle different workflows
    if workflow_choice == "execute_existing":
        # For execute existing workflow, go directly to execution
        step_3_execute_existing_solutions()
        return
    
    # For create_new workflow, continue with the original logic
    # Execution Configuration (moved from sidebar)
    st.markdown("### ⚙️ Execution Configuration")
    
    project_info = st.session_state.meta_artemis_state.get("project_info", {})
    
    # Show project defaults first
    st.markdown("#### 📋 Project Defaults")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text(f"Worker: {project_info.get('runner_name', 'Not set')}")
    with col2:
        st.text(f"Command: {project_info.get('perf_command', 'Not set')}")
    with col3:
        st.text(f"Compile: {project_info.get('compile_command', 'Not set')}")
    
    st.markdown("#### ⚙️ Override Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        custom_worker_name = st.text_input(
            "Custom Worker (optional):",
            value=st.session_state.meta_artemis_state.get("custom_worker_name", "") or "",
            help="Override project default worker"
        )
        st.session_state.meta_artemis_state["custom_worker_name"] = custom_worker_name if custom_worker_name else None
        
        custom_command = st.text_input(
            "Custom Command (optional):",
            value=st.session_state.meta_artemis_state.get("custom_command", "") or "",
            help="Override project default performance command"
        )
        st.session_state.meta_artemis_state["custom_command"] = custom_command if custom_command else None
    
    with col2:
        evaluation_repetitions = st.slider(
            "Evaluation Repetitions:",
            min_value=1,
            max_value=10,
            value=st.session_state.meta_artemis_state.get("evaluation_repetitions", 3),
            help="Number of times to execute each solution for reliable metrics"
        )
        st.session_state.meta_artemis_state["evaluation_repetitions"] = evaluation_repetitions
        
        st.info(f"ℹ️ Each solution will be executed with {evaluation_repetitions} repetitions for reliable performance metrics.")
    
    st.divider()
    
    # Initialize step 3 state if not exists
    if "step_3_state" not in st.session_state.meta_artemis_state:
        st.session_state.meta_artemis_state["step_3_state"] = {
            "solutions_created": False,
            "optimization_selected": False,
            "selected_optimization_id": None,
            "created_solutions": [],
            "execution_ready": False
        }
    
    step_3_state = st.session_state.meta_artemis_state["step_3_state"]
    
    # Step 3A: Show Selected Recommendations Summary (like Step 2)
    st.markdown("### 📋 Selected Recommendations Summary")
    
    recommendation_mode = st.session_state.meta_artemis_state["recommendation_mode"]
    
    # Display selected recommendations in a clear format (similar to Step 2)
    if recommendation_mode == "new":
        # Show selected generated recommendations
        generated_recs = st.session_state.meta_artemis_state.get("generated_recommendations")
        if generated_recs and "spec_results" in generated_recs:
            
            # Count and display selected recommendations
            selected_recommendations = []
            for spec_result in generated_recs["spec_results"]:
                spec_info = spec_result["spec_info"]
                for template_id, template_result in spec_result.get("template_results", {}).items():
                    if template_result["recommendation"].recommendation_success:
                        if template_id == "baseline":
                            template_name = "📝 Baseline Prompt"
                        else:
                            template_name = f"🧠 {META_PROMPT_TEMPLATES[template_id]['name']}"
                        
                        selected_recommendations.append({
                            'template_name': template_name,
                            'spec_name': spec_info['name'],
                            'file': spec_info['file'],
                            'lines': f"{spec_info['lineno']}-{spec_info['end_lineno']}",
                            'new_spec_id': template_result["recommendation"].new_spec_id
                        })
            
            if selected_recommendations:
                st.info(f"✅ **{len(selected_recommendations)} successful recommendations** ready for solution creation:")
                
                for i, rec in enumerate(selected_recommendations, 1):
                    st.markdown(f"**{i}.** {rec['template_name']} | **{rec['spec_name']}** | `{rec['file']}:{rec['lines']}` | Spec ID: `{rec['new_spec_id'][:8]}...`")
            else:
                st.error("❌ No successful recommendations found")
                return
        else:
            st.error("❌ No generated recommendations found")
            return
    
    elif recommendation_mode == "existing":
        # Show ONLY the selected existing recommendations
        selected_recs = st.session_state.meta_artemis_state.get("selected_existing_recommendations", [])
        existing_recs = st.session_state.meta_artemis_state.get("existing_recommendations", {})
        
        if selected_recs and existing_recs:
            # Create a lookup for all recommendations
            all_recs_lookup = {}
            
            for rec in existing_recs.get("meta_recommendations", []):
                all_recs_lookup[rec["ai_run_id"]] = {**rec, "type": "🧠 Meta-prompting"}
            
            for rec in existing_recs.get("baseline_recommendations", []):
                all_recs_lookup[rec["ai_run_id"]] = {**rec, "type": "📝 Baseline"}
            
            for rec in existing_recs.get("other_recommendations", []):
                all_recs_lookup[rec["ai_run_id"]] = {**rec, "type": "❓ Other"}
            
            # Filter to ONLY the selected recommendations
            selected_rec_info = []
            for rec_id in selected_recs:
                if rec_id in all_recs_lookup:
                    selected_rec_info.append(all_recs_lookup[rec_id])
            
            if selected_rec_info:
                st.info(f"✅ **{len(selected_rec_info)} existing recommendations** selected for solution creation:")
                
                for i, rec in enumerate(selected_rec_info, 1):
                    status_color = "🟢" if rec['status'] == 'completed' else "🔴" if rec['status'] == 'failed' else "🟡"
                    st.markdown(f"**{i}.** {rec['type']} | **{rec['spec_name']}** | `{rec['construct_file']}:{rec['construct_lines']}` | {status_color} {rec['status']}")
            else:
                st.error("❌ No matching existing recommendations found")
                return
        else:
            st.error("❌ No existing recommendations selected")
            return
    
    elif recommendation_mode == "mixed":
        # Show both existing and generated recommendations
        selected_recs = st.session_state.meta_artemis_state.get("selected_existing_recommendations", [])
        selected_template_results = st.session_state.meta_artemis_state.get("selected_template_results", [])
        existing_recs = st.session_state.meta_artemis_state.get("existing_recommendations", {})
        generated_recs = st.session_state.meta_artemis_state.get("generated_recommendations")
        
        total_count = len(selected_recs) + len(selected_template_results)
        st.info(f"✅ **{total_count} mixed recommendations** selected for solution creation:")
        
        # Show existing recommendations
        if selected_recs and existing_recs:
            st.markdown("**📁 Existing Recommendations:**")
            all_recs_lookup = {}
            for rec in existing_recs.get("meta_recommendations", []):
                all_recs_lookup[rec["ai_run_id"]] = {**rec, "type": "🧠 Meta-prompting"}
            for rec in existing_recs.get("baseline_recommendations", []):
                all_recs_lookup[rec["ai_run_id"]] = {**rec, "type": "📝 Baseline"}
            for rec in existing_recs.get("other_recommendations", []):
                all_recs_lookup[rec["ai_run_id"]] = {**rec, "type": "❓ Other"}
            
            for i, rec_id in enumerate(selected_recs, 1):
                if rec_id in all_recs_lookup:
                    rec = all_recs_lookup[rec_id]
                    status_color = "🟢" if rec['status'] == 'completed' else "🔴" if rec['status'] == 'failed' else "🟡"
                    st.markdown(f"**{i}.** {rec['type']} | **{rec['spec_name']}** | `{rec['construct_file']}:{rec['construct_lines']}` | {status_color} {rec['status']}")
        
        # Show generated recommendations
        if selected_template_results:
            st.markdown("**🆕 Generated Recommendations:**")
            for i, template_result in enumerate(selected_template_results, len(selected_recs) + 1):
                st.markdown(f"**{i}.** 🧠 Generated | Template: {template_result['template_id']} | Spec: {template_result['spec_id'][:8]}...")
    
    st.markdown("---")
    
    # Step 3B: Optimization Selection & Solution Creation
    st.markdown("### 🎯 Optimization Selection & Solution Creation")
    st.info("📝 **Note**: In Artemis, the data structure is: Project → Optimization → Solutions → Specs. You need to provide an existing optimization ID to contain the solutions.")
    
    # Optimization ID selection with default options
    optimization_id_option = st.radio(
        "Optimization ID Option:",
        ["Use default optimization", "Enter custom optimization ID"],
        help="Select a default optimization or enter your own optimization ID"
    )
    
    optimization_id = None
    
    if optimization_id_option == "Use default optimization":
        # Check if current project is the default project
        current_project_id = st.session_state.meta_artemis_state["project_id"]
        
        if current_project_id == "6c47d53e-7384-44d8-be9d-c186a7af480a":
            default_optimizations = {
                "eef157cf-c8d4-4e7a-a2e5-79cf2f07be88": "Default Optimization 1",
                # Add more default optimizations for this project if needed
            }
            
            optimization_id = st.selectbox(
                "Select default optimization:",
                options=list(default_optimizations.keys()),
                format_func=lambda x: f"{default_optimizations[x]} ({x[:8]}...)",
                help="Select from available default optimizations for this project"
            )
        else:
            st.warning("⚠️ Default optimizations are only available for the default project.")
            st.info("Please switch to 'Enter custom optimization ID' or use the default project.")
    
    else:
        # Custom optimization ID input
        optimization_id = st.text_input(
            "Enter Optimization ID:",
            placeholder="e.g., eef157cf-c8d4-4e7a-a2e5-79cf2f07be88",
            help="Enter the UUID of an existing optimization"
        )
    
    # Show Create Solutions button if we have an optimization ID
    if optimization_id and not step_3_state["solutions_created"]:
        if st.button("🔧 Create Solutions", type="primary", key="create_solutions"):
            with st.spinner("Creating solutions..."):
                created_solutions = create_solutions_from_recommendations(
                    st.session_state.meta_artemis_state,
                    optimization_id
                )
                
                if created_solutions:
                    step_3_state["created_solutions"] = created_solutions
                    step_3_state["solutions_created"] = True
                    step_3_state["execution_ready"] = True
                    step_3_state["selected_optimization_id"] = optimization_id
                    step_3_state["optimization_selected"] = True
                    st.success(f"✅ Created {len(created_solutions)} solutions successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to create solutions")
                    
                    # Show detailed error information if available
                    error_details = st.session_state.meta_artemis_state.get("solution_creation_errors", [])
                    if error_details:
                        with st.expander("🔍 Error Details", expanded=True):
                            st.markdown("**Detailed error information:**")
                            for error in error_details:
                                st.markdown(f"• {error}")
                            
                            st.markdown("**Troubleshooting steps:**")
                            st.markdown("1. Check that recommendations were generated successfully in Step 2")
                            st.markdown("2. Verify the optimization ID is correct and accessible")
                            st.markdown("3. Ensure you have permission to create solutions in this project")
                            st.markdown("4. Check the logs for more detailed error information")
                    else:
                        st.error("No detailed error information available. Check the logs for more details.")
    
    elif step_3_state["solutions_created"]:
        st.success(f"✅ **Solutions Created:** {len(step_3_state['created_solutions'])} solutions ready for execution")
        st.info(f"📋 **Using Optimization:** `{step_3_state['selected_optimization_id'][:8]}...`")
    
    # Step 3D: Show Created Solutions and Execute (only show if solutions are created)
    if step_3_state["solutions_created"]:
        st.markdown("### 📊 Created Solutions")
        
        created_solutions = step_3_state["created_solutions"]
        
        # Show solutions summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Solutions Created", len(created_solutions))
        with col2:
            if step_3_state["selected_optimization_id"]:
                st.metric("Optimization ID", f"{step_3_state['selected_optimization_id'][:8]}...")
            else:
                st.metric("Optimization ID", "None")
        with col3:
            st.metric("Status", "Ready for Execution")
        
        # Show individual solutions with detailed information
        st.markdown("#### 📋 Solution Details")
        
        for i, solution in enumerate(created_solutions, 1):
            with st.expander(f"🔧 Solution {i}: {solution.get('spec_name', f'Solution {i}')} - {solution.get('template_name', 'Unknown Template')}", expanded=i==1):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Solution ID:** `{solution['solution_id']}`")
                    st.write(f"**Spec ID:** `{solution['spec_id']}`")
                    st.write(f"**Construct ID:** `{solution['construct_id']}`")
                    st.write(f"**Optimization ID:** `{solution['optimization_id']}`")
                
                with col2:
                    st.write(f"**Spec Name:** {solution.get('spec_name', 'Unknown')}")
                    st.write(f"**Template:** {solution.get('template_name', 'Unknown')}")
                    st.write(f"**Type:** {solution.get('recommendation_type', 'Unknown')}")
                    st.write(f"**File:** {solution.get('construct_file', 'Unknown')}")
                    st.write(f"**Lines:** {solution.get('construct_lines', 'Unknown')}")
                    st.write(f"**Status:** {solution.get('status', 'Created')}")
                    st.write(f"**Created:** {solution.get('created_at', 'Just now')}")
        
        # Execution configuration
        st.markdown("### ⚙️ Execution Configuration")
        
        project_info = st.session_state.meta_artemis_state["project_info"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Worker:** {st.session_state.meta_artemis_state['custom_worker_name'] or project_info.get('runner_name', 'jing_runner')}")
            st.write(f"**Repetitions:** {st.session_state.meta_artemis_state['evaluation_repetitions']}")
        with col2:
            st.write(f"**Command:** {st.session_state.meta_artemis_state['custom_command'] or project_info.get('perf_command', 'Project default')}")
            st.write(f"**Mode:** {recommendation_mode}")
        
        # Execution button
        st.markdown("### 🚀 Execute Solutions")
        
        if not st.session_state.meta_artemis_state.get("execution_in_progress", False):
            if st.button("🚀 Start Solution Execution", type="primary", key="start_execution"):
                st.session_state.meta_artemis_state["execution_in_progress"] = True
                st.rerun()
        
        # Execute if in progress
        if st.session_state.meta_artemis_state.get("execution_in_progress", False):
            execute_solutions_step3()

async def wait_for_solution_completion(falcon_client, solution_id: str, timeout: int = 600):
    """Wait for solution evaluation to complete"""
    import time
    start_time = time.time()
    last_status = None
    stuck_count = 0
    
    while time.time() - start_time < timeout:
        try:
            solution_details = falcon_client.get_solution(solution_id)
            status_str = str(solution_details.status).lower()
            
            # Only log status changes to avoid spam
            if status_str != last_status:
                logger.info(f"Solution {solution_id} status changed: {last_status} -> {status_str}")
                last_status = status_str
                stuck_count = 0
            else:
                stuck_count += 1
            
            # Check for completion
            if status_str in ['completed', 'failed', 'cancelled', 'success', 'solutionstatusenum.success']:
                logger.info(f"Solution {solution_id} finished with status: {status_str}")
                return solution_details
            
            # Check if solution is stuck in created status for too long
            if status_str == 'solutionstatusenum.created' and stuck_count > 12:  # 2 minutes
                logger.warning(f"Solution {solution_id} stuck in 'created' status for {stuck_count * 10} seconds")
                logger.warning("This might indicate worker is not available or busy")
            
            # If stuck for too long, consider it failed
            if stuck_count > 18:  # 3 minutes in created status
                logger.error(f"Solution {solution_id} stuck in '{status_str}' for too long, treating as failed")
                return solution_details
                
            await asyncio.sleep(10)  # Wait 10 seconds before checking again
        except Exception as e:
            logger.error(f"Error checking solution status: {str(e)}")
            await asyncio.sleep(10)
    
    raise TimeoutError(f"Solution {solution_id} did not complete within {timeout} seconds")

async def execute_solutions_async():
    """Execute solutions that were created in Step 3"""
    try:
        logger.info("🚀 Starting solution execution...")
        state = st.session_state.meta_artemis_state
        
        # Check if we have solutions_for_execution (from execute existing workflow)
        # or step_3_state with created_solutions (from create new workflow)
        if "solutions_for_execution" in state and state["solutions_for_execution"]:
            created_solutions = state["solutions_for_execution"]
            logger.info(f"📊 Found {len(created_solutions)} existing solutions to execute")
        elif "step_3_state" in state and state["step_3_state"].get("created_solutions"):
            step_3_state = state["step_3_state"]
            created_solutions = step_3_state["created_solutions"]
            logger.info(f"📊 Found {len(created_solutions)} created solutions to execute")
        else:
            error_msg = "No solutions found to execute - neither solutions_for_execution nor created_solutions available"
            logger.error(f"❌ {error_msg}")
            st.error(error_msg)
            return None
        
        if not created_solutions:
            error_msg = "No solutions found to execute"
            logger.error(f"❌ {error_msg}")
            st.error(error_msg)
            return None
        
        # Setup Falcon client for execution
        logger.info("🔌 Setting up Falcon client for execution...")
        falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
        thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
        falcon_client = FalconClient(falcon_settings, thanos_settings)
        
        logger.info("🔐 Authenticating with Falcon for execution...")
        falcon_client.authenticate()
        logger.info("✅ Falcon client setup and authentication completed for execution")
        
        # Execution configuration
        worker_name = state["custom_worker_name"] or state["project_info"].get("runner_name", "jing_runner")
        custom_command = state["custom_command"]
        evaluation_repetitions = state["evaluation_repetitions"]
        logger.info(f"⚙️ Execution configuration:")
        logger.info(f"   Worker name: {worker_name}")
        logger.info(f"   Custom command: {custom_command}")
        logger.info(f"   Evaluation repetitions: {evaluation_repetitions}")
        
        # Execute each solution
        execution_results = []
        total_solutions = len(created_solutions)
        successful_executions = 0
        logger.info(f"🔄 Starting execution of {total_solutions} solutions...")
        
        for i, solution in enumerate(created_solutions):
            solution_id = solution["solution_id"]
            solution_name = solution.get("name", f"Solution {i+1}")
            
            logger.info(f"🔄 Processing solution {i+1}/{total_solutions}: {solution_name}")
            logger.info(f"   Solution ID: {solution_id}")
            logger.info(f"   Spec ID: {solution.get('spec_id', 'Unknown')}")
            logger.info(f"   Template: {solution.get('template_name', 'Unknown')}")
            
            try:
                # Update progress
                progress = (i + 1) / total_solutions
                st.session_state.execution_progress = {
                    "current": i + 1,
                    "total": total_solutions,
                    "progress": progress,
                    "current_solution": solution_name,
                    "status": f"Executing solution {i + 1}/{total_solutions}: {solution_name}"
                }
                
                logger.info(f"📊 Updated progress: {progress:.1%} ({i+1}/{total_solutions})")
                
                # Execute the solution with evaluation_repetitions
                logger.info(f"🔄 Executing solution {solution_id} with {evaluation_repetitions} repetitions")
                
                execution_start = time.time()
                
                # Execute solution with repetitions using the evaluation_repetitions parameter
                logger.info(f"🚀 Starting execution with {evaluation_repetitions} repetitions for solution {solution_id}")
                evaluation_response = falcon_client.evaluate_solution(
                    solution_id=UUID(solution_id),
                    evaluation_repetitions=evaluation_repetitions,
                    custom_worker_name=worker_name,
                    custom_command=custom_command,
                    unit_test=True
                )
                
                # Wait for completion with proper monitoring
                logger.info(f"⏳ Waiting for execution with {evaluation_repetitions} repetitions to complete...")
                await wait_for_solution_completion(falcon_client, solution_id, timeout=300)
                
                total_execution_time = time.time() - execution_start
                
                # Get solution results for all repetitions
                logger.info(f"📊 Collecting results for all {evaluation_repetitions} repetitions...")
                solution_details = falcon_client.get_solution(solution_id)
                
                runtime_metrics = {}
                memory_metrics = {}
                cpu_metrics = {}
                
                if solution_details.results and hasattr(solution_details.results, 'values'):
                    logger.info(f"📈 Found {len(solution_details.results.values)} metrics for all repetitions")
                    
                    for metric_name, values in solution_details.results.values.items():
                        logger.info(f"   - {metric_name}: {len(values)} values = {values}")
                        
                        # The API now returns aggregated results from all repetitions
                        if values:
                            if 'runtime' in metric_name.lower() or 'time' in metric_name.lower():
                                runtime_metrics[f"{metric_name}_avg"] = np.mean(values) if values else 0.0
                                runtime_metrics[f"{metric_name}_std"] = np.std(values) if len(values) > 1 else 0.0
                                runtime_metrics[f"{metric_name}_min"] = np.min(values) if values else 0.0
                                runtime_metrics[f"{metric_name}_max"] = np.max(values) if values else 0.0
                                runtime_metrics[f"{metric_name}_all"] = values
                            elif 'memory' in metric_name.lower() or 'mem' in metric_name.lower():
                                memory_metrics[f"{metric_name}_avg"] = np.mean(values) if values else 0.0
                                memory_metrics[f"{metric_name}_std"] = np.std(values) if len(values) > 1 else 0.0
                                memory_metrics[f"{metric_name}_min"] = np.min(values) if values else 0.0
                                memory_metrics[f"{metric_name}_max"] = np.max(values) if values else 0.0
                                memory_metrics[f"{metric_name}_all"] = values
                            elif 'cpu' in metric_name.lower():
                                cpu_metrics[f"{metric_name}_avg"] = np.mean(values) if values else 0.0
                                cpu_metrics[f"{metric_name}_std"] = np.std(values) if len(values) > 1 else 0.0
                                cpu_metrics[f"{metric_name}_min"] = np.min(values) if values else 0.0
                                cpu_metrics[f"{metric_name}_max"] = np.max(values) if values else 0.0
                                cpu_metrics[f"{metric_name}_all"] = values
                
                success = True
                logger.info(f"✅ All {evaluation_repetitions} repetitions completed successfully for solution {solution_id}")
                logger.info(f"   - Runtime metrics: {len(runtime_metrics)}")
                logger.info(f"   - Memory metrics: {len(memory_metrics)}")
                logger.info(f"   - CPU metrics: {len(cpu_metrics)}")
                
                # Get final solution status
                final_solution_details = falcon_client.get_solution(solution_id)
                status_str = str(final_solution_details.status).lower()
                success = success and ('completed' in status_str or 'success' in status_str)
                
                execution_result = {
                    'solution_id': solution_id,
                    'solution_name': solution['name'],
                    'spec_id': solution['spec_id'],
                    'construct_id': solution['construct_id'],
                    'template_id': solution.get('template_id', 'unknown'),
                    'status': str(final_solution_details.status),
                    'success': success,
                    'runtime_metrics': runtime_metrics,
                    'memory_metrics': memory_metrics,
                    'cpu_metrics': cpu_metrics,
                    'execution_time': total_execution_time,
                    'repetitions': evaluation_repetitions
                }
                
                execution_results.append(execution_result)
                
                if success:
                    successful_executions += 1
                
            except Exception as e:
                logger.error(f"Error executing solution {solution_id}: {str(e)}")
                execution_results.append({
                    'solution_id': solution_id,
                    'solution_name': solution['name'],
                    'spec_id': solution['spec_id'],
                    'construct_id': solution['construct_id'],
                    'template_id': solution.get('template_id', 'unknown'),
                    'status': 'failed',
                    'success': False,
                    'error': str(e),
                    'runtime_metrics': {},
                    'memory_metrics': {},
                    'cpu_metrics': {},
                    'execution_time': 0.0,
                    'repetitions': evaluation_repetitions
                })
        
        # Get optimization_id from different sources depending on workflow
        optimization_id = None
        if "step_3_state" in state and state["step_3_state"].get("selected_optimization_id"):
            optimization_id = state["step_3_state"]["selected_optimization_id"]
        elif created_solutions and len(created_solutions) > 0:
            # Try to get optimization_id from the first solution
            optimization_id = created_solutions[0].get("optimization_id", "unknown")
        
        # Compile final results
        results = {
            'project_info': state["project_info"],
            'execution_results': execution_results,
            'created_solutions': created_solutions,
            'optimization_id': optimization_id,
            'summary': {
                'total_solutions': total_solutions,
                'successful_executions': successful_executions,
                'failed_executions': total_solutions - successful_executions,
                'success_rate': (successful_executions / total_solutions) * 100 if total_solutions > 0 else 0
            },
            'execution_config': {
                'worker_name': worker_name,
                'custom_command': custom_command,
                'evaluation_repetitions': evaluation_repetitions,
                                    # Manual repetition removed - using evaluation_repetitions parameter
            }
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error executing solutions: {str(e)}")
        st.error(f"Error executing solutions: {str(e)}")
        return None

def execute_solutions_step3():
    """Execute solutions in Step 3 - using simple non-blocking approach like benchmark Artemis app"""
    logger.info("🚀 Starting execute_solutions_step3 function")
    
    # Check if execution is already completed
    if st.session_state.meta_artemis_state.get("step_3_complete", False):
        logger.info("📊 Step 3 already completed, showing results")
        results = st.session_state.meta_artemis_state.get("evaluation_results")
        if results:
            _display_execution_results(results)
            return
    
    # Get the solutions to execute
    solutions_for_execution = st.session_state.meta_artemis_state.get("solutions_for_execution", [])
    if not solutions_for_execution:
        st.error("❌ No solutions found to execute")
        return
    
    # Get execution configuration
    execution_config = st.session_state.meta_artemis_state.get("execution_config", {})
    
    # Check if execution is not started yet - start it automatically
    if not st.session_state.meta_artemis_state.get("execution_started", False):
        try:
            # Set up Falcon client
            from benchmark_evaluator_meta_artemis import MetaArtemisEvaluator, LLMType
            project_id = st.session_state.meta_artemis_state.get("project_id")
            
            worker = execution_config.get("worker_name", "jing_runner") 
            command = execution_config.get("custom_command", "Project default")
            reps = execution_config.get("repetition_count", 1)
            
            with st.spinner("🚀 Starting execution..."):
                evaluator = MetaArtemisEvaluator(
                    task_name="runtime_performance",
                    meta_prompt_llm_type=LLMType("gpt-4-o"),
                    code_optimization_llm_type=LLMType("gpt-4-o"),
                    project_id=project_id,
                    selected_templates=["standard"],
                    evaluation_repetitions=reps
                )
                
                # Setup clients synchronously (this is safe)
                asyncio.run(evaluator.setup_clients())
                falcon_client = evaluator.falcon_client
                
                # Start execution for each solution (non-blocking like benchmark app)
                execution_started = False
                for solution_info in solutions_for_execution:
                    solution_id = solution_info["solution_id"]
                    
                    try:
                        # Start execution (this just starts it, doesn't wait)
                        response = falcon_client.evaluate_solution(
                            solution_id=UUID(solution_id),
                            evaluation_repetitions=reps,
                            custom_worker_name=worker,
                            custom_command=command if command != "Project default" else None,
                            unit_test=True
                        )
                        
                        logger.info(f"✅ Started execution for solution {solution_id[:8]}...")
                        execution_started = True
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to start execution for solution {solution_id}: {str(e)}")
                        st.error(f"Failed to start execution for solution {solution_id[:8]}...: {str(e)}")
                
                if execution_started:
                    # Mark execution as started
                    st.session_state.meta_artemis_state["execution_started"] = True
                    st.session_state.meta_artemis_state["falcon_client"] = falcon_client
                    st.success("✅ Execution started! Use the 'Check Status' button below to monitor progress.")
                    st.rerun()
                else:
                    st.error("❌ Failed to start any executions")
                    return
                    
        except Exception as e:
            logger.error(f"❌ Exception starting execution: {str(e)}")
            st.error(f"Error starting execution: {str(e)}")
            return
    
    else:
        # Execution is started, show status checking interface
        st.info("🔄 Solution execution in progress...")
        
        if st.button("🔄 Check Status", key="check_execution_status"):
            try:
                falcon_client = st.session_state.meta_artemis_state.get("falcon_client")
                if not falcon_client:
                    st.error("❌ Falcon client not found")
                    return
                
                # Check status of all solutions
                all_complete = True
                results = []
                
                for solution_info in solutions_for_execution:
                    solution_id = solution_info["solution_id"]
                    
                    try:
                        # Get solution details (like benchmark app)
                        solution_detail = falcon_client.get_solution(solution_id)
                        status = str(solution_detail.status).lower()
                        
                        logger.info(f"Solution {solution_id[:8]}: {status}")
                        
                        # Check if complete
                        is_complete = any(term in status for term in ['success', 'complete', 'fail', 'error', 'cancelled'])
                        
                        if not is_complete:
                            all_complete = False
                        
                        # Extract results if available
                        result_data = {
                            'solution_id': solution_id,
                            'solution_name': solution_info.get('name', f'Solution {solution_id[:8]}'),
                            'status': status,
                            'success': 'success' in status or 'complete' in status,
                            'runtime_metrics': {},
                            'memory_metrics': {},
                            'cpu_metrics': {},
                            'execution_time': 0.0,
                            'repetitions': execution_config.get("repetition_count", 1),
                            'template_id': solution_info.get('template_id', 'existing_solution'),
                            'template_name': solution_info.get('template_name', 'Existing Solution'),
                            'spec_id': solution_info.get('spec_id', 'unknown'),
                            'construct_id': solution_info.get('construct_id', 'unknown')
                        }
                        
                        # Extract metrics if available
                        if hasattr(solution_detail, 'results') and solution_detail.results:
                            if hasattr(solution_detail.results, 'values') and solution_detail.results.values:
                                values = solution_detail.results.values
                                
                                for metric_name, metric_values in values.items():
                                    if metric_values:
                                        avg_value = sum(metric_values) / len(metric_values)
                                        
                                        if 'runtime' in metric_name.lower() or 'time' in metric_name.lower():
                                            result_data['runtime_metrics'][f"{metric_name}_avg"] = avg_value
                                            result_data['runtime_metrics'][f"{metric_name}_all"] = metric_values
                                        elif 'memory' in metric_name.lower():
                                            result_data['memory_metrics'][f"{metric_name}_avg"] = avg_value
                                            result_data['memory_metrics'][f"{metric_name}_all"] = metric_values
                                        elif 'cpu' in metric_name.lower():
                                            result_data['cpu_metrics'][f"{metric_name}_avg"] = avg_value
                                            result_data['cpu_metrics'][f"{metric_name}_all"] = metric_values
                        
                        results.append(result_data)
                        
                        # Show individual solution status
                        if 'success' in status or 'complete' in status:
                            st.success(f"✅ Solution {solution_id[:8]}: {status}")
                        elif 'fail' in status or 'error' in status:
                            st.error(f"❌ Solution {solution_id[:8]}: {status}")
                        else:
                            st.info(f"⏳ Solution {solution_id[:8]}: {status}")
                    
                    except Exception as e:
                        logger.error(f"❌ Error checking solution {solution_id}: {str(e)}")
                        st.error(f"Error checking solution {solution_id[:8]}: {str(e)}")
                        results.append({
                            'solution_id': solution_id,
                            'solution_name': solution_info.get('name', f'Solution {solution_id[:8]}'),
                            'status': 'error',
                            'success': False,
                            'error': str(e),
                            'runtime_metrics': {},
                            'memory_metrics': {},
                            'cpu_metrics': {},
                            'execution_time': 0.0,
                            'repetitions': execution_config.get("repetition_count", 1),
                            'template_id': solution_info.get('template_id', 'existing_solution'),
                            'template_name': solution_info.get('template_name', 'Existing Solution'),
                            'spec_id': solution_info.get('spec_id', 'unknown'),
                            'construct_id': solution_info.get('construct_id', 'unknown')
                        })
                
                # If all complete, store results and mark as complete
                if all_complete:
                    # Compile final results
                    project_info = st.session_state.meta_artemis_state.get("project_info", {})
                    successful_executions = sum(1 for r in results if r['success'])
                    
                    final_results = {
                        'project_info': project_info,
                        'execution_results': results,
                        'created_solutions': solutions_for_execution,
                        'optimization_id': solutions_for_execution[0].get("optimization_id", "unknown"),
                        'summary': {
                            'total_solutions': len(results),
                            'successful_executions': successful_executions,
                            'failed_executions': len(results) - successful_executions,
                            'success_rate': (successful_executions / len(results)) * 100 if results else 0
                        },
                        'execution_config': execution_config
                    }
                    
                    # Store results and mark as complete (no saving needed - results are in Artemis)
                    st.session_state.meta_artemis_state["evaluation_results"] = final_results
                    st.session_state.meta_artemis_state["step_3_complete"] = True
                    
                    st.success("✅ All solutions completed!")
                    st.rerun()
                else:
                    st.info("⏳ Some solutions are still running. Click 'Check Status' again in a moment.")
                    
            except Exception as e:
                logger.error(f"❌ Exception checking status: {str(e)}")
                st.error(f"Error checking status: {str(e)}")
        
        # Reset button
        if st.button("🔄 Reset Execution", key="reset_execution"):
            st.session_state.meta_artemis_state["execution_started"] = False
            st.session_state.meta_artemis_state["execution_in_progress"] = False
            st.session_state.meta_artemis_state["step_3_complete"] = False
            st.rerun()

def _display_execution_results(results):
    """Display execution results"""
    st.success("✅ Solution execution completed!")
            
    # Show execution summary
    summary = results["summary"]
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Solutions", summary["total_solutions"])
    with col2:
        st.metric("Successful", summary["successful_executions"])
    with col3:
        st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
    
    # Show detailed results
    st.markdown("#### 📊 Execution Results")
    
    for result in results["execution_results"]:
        status_icon = "✅" if result["success"] else "❌"
        template_id = result.get('template_id', 'existing_solution')
        with st.expander(f"{status_icon} {result['solution_name']} - {result['status']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Solution ID:** `{result['solution_id'][:8]}...`")
                st.write(f"**Template:** {template_id}")
                st.write(f"**Status:** {result['status']}")
                st.write(f"**Execution Time:** {result['execution_time']:.2f}s")
            
            with col2:
                if result["runtime_metrics"]:
                    st.write("**Runtime Metrics:**")
                    for metric, value in result["runtime_metrics"].items():
                        if isinstance(value, (int, float)):
                            st.write(f"  - {metric}: {value:.4f}")
                
                if result["memory_metrics"]:
                    st.write("**Memory Metrics:**")
                    for metric, value in result["memory_metrics"].items():
                        if isinstance(value, (int, float)):
                            st.write(f"  - {metric}: {value:.4f}")
                
                if not result["success"] and "error" in result:
                    st.error(f"Error: {result['error']}")
    
    # Proceed button
    if st.button("➡️ View Results (Step 4)", key="proceed_to_step_4"):
        logger.info("🔄 View Results button clicked - transitioning to Step 4")
        st.session_state.meta_artemis_state["current_results"] = results
        st.session_state.meta_artemis_state["current_step"] = 4
        logger.info(f"✅ Step set to: {st.session_state.meta_artemis_state['current_step']}")
        st.rerun()


async def get_solution_details_from_artemis(solution_ids: List[str]) -> Dict[str, Any]:
    """Get detailed solution information from Artemis"""
    logger.info(f"🔍 Getting solution details from Artemis for {len(solution_ids)} solutions")
    
    try:
        project_id = st.session_state.meta_artemis_state.get("project_id")
        if not project_id:
            raise ValueError("No project_id found in session state")
        
        # Setup evaluator to access Falcon client
        evaluator = MetaArtemisEvaluator(
            task_name="runtime_performance",
            meta_prompt_llm_type=LLMType("gpt-4-o"),
            code_optimization_llm_type=LLMType("gpt-4-o"),
            project_id=project_id,  # Add the missing project_id parameter
            selected_templates=["standard"],
            evaluation_repetitions=3
        )
        
        # Setup the clients
        await evaluator.setup_clients()
        
        # Get solution details
        solution_details = []
        for solution_id in solution_ids:
            logger.info(f"📋 Getting details for solution: {solution_id}")
            
            try:
                solution = evaluator.falcon_client.get_solution(solution_id)
                logger.info(f"✅ Retrieved solution {solution_id}, status: {solution.status}")
                
                # Extract results if available
                results_data = None
                runtime_metrics = {}
                memory_metrics = {}
                
                if hasattr(solution, 'results') and solution.results:
                    logger.info(f"📊 Solution has results with {len(solution.results.values) if hasattr(solution.results, 'values') else 0} metrics")
                    
                    if hasattr(solution.results, 'values') and solution.results.values:
                        # Extract metrics from results
                        cpu_metrics = {}
                        for metric_name, values in solution.results.values.items():
                            if values:  # Only process non-empty value lists
                                avg_value = np.mean(values)
                                metric_data = {
                                    'avg': avg_value,
                                    'min': np.min(values),
                                    'max': np.max(values),
                                    'std': np.std(values),
                                    'count': len(values),
                                    'values': values  # Store raw values for box plots
                                }
                                
                                if 'runtime' in metric_name.lower() or 'time' in metric_name.lower():
                                    runtime_metrics[metric_name] = metric_data
                                elif 'memory' in metric_name.lower() or 'mem' in metric_name.lower():
                                    memory_metrics[metric_name] = metric_data
                                elif 'cpu' in metric_name.lower():
                                    cpu_metrics[metric_name] = metric_data
                    
                    results_data = {
                        'runtime_metrics': runtime_metrics,
                        'memory_metrics': memory_metrics,
                        'cpu_metrics': cpu_metrics,
                        'has_results': bool(runtime_metrics or memory_metrics or cpu_metrics),
                        'total_metrics': len(runtime_metrics) + len(memory_metrics) + len(cpu_metrics)
                    }
                else:
                    logger.info(f"⚠️ Solution {solution_id} has no results")
                    results_data = {
                        'runtime_metrics': {},
                        'memory_metrics': {},
                        'cpu_metrics': {},
                        'has_results': False,
                        'total_metrics': 0
                    }
                
                # Get spec information
                spec_info = []
                if hasattr(solution, 'specs') and solution.specs:
                    for spec in solution.specs:
                        spec_info.append({
                            'spec_id': str(spec.spec_id),
                            'construct_id': str(spec.construct_id) if hasattr(spec, 'construct_id') else 'Unknown'
                        })
                
                solution_details.append({
                    'solution_id': solution_id,
                    'status': str(solution.status),
                    'created_at': solution.created_at.isoformat() if hasattr(solution, 'created_at') and solution.created_at else 'Unknown',
                    'optimization_id': str(solution.optimisation_id) if hasattr(solution, 'optimisation_id') and solution.optimisation_id else 'Unknown',
                    'results': results_data,
                    'specs': spec_info,
                    'has_results': results_data['has_results']
                })
                
                logger.info(f"✅ Processed solution {solution_id}: {results_data['total_metrics']} metrics, has_results: {results_data['has_results']}")
                
            except Exception as e:
                logger.error(f"❌ Error getting solution {solution_id}: {str(e)}")
                solution_details.append({
                    'solution_id': solution_id,
                    'status': 'error',
                    'error': str(e),
                    'results': None,
                    'has_results': False
                })
        
        logger.info(f"✅ Retrieved details for {len(solution_details)} solutions")
        return {
            'solution_details': solution_details,
            'project_id': project_id,
            'total_solutions': len(solution_ids)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting solution details: {str(e)}")
        return {'error': str(e)}

def step_4_results_visualization():
    """Step 4: Results Visualization - Reusing Step 1 solution analysis approach"""
    logger.info("📊 Step 4: Results Visualization function called")
    st.header("📊 Step 4: Results Visualization & Analysis")
    
    # Check if we have executed solutions from Step 3
    evaluation_results = st.session_state.meta_artemis_state.get("evaluation_results")
    solutions_for_execution = st.session_state.meta_artemis_state.get("solutions_for_execution", [])
    
    if not evaluation_results and not solutions_for_execution:
        st.warning("⚠️ No execution results found. Please execute solutions in Step 3 first.")
        return
    
    # Get project ID
    project_id = st.session_state.meta_artemis_state.get("project_id")
    if not project_id:
        st.error("❌ No project ID found. Please go back to Step 1.")
        return
    
    # Solution selection for analysis
    st.markdown("### 🎯 Select Solutions for Analysis")
    
    # Get solution IDs from Step 3 execution
    executed_solution_ids = []
    if evaluation_results and 'execution_results' in evaluation_results:
        executed_solution_ids = [result['solution_id'] for result in evaluation_results['execution_results']]
    elif solutions_for_execution:
        executed_solution_ids = [sol['solution_id'] for sol in solutions_for_execution]
    
    if not executed_solution_ids:
        st.error("❌ No solution IDs found from Step 3 execution.")
        return
    
    st.info(f"📋 Found {len(executed_solution_ids)} solutions from your Step 3 execution")
    
    # Multi-selection for solutions
    selected_solution_ids = st.multiselect(
        "Select solutions to analyze:",
        options=executed_solution_ids,
        format_func=lambda x: f"{x[:12]}...",
        default=executed_solution_ids,  # Select all by default
        help="Choose which solutions you want to analyze in detail"
    )
    
    if not selected_solution_ids:
        st.warning("⚠️ Please select at least one solution to analyze.")
        return
    
    # Load and analyze solutions button
    if st.button("🔍 Load & Analyze Solutions", type="primary"):
        with st.spinner("🔄 Loading solution details from Artemis..."):
            try:
                solution_details = asyncio.run(get_solution_details_from_artemis(selected_solution_ids))
                
                if 'error' in solution_details:
                    st.error(f"❌ Error loading solution details: {solution_details['error']}")
                    return
                
                # Store in session state using Step 1 format
                st.session_state.meta_artemis_state["step4_analysis_results"] = {
                    'type': 'step4_execution_results',
                    'solution_details': solution_details['solution_details'],
                    'selected_solution_ids': selected_solution_ids,
                    'project_id': solution_details['project_id']
                }
                
                st.success("✅ Solution analysis loaded successfully!")
                st.rerun()
                
            except Exception as e:
                logger.error(f"❌ Error loading solution details: {str(e)}")
                st.error(f"❌ Error loading solution details: {str(e)}")
    
    # Display analysis results if available (reuse Step 1 analysis function)
    step4_results = st.session_state.meta_artemis_state.get("step4_analysis_results")
    if step4_results and step4_results.get('type') == 'step4_execution_results':
        st.divider()
        st.markdown("### 📊 Execution Results Analysis")
        
        # Reuse the existing solutions analysis function from Step 1
        display_existing_solutions_analysis(step4_results)

def display_existing_solutions_analysis(results: dict):
    """Display analysis for existing solutions from Artemis"""
    logger.info("📊 Displaying existing solutions analysis")
    
    solution_details = results['solution_details']
    
    # Filter solutions with results
    solutions_with_results = [sol for sol in solution_details if sol.get('has_results', False)]
    solutions_without_results = [sol for sol in solution_details if not sol.get('has_results', False)]
    
    # Summary metrics
    st.markdown("### 📈 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Solutions", len(solution_details))
    with col2:
        st.metric("With Results", len(solutions_with_results))
    with col3:
        st.metric("Without Results", len(solutions_without_results))
    with col4:
        if solutions_with_results:
            total_metrics = sum(sol['results']['total_metrics'] for sol in solutions_with_results)
            st.metric("Total Metrics", total_metrics)
        else:
            st.metric("Total Metrics", "0")
    
    # Performance Analysis
    if solutions_with_results:
        st.markdown("### 🚀 Performance Analysis")
        
        # Collect all performance data
        runtime_data = []
        memory_data = []
        cpu_data = []
        
        # For box plots - collect all individual measurements
        runtime_box_data = []
        memory_box_data = []
        cpu_box_data = []
        
        for sol in solutions_with_results:
            solution_name = sol['solution_id'][:12] + '...'
            
            # Runtime metrics
            if sol['results']['runtime_metrics']:
                for metric_name, metric_data in sol['results']['runtime_metrics'].items():
                    runtime_data.append({
                        'Solution': solution_name,
                        'Metric': metric_name,
                        'Average': metric_data['avg'],
                        'Min': metric_data['min'],
                        'Max': metric_data['max'],
                        'Std Dev': metric_data['std'],
                        'Count': metric_data['count'],
                        'Solution_ID': sol['solution_id']
                    })
                    
                    # Add individual measurements for box plot
                    if 'values' in metric_data:
                        for value in metric_data['values']:
                            runtime_box_data.append({
                                'Solution': solution_name,
                                'Metric': metric_name,
                                'Value': value,
                                'Solution_ID': sol['solution_id']
                            })
            
            # Memory metrics
            if sol['results']['memory_metrics']:
                for metric_name, metric_data in sol['results']['memory_metrics'].items():
                    memory_data.append({
                        'Solution': solution_name,
                        'Metric': metric_name,
                        'Average': metric_data['avg'],
                        'Min': metric_data['min'],
                        'Max': metric_data['max'],
                        'Std Dev': metric_data['std'],
                        'Count': metric_data['count'],
                        'Solution_ID': sol['solution_id']
                    })
                    
                    # Add individual measurements for box plot
                    if 'values' in metric_data:
                        for value in metric_data['values']:
                            memory_box_data.append({
                                'Solution': solution_name,
                                'Metric': metric_name,
                                'Value': value,
                                'Solution_ID': sol['solution_id']
                            })
            
            # CPU metrics
            if sol['results']['cpu_metrics']:
                for metric_name, metric_data in sol['results']['cpu_metrics'].items():
                    cpu_data.append({
                        'Solution': solution_name,
                        'Metric': metric_name,
                        'Average': metric_data['avg'],
                        'Min': metric_data['min'],
                        'Max': metric_data['max'],
                        'Std Dev': metric_data['std'],
                        'Count': metric_data['count'],
                        'Solution_ID': sol['solution_id']
                    })
                    
                    # Add individual measurements for box plot
                    if 'values' in metric_data:
                        for value in metric_data['values']:
                            cpu_box_data.append({
                                'Solution': solution_name,
                                'Metric': metric_name,
                                'Value': value,
                                'Solution_ID': sol['solution_id']
                            })
        
        # Create tabs for different metric types
        metric_tabs = []
        if runtime_data:
            metric_tabs.append("⏱️ Runtime")
        if memory_data:
            metric_tabs.append("💾 Memory")
        if cpu_data:
            metric_tabs.append("🖥️ CPU")
        
        if metric_tabs:
            tabs = st.tabs(metric_tabs)
            
            # Runtime analysis
            if runtime_data:
                tab_idx = metric_tabs.index("⏱️ Runtime")
                with tabs[tab_idx]:
                    st.markdown("#### ⏱️ Runtime Performance")
                    df_runtime = pd.DataFrame(runtime_data)
                    
                    # Show runtime metrics table with individual measurements
                    runtime_table_data = []
                    for sol in solutions_with_results:
                        solution_name = sol['solution_id'][:12] + '...'
                        if sol['results']['runtime_metrics']:
                            for metric_name, metric_data in sol['results']['runtime_metrics'].items():
                                # Add row with all individual measurements
                                measurements_str = ', '.join([f"{v:.4f}" for v in metric_data.get('values', [])])
                                runtime_table_data.append({
                                    'Solution': solution_name,
                                    'Metric': metric_name,
                                    'Individual Measurements (s)': measurements_str,
                                    'Average': f"{metric_data['avg']:.4f}s",
                                    'Min': f"{metric_data['min']:.4f}s",
                                    'Max': f"{metric_data['max']:.4f}s",
                                    'Std Dev': f"{metric_data['std']:.4f}s"
                                })
                    
                    if runtime_table_data:
                        df_runtime_detailed = pd.DataFrame(runtime_table_data)
                        st.dataframe(df_runtime_detailed, use_container_width=True)
                    
                    # Runtime box plot
                    if runtime_box_data:
                        df_runtime_box = pd.DataFrame(runtime_box_data)
                        fig_runtime_box = px.box(
                            df_runtime_box,
                            x='Solution',
                            y='Value',
                            color='Metric',
                            title='Runtime Performance Distribution by Solution',
                            labels={'Value': 'Runtime (s)'},
                            points='all'  # Show all individual points
                        )
                        
                        # Add mean markers
                        means = df_runtime_box.groupby(['Solution', 'Metric'])['Value'].mean().reset_index()
                        for _, row in means.iterrows():
                            fig_runtime_box.add_scatter(
                                x=[row['Solution']], 
                                y=[row['Value']], 
                                mode='markers',
                                marker=dict(color='red', size=8, symbol='diamond'),
                                name=f'Mean ({row["Metric"]})',
                                showlegend=False,
                                hovertemplate=f'Mean: {row["Value"]:.4f}s<extra></extra>'
                            )
                        
                        # Set figure width and layout
                        fig_runtime_box.update_layout(
                            xaxis_tickangle=-45,
                            width=min(800, max(400, len(df_runtime_box['Solution'].unique()) * 120)),
                            height=500
                        )
                        st.plotly_chart(fig_runtime_box, use_container_width=False)
            
            # Memory analysis
            if memory_data:
                tab_idx = metric_tabs.index("💾 Memory")
                with tabs[tab_idx]:
                    st.markdown("#### 💾 Memory Usage")
                    df_memory = pd.DataFrame(memory_data)
                    
                    # Show memory metrics table with individual measurements
                    memory_table_data = []
                    for sol in solutions_with_results:
                        solution_name = sol['solution_id'][:12] + '...'
                        if sol['results']['memory_metrics']:
                            for metric_name, metric_data in sol['results']['memory_metrics'].items():
                                # Add row with all individual measurements
                                measurements_str = ', '.join([f"{v:.2f}" for v in metric_data.get('values', [])])
                                memory_table_data.append({
                                    'Solution': solution_name,
                                    'Metric': metric_name,
                                    'Individual Measurements (MB)': measurements_str,
                                    'Average': f"{metric_data['avg']:.2f}MB",
                                    'Min': f"{metric_data['min']:.2f}MB",
                                    'Max': f"{metric_data['max']:.2f}MB",
                                    'Std Dev': f"{metric_data['std']:.2f}MB"
                                })
                    
                    if memory_table_data:
                        df_memory_detailed = pd.DataFrame(memory_table_data)
                        st.dataframe(df_memory_detailed, use_container_width=True)
                    
                    # Memory box plot
                    if memory_box_data:
                        df_memory_box = pd.DataFrame(memory_box_data)
                        fig_memory_box = px.box(
                            df_memory_box,
                            x='Solution',
                            y='Value',
                            color='Metric',
                            title='Memory Usage Distribution by Solution',
                            labels={'Value': 'Memory (MB)'},
                            points='all'  # Show all individual points
                        )
                        
                        # Add mean markers
                        means = df_memory_box.groupby(['Solution', 'Metric'])['Value'].mean().reset_index()
                        for _, row in means.iterrows():
                            fig_memory_box.add_scatter(
                                x=[row['Solution']], 
                                y=[row['Value']], 
                                mode='markers',
                                marker=dict(color='red', size=8, symbol='diamond'),
                                name=f'Mean ({row["Metric"]})',
                                showlegend=False,
                                hovertemplate=f'Mean: {row["Value"]:.2f}MB<extra></extra>'
                            )
                        
                        # Set figure width and layout
                        fig_memory_box.update_layout(
                            xaxis_tickangle=-45,
                            width=min(800, max(400, len(df_memory_box['Solution'].unique()) * 120)),
                            height=500
                        )
                        st.plotly_chart(fig_memory_box, use_container_width=False)
            
            # CPU analysis
            if cpu_data:
                tab_idx = metric_tabs.index("🖥️ CPU")
                with tabs[tab_idx]:
                    st.markdown("#### 🖥️ CPU Usage")
                    df_cpu = pd.DataFrame(cpu_data)
                    
                    # Show CPU metrics table with individual measurements
                    cpu_table_data = []
                    for sol in solutions_with_results:
                        solution_name = sol['solution_id'][:12] + '...'
                        if sol['results']['cpu_metrics']:
                            for metric_name, metric_data in sol['results']['cpu_metrics'].items():
                                # Add row with all individual measurements
                                measurements_str = ', '.join([f"{v:.2f}" for v in metric_data.get('values', [])])
                                cpu_table_data.append({
                                    'Solution': solution_name,
                                    'Metric': metric_name,
                                    'Individual Measurements (%)': measurements_str,
                                    'Average': f"{metric_data['avg']:.2f}%",
                                    'Min': f"{metric_data['min']:.2f}%",
                                    'Max': f"{metric_data['max']:.2f}%",
                                    'Std Dev': f"{metric_data['std']:.2f}%"
                                })
                    
                    if cpu_table_data:
                        df_cpu_detailed = pd.DataFrame(cpu_table_data)
                        st.dataframe(df_cpu_detailed, use_container_width=True)
                    
                    # CPU box plot
                    if cpu_box_data:
                        df_cpu_box = pd.DataFrame(cpu_box_data)
                        fig_cpu_box = px.box(
                            df_cpu_box,
                            x='Solution',
                            y='Value',
                            color='Metric',
                            title='CPU Usage Distribution by Solution',
                            labels={'Value': 'CPU Usage (%)'},
                            points='all'  # Show all individual points
                        )
                        
                        # Add mean markers
                        means = df_cpu_box.groupby(['Solution', 'Metric'])['Value'].mean().reset_index()
                        for _, row in means.iterrows():
                            fig_cpu_box.add_scatter(
                                x=[row['Solution']], 
                                y=[row['Value']], 
                                mode='markers',
                                marker=dict(color='red', size=8, symbol='diamond'),
                                name=f'Mean ({row["Metric"]})',
                                showlegend=False,
                                hovertemplate=f'Mean: {row["Value"]:.2f}%<extra></extra>'
                            )
                        
                        # Set figure width and layout
                        fig_cpu_box.update_layout(
                            xaxis_tickangle=-45,
                            width=min(800, max(400, len(df_cpu_box['Solution'].unique()) * 120)),
                            height=500
                        )
                        st.plotly_chart(fig_cpu_box, use_container_width=False)
    
    else:
        st.info("ℹ️ No solutions with execution results found. Solutions may not have been executed yet or execution failed.")

def display_single_solution_analysis(analysis_data: dict):
    """Display detailed analysis for a single solution"""
    logger.info("🔬 Displaying single solution analysis")
    
    solution_details = analysis_data['solution_details']
    selected_solution_info = analysis_data['selected_solution_info']
    
    if not solution_details:
        st.error("❌ No solution details available")
        return
    
    solution = solution_details[0]  # Should be only one solution
    
    # Solution Overview
    st.markdown("#### 📋 Solution Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Status", solution['status'])
    with col2:
        st.metric("Has Results", "✅" if solution.get('has_results', False) else "❌")
    with col3:
        if solution.get('has_results', False):
            total_metrics = solution['results']['total_metrics']
            st.metric("Total Metrics", total_metrics)
        else:
            st.metric("Total Metrics", "0")
    
    # Performance Analysis
    if solution.get('has_results', False):
        st.markdown("#### 🚀 Performance Analysis")
        
        results = solution['results']
        
        # Create tabs for different metric types
        metric_tabs = []
        if results['runtime_metrics']:
            metric_tabs.append("⏱️ Runtime")
        if results['memory_metrics']:
            metric_tabs.append("💾 Memory")
        if results.get('cpu_metrics'):
            metric_tabs.append("🖥️ CPU")
        
        if metric_tabs:
            tabs = st.tabs(metric_tabs)
            
            # Runtime metrics
            if results['runtime_metrics']:
                tab_idx = metric_tabs.index("⏱️ Runtime")
                with tabs[tab_idx]:
                    st.markdown("##### ⏱️ Runtime Performance")
                    runtime_data = []
                    runtime_box_data = []
                    
                    for metric_name, metric_data in results['runtime_metrics'].items():
                        # Add row with all individual measurements
                        measurements_str = ', '.join([f"{v:.4f}" for v in metric_data.get('values', [])])
                        runtime_data.append({
                            'Metric': metric_name,
                            'Individual Measurements (s)': measurements_str,
                            'Average': f"{metric_data['avg']:.4f}s",
                            'Min': f"{metric_data['min']:.4f}s",
                            'Max': f"{metric_data['max']:.4f}s",
                            'Std Dev': f"{metric_data['std']:.4f}s"
                        })
                        
                        # Add individual measurements for box plot
                        if 'values' in metric_data:
                            for value in metric_data['values']:
                                runtime_box_data.append({
                                    'Metric': metric_name,
                                    'Value': value
                                })
                    
                    if runtime_data:
                        st.dataframe(pd.DataFrame(runtime_data), use_container_width=True)
                        
                        # Runtime box plot
                        if runtime_box_data:
                            df_runtime_box = pd.DataFrame(runtime_box_data)
                            fig_runtime_box = px.box(
                                df_runtime_box,
                                x='Metric',
                                y='Value',
                                title='Runtime Performance Distribution by Metric',
                                labels={'Value': 'Runtime (s)'},
                                points='all'  # Show all individual points
                            )
                            
                            # Add mean markers
                            means = df_runtime_box.groupby('Metric')['Value'].mean().reset_index()
                            for _, row in means.iterrows():
                                fig_runtime_box.add_scatter(
                                    x=[row['Metric']], 
                                    y=[row['Value']], 
                                    mode='markers',
                                    marker=dict(color='red', size=8, symbol='diamond'),
                                    name='Mean',
                                    showlegend=False,
                                    hovertemplate=f'Mean: {row["Value"]:.4f}s<extra></extra>'
                                )
                            
                            # Set figure width and layout
                            fig_runtime_box.update_layout(
                                width=min(800, max(400, len(df_runtime_box['Metric'].unique()) * 120)),
                                height=500
                            )
                            st.plotly_chart(fig_runtime_box, use_container_width=False)
            
            # Memory metrics
            if results['memory_metrics']:
                tab_idx = metric_tabs.index("💾 Memory")
                with tabs[tab_idx]:
                    st.markdown("##### 💾 Memory Usage")
                    memory_data = []
                    memory_box_data = []
                    
                    for metric_name, metric_data in results['memory_metrics'].items():
                        # Add row with all individual measurements
                        measurements_str = ', '.join([f"{v:.2f}" for v in metric_data.get('values', [])])
                        memory_data.append({
                            'Metric': metric_name,
                            'Individual Measurements (MB)': measurements_str,
                            'Average': f"{metric_data['avg']:.2f}MB",
                            'Min': f"{metric_data['min']:.2f}MB",
                            'Max': f"{metric_data['max']:.2f}MB",
                            'Std Dev': f"{metric_data['std']:.2f}MB"
                        })
                        
                        # Add individual measurements for box plot
                        if 'values' in metric_data:
                            for value in metric_data['values']:
                                memory_box_data.append({
                                    'Metric': metric_name,
                                    'Value': value
                                })
                    
                    if memory_data:
                        st.dataframe(pd.DataFrame(memory_data), use_container_width=True)
                        
                        # Memory box plot
                        if memory_box_data:
                            df_memory_box = pd.DataFrame(memory_box_data)
                            fig_memory_box = px.box(
                                df_memory_box,
                                x='Metric',
                                y='Value',
                                title='Memory Usage Distribution by Metric',
                                labels={'Value': 'Memory (MB)'},
                                points='all'  # Show all individual points
                            )
                            
                            # Add mean markers
                            means = df_memory_box.groupby('Metric')['Value'].mean().reset_index()
                            for _, row in means.iterrows():
                                fig_memory_box.add_scatter(
                                    x=[row['Metric']], 
                                    y=[row['Value']], 
                                    mode='markers',
                                    marker=dict(color='red', size=8, symbol='diamond'),
                                    name='Mean',
                                    showlegend=False,
                                    hovertemplate=f'Mean: {row["Value"]:.2f}MB<extra></extra>'
                                )
                            
                            # Set figure width and layout
                            fig_memory_box.update_layout(
                                width=min(800, max(400, len(df_memory_box['Metric'].unique()) * 120)),
                                height=500
                            )
                            st.plotly_chart(fig_memory_box, use_container_width=False)
            
            # CPU metrics
            if results.get('cpu_metrics'):
                tab_idx = metric_tabs.index("🖥️ CPU")
                with tabs[tab_idx]:
                    st.markdown("##### 🖥️ CPU Usage")
                    cpu_data = []
                    cpu_box_data = []
                    
                    for metric_name, metric_data in results['cpu_metrics'].items():
                        # Add row with all individual measurements
                        measurements_str = ', '.join([f"{v:.2f}" for v in metric_data.get('values', [])])
                        cpu_data.append({
                            'Metric': metric_name,
                            'Individual Measurements (%)': measurements_str,
                            'Average': f"{metric_data['avg']:.2f}%",
                            'Min': f"{metric_data['min']:.2f}%",
                            'Max': f"{metric_data['max']:.2f}%",
                            'Std Dev': f"{metric_data['std']:.2f}%"
                        })
                        
                        # Add individual measurements for box plot
                        if 'values' in metric_data:
                            for value in metric_data['values']:
                                cpu_box_data.append({
                                    'Metric': metric_name,
                                    'Value': value
                                })
                    
                    if cpu_data:
                        st.dataframe(pd.DataFrame(cpu_data), use_container_width=True)
                        
                        # CPU box plot
                        if cpu_box_data:
                            df_cpu_box = pd.DataFrame(cpu_box_data)
                            fig_cpu_box = px.box(
                                df_cpu_box,
                                x='Metric',
                                y='Value',
                                title='CPU Usage Distribution by Metric',
                                labels={'Value': 'CPU Usage (%)'},
                                points='all'  # Show all individual points
                            )
                            
                            # Add mean markers
                            means = df_cpu_box.groupby('Metric')['Value'].mean().reset_index()
                            for _, row in means.iterrows():
                                fig_cpu_box.add_scatter(
                                    x=[row['Metric']], 
                                    y=[row['Value']], 
                                    mode='markers',
                                    marker=dict(color='red', size=8, symbol='diamond'),
                                    name='Mean',
                                    showlegend=False,
                                    hovertemplate=f'Mean: {row["Value"]:.2f}%<extra></extra>'
                                )
                            
                            # Set figure width and layout
                            fig_cpu_box.update_layout(
                                width=min(800, max(400, len(df_cpu_box['Metric'].unique()) * 120)),
                                height=500
                            )
                            st.plotly_chart(fig_cpu_box, use_container_width=False)
    
    # Clear analysis button
    if st.button("🗑️ Clear Analysis", key="clear_analysis_btn"):
        if "selected_solution_analysis" in st.session_state.meta_artemis_state:
            del st.session_state.meta_artemis_state["selected_solution_analysis"]
        st.rerun()


def main():
    """Display execution performance metrics"""
    st.markdown("### 📊 Performance Metrics")
    
    if "execution_results" not in results:
        st.info("No execution results available")
        return
    
    execution_results = results["execution_results"]
    
    # Check if any results use manual repetition
    has_manual_repetition = any(
        result.get("manual_repetition", False) if isinstance(result, dict) else getattr(result, "manual_repetition", False)
        for result in execution_results
    )
    
    if has_manual_repetition:
        st.info("🔄 **Manual Repetition Mode Detected**: Some solutions were executed multiple times manually. Individual measurements and statistics are available below.")
    
    # Collect performance data
    performance_data = []
    runtime_data = []
    memory_data = []
    individual_measurements = []  # For manual repetition data
    
    for result in execution_results:
        # Handle both dict and object formats
        solution_name = result.get("solution_name", "Unknown") if isinstance(result, dict) else getattr(result, "solution_name", "Unknown")
        template_id = result.get("template_id", "Unknown") if isinstance(result, dict) else getattr(result, "template_id", "Unknown")
        manual_repetition = result.get("manual_repetition", False) if isinstance(result, dict) else getattr(result, "manual_repetition", False)
        
        # Runtime metrics
        runtime_metrics = result.get("runtime_metrics", {}) if isinstance(result, dict) else getattr(result, "runtime_metrics", {})
        for metric_name, value in runtime_metrics.items():
            # Skip _all metrics for summary (they contain individual measurements)
            if metric_name.endswith("_all"):
                # Store individual measurements for manual repetition analysis
                if manual_repetition and isinstance(value, list):
                    for i, measurement in enumerate(value):
                        individual_measurements.append({
                            "Solution": solution_name,
                            "Template": template_id,
                            "Metric Type": "Runtime",
                            "Metric": metric_name.replace("_all", ""),
                            "Repetition": i + 1,
                            "Value": measurement
                        })
                continue
                
            performance_data.append({
                "Solution": solution_name,
                "Template": template_id,
                "Metric Type": "Runtime",
                "Metric": metric_name,
                "Value": value
            })
            runtime_data.append({
                "Solution": solution_name,
                "Template": template_id,
                "Metric": metric_name,
                "Value": value
            })
        
        # Memory metrics
        memory_metrics = result.get("memory_metrics", {}) if isinstance(result, dict) else getattr(result, "memory_metrics", {})
        for metric_name, value in memory_metrics.items():
            # Skip _all metrics for summary
            if metric_name.endswith("_all"):
                # Store individual measurements for manual repetition analysis
                if manual_repetition and isinstance(value, list):
                    for i, measurement in enumerate(value):
                        individual_measurements.append({
                            "Solution": solution_name,
                            "Template": template_id,
                            "Metric Type": "Memory",
                            "Metric": metric_name.replace("_all", ""),
                            "Repetition": i + 1,
                            "Value": measurement
                        })
                continue
                
            performance_data.append({
                "Solution": solution_name,
                "Template": template_id,
                "Metric Type": "Memory",
                "Metric": metric_name,
                "Value": value
            })
            memory_data.append({
                "Solution": solution_name,
                "Template": template_id,
                "Metric": metric_name,
                "Value": value
            })
    
    if performance_data:
        # Runtime metrics visualization
        if runtime_data:
            st.markdown("#### ⏱️ Runtime Metrics")
            runtime_df = pd.DataFrame(runtime_data)
            
            if len(runtime_df) > 0:
                fig = px.bar(
                    runtime_df, 
                    x="Solution", 
                    y="Value", 
                    color="Template",
                    facet_col="Metric",
                    title="Runtime Performance by Solution and Template"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Memory metrics visualization
        if memory_data:
            st.markdown("#### 💾 Memory Metrics")
            memory_df = pd.DataFrame(memory_data)
            
            if len(memory_df) > 0:
                fig = px.bar(
                    memory_df, 
                    x="Solution", 
                    y="Value", 
                    color="Template",
                    facet_col="Metric",
                    title="Memory Usage by Solution and Template"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary table
        st.markdown("#### 📋 Performance Summary")
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
        
        # Execution time comparison
        st.markdown("#### ⏱️ Execution Time Comparison")
        exec_time_data = []
        for result in execution_results:
            solution_name = result.get("solution_name", "Unknown") if isinstance(result, dict) else getattr(result, "solution_name", "Unknown")
            template_id = result.get("template_id", "Unknown") if isinstance(result, dict) else getattr(result, "template_id", "Unknown")
            execution_time = result.get("execution_time", 0) if isinstance(result, dict) else getattr(result, "execution_time", 0)
            repetitions = result.get("repetitions", 1) if isinstance(result, dict) else getattr(result, "repetitions", 1)
            
            exec_time_data.append({
                "Solution": solution_name,
                "Template": template_id,
                "Execution Time": execution_time,
                "Repetitions": repetitions
            })
        
        if exec_time_data:
            exec_df = pd.DataFrame(exec_time_data)
            fig = px.bar(
                exec_df,
                x="Solution",
                y="Execution Time",
                color="Template",
                title="Solution Execution Time Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Manual repetition individual measurements
        if individual_measurements:
            st.markdown("#### 🔄 Manual Repetition Individual Measurements")
            st.info("📊 These charts show individual measurements from each execution repetition, allowing you to see performance variability.")
            
            individual_df = pd.DataFrame(individual_measurements)
            
            # Group by metric type for better visualization
            runtime_individual = individual_df[individual_df["Metric Type"] == "Runtime"]
            memory_individual = individual_df[individual_df["Metric Type"] == "Memory"]
            
            if not runtime_individual.empty:
                st.markdown("**Runtime Measurements by Repetition:**")
                fig = px.box(
                    runtime_individual,
                    x="Solution",
                    y="Value",
                    color="Template",
                    facet_col="Metric",
                    title="Runtime Performance Variability (Individual Measurements)",
                    points="all"  # Show all individual points
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Also show as scatter plot to see trends
                fig_scatter = px.scatter(
                    runtime_individual,
                    x="Repetition",
                    y="Value",
                    color="Solution",
                    facet_col="Metric",
                    title="Runtime Performance by Repetition Number",
                    hover_data=["Template"]
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            if not memory_individual.empty:
                st.markdown("**Memory Measurements by Repetition:**")
                fig = px.box(
                    memory_individual,
                    x="Solution",
                    y="Value",
                    color="Template",
                    facet_col="Metric",
                    title="Memory Usage Variability (Individual Measurements)",
                    points="all"  # Show all individual points
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Also show as scatter plot to see trends
                fig_scatter = px.scatter(
                    memory_individual,
                    x="Repetition",
                    y="Value",
                    color="Solution",
                    facet_col="Metric",
                    title="Memory Usage by Repetition Number",
                    hover_data=["Template"]
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Statistical summary for manual repetitions
            st.markdown("#### 📈 Statistical Summary of Individual Measurements")
            if not individual_df.empty:
                stats_summary = individual_df.groupby(["Solution", "Template", "Metric Type", "Metric"])["Value"].agg([
                    "count", "mean", "std", "min", "max"
                ]).round(6)
                stats_summary.columns = ["Count", "Mean", "Std Dev", "Min", "Max"]
                st.dataframe(stats_summary, use_container_width=True)
    else:
        st.info("No performance metrics available")



async def retrieve_live_results_from_artemis(saved_results: dict) -> dict:
    """Retrieve live results from Artemis using saved solution IDs"""
    logger.info("🔄 Starting to retrieve live results from Artemis")
    
    try:
        # Extract project info and solution IDs from saved results
        project_info = saved_results.get("project_info", {})
        project_id = st.session_state.meta_artemis_state.get("project_id")
        
        if not project_id:
            raise ValueError("No project ID available")
        
        # Get solution IDs from saved results
        solution_ids = []
        if "execution_results" in saved_results:
            solution_ids = [result.get("solution_id") for result in saved_results["execution_results"] if result.get("solution_id")]
        elif "created_solutions" in saved_results:
            solution_ids = [solution.get("solution_id") for solution in saved_results["created_solutions"] if solution.get("solution_id")]
        
        logger.info(f"📋 Found {len(solution_ids)} solution IDs to retrieve: {solution_ids}")
        
        if not solution_ids:
            raise ValueError("No solution IDs found in saved results")
        
        # Setup Artemis client
        logger.info("🤖 Setting up Artemis client")
        evaluator = MetaArtemisEvaluator(
            task_name="runtime_performance",
            meta_prompt_llm_type=LLMType("gpt-4-o"),
            code_optimization_llm_type=LLMType("gpt-4-o"),
            project_id=project_id
        )
        await evaluator.setup_clients()
        
        # Retrieve live results
        live_execution_results = []
        live_created_solutions = []
        
        for solution_id in solution_ids:
            try:
                logger.info(f"🔍 Retrieving solution {solution_id}")
                
                # Get live solution details
                solution_details = evaluator.falcon_client.get_solution(solution_id)
                
                # Extract metrics from live results
                runtime_metrics = {}
                memory_metrics = {}
                
                if solution_details.results and hasattr(solution_details.results, 'values'):
                    for metric_name, values in solution_details.results.values.items():
                        if 'runtime' in metric_name.lower() or 'time' in metric_name.lower():
                            runtime_metrics[metric_name] = np.mean(values) if values else 0.0
                        elif 'memory' in metric_name.lower() or 'mem' in metric_name.lower():
                            memory_metrics[metric_name] = np.mean(values) if values else 0.0
                
                # Find corresponding saved solution info
                saved_solution_info = None
                if "created_solutions" in saved_results:
                    for saved_sol in saved_results["created_solutions"]:
                        if saved_sol.get("solution_id") == solution_id:
                            saved_solution_info = saved_sol
                            break
                
                # Create live execution result
                live_execution_result = {
                    "solution_id": solution_id,
                    "solution_name": saved_solution_info.get("name", "Unknown") if saved_solution_info else "Unknown",
                    "spec_id": str(solution_details.specs[0].spec_id) if solution_details.specs else "Unknown",
                    "construct_id": str(solution_details.specs[0].construct_id) if solution_details.specs else "Unknown",
                    "template_id": saved_solution_info.get("template_id", "Unknown") if saved_solution_info else "Unknown",
                    "status": str(solution_details.status),
                    "success": str(solution_details.status).lower() in ['completed', 'success'],
                    "runtime_metrics": runtime_metrics,
                    "memory_metrics": memory_metrics,
                    "execution_time": 0.0,  # Not available from live data
                    "repetitions": 1,  # Default
                    "last_updated": datetime.now().isoformat()
                }
                
                live_execution_results.append(live_execution_result)
                
                # Create live solution info
                live_solution_info = {
                    "solution_id": solution_id,
                    "spec_id": str(solution_details.specs[0].spec_id) if solution_details.specs else "Unknown",
                    "construct_id": str(solution_details.specs[0].construct_id) if solution_details.specs else "Unknown",
                    "template_id": saved_solution_info.get("template_id", "Unknown") if saved_solution_info else "Unknown",
                    "template_name": saved_solution_info.get("template_name", "Unknown") if saved_solution_info else "Unknown",
                    "spec_name": saved_solution_info.get("spec_name", "Unknown") if saved_solution_info else "Unknown",
                    "construct_file": saved_solution_info.get("construct_file", "Unknown") if saved_solution_info else "Unknown",
                    "construct_lines": saved_solution_info.get("construct_lines", "Unknown") if saved_solution_info else "Unknown",
                    "name": saved_solution_info.get("name", "Unknown") if saved_solution_info else "Unknown",
                    "status": str(solution_details.status),
                    "created_at": str(solution_details.created_at) if hasattr(solution_details, 'created_at') else "Unknown",
                    "optimization_id": str(solution_details.optimisation_id) if solution_details.optimisation_id else "Unknown",
                    "recommendation_type": saved_solution_info.get("recommendation_type", "Unknown") if saved_solution_info else "Unknown",
                    "ai_run_id": saved_solution_info.get("ai_run_id", "Unknown") if saved_solution_info else "Unknown",
                    "live_data": True
                }
                
                live_created_solutions.append(live_solution_info)
                
                logger.info(f"✅ Successfully retrieved live data for solution {solution_id}")
                
            except Exception as e:
                logger.error(f"❌ Error retrieving solution {solution_id}: {str(e)}")
                # Keep the saved data for this solution
                continue
        
        # Create live results structure
        live_results = {
            "project_info": project_info,
            "execution_results": live_execution_results,
            "created_solutions": live_created_solutions,
            "optimization_id": saved_results.get("optimization_id", "Unknown"),
            "summary": {
                "total_solutions": len(live_execution_results),
                "successful_executions": sum(1 for r in live_execution_results if r.get("success", False)),
                "failed_executions": sum(1 for r in live_execution_results if not r.get("success", False)),
                "success_rate": (sum(1 for r in live_execution_results if r.get("success", False)) / len(live_execution_results) * 100) if live_execution_results else 0
            },
            "execution_config": saved_results.get("execution_config", {}),
            "live_data": True,
            "retrieved_at": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Successfully retrieved live results for {len(live_execution_results)} solutions")
        return live_results
        
    except Exception as e:
        logger.error(f"❌ Error retrieving live results from Artemis: {str(e)}")
        raise

def main():
    st.set_page_config(
        page_title="Meta-Prompting + Artemis Evaluator",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Meta-Prompting + Artemis Code Optimization Evaluator")
    st.markdown("Combines meta-prompting framework with Artemis execution to measure real performance impact")
    
    # Initialize session state
    initialize_session_state()
    
    # Main workflow - conditional rendering like benchmark Artemis app
    current_step = st.session_state.meta_artemis_state["current_step"]
    
    # Progress indicator
    progress_steps = ["📊 Project Analysis", "🎯 Recommendation Selection", "🚀 Solution Execution", "📈 Results Visualization"]
    
    # Display progress bar
    progress_cols = st.columns(len(progress_steps))
    for i, step_name in enumerate(progress_steps):
        with progress_cols[i]:
            if i + 1 < current_step:
                st.success(f"✅ {step_name}")
            elif i + 1 == current_step:
                st.info(f"🔄 {step_name}")
            else:
                st.write(f"⏳ {step_name}")
    
    st.divider()
    
    # Step 1: Project Analysis
    if current_step == 1:
        step_1_project_analysis()
    
    # Step 2: Workflow-specific actions
    elif current_step == 2:
        workflow_choice = st.session_state.meta_artemis_state.get("workflow_choice")
        
        # Back button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("⬅️ Back to Step 1", key="back_to_step_1"):
                st.session_state.meta_artemis_state["current_step"] = 1
                st.rerun()
        
        if workflow_choice == "create_new":
            step_2_recommendation_selection()
        elif workflow_choice == "execute_existing":
            step_2_workflow_handler()
        elif workflow_choice == "view_existing":
            step_2_workflow_handler()
        else:
            # Skip to step 4 for other workflows
            st.session_state.meta_artemis_state["current_step"] = 4
            st.rerun()
    
    # Step 3: Solution Execution (for create_new and execute_existing workflows)
    elif current_step == 3:
        workflow_choice = st.session_state.meta_artemis_state.get("workflow_choice")
        came_from_step1 = st.session_state.meta_artemis_state.get("came_from_step1", False)
        
        if workflow_choice in ["create_new", "execute_existing"]:
            # Only show back button if we didn't come directly from Step 1
            if not came_from_step1:
                # Back button
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("⬅️ Back to Step 2", key="back_to_step_2"):
                        st.session_state.meta_artemis_state["current_step"] = 2
                        st.rerun()
            
            step_3_solution_execution()
        else:
            # Skip to step 4 for view_existing workflow
            st.session_state.meta_artemis_state["current_step"] = 4
            st.rerun()
    
    # Step 4: Results Visualization
    elif current_step == 4:
        workflow_choice = st.session_state.meta_artemis_state.get("workflow_choice")
        
        # Back button - different behavior based on workflow
        col1, col2 = st.columns([1, 4])
        with col1:
            if workflow_choice in ["create_new", "execute_existing"]:
                if st.button("⬅️ Back to Step 3", key="back_to_step_3"):
                    st.session_state.meta_artemis_state["current_step"] = 3
                    st.rerun()
            else:
                if st.button("⬅️ Back to Step 1", key="back_to_step_1_from_4"):
                    st.session_state.meta_artemis_state["current_step"] = 1
                    st.rerun()
        
        step_4_results_visualization()
    
    # Handle saved results case
    elif st.session_state.meta_artemis_state["saved_results"]:
        # Back button for saved results
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("⬅️ Back to Step 1", key="back_to_step_1_from_results"):
                st.session_state.meta_artemis_state["current_step"] = 1
                st.session_state.meta_artemis_state["saved_results"] = None
                st.rerun()
        
        step_4_results_visualization()
    
    else:
        logger.error(f"❌ Invalid step state: {current_step}")
        st.error("Invalid step state")
    


if __name__ == "__main__":
    main() 