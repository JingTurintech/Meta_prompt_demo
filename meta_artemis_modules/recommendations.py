"""
Recommendation generation and management functions for the Meta Artemis application.
Handles meta-prompting, baseline recommendations, and progress tracking.
"""

import streamlit as st
import asyncio
from loguru import logger
from typing import Dict, Any, List, Optional, Callable
from benchmark_evaluator_meta_artemis import (
    MetaArtemisEvaluator, LLMType, RecommendationResult
)
from meta_artemis_modules.shared_templates import META_PROMPT_TEMPLATES, DEFAULT_PROJECT_OPTIMISATION_IDS
from .utils import get_session_state
from datetime import datetime
import json
import pandas as pd


async def generate_recommendations_async(progress_callback: Optional[Callable] = None):
    """Generate recommendations asynchronously"""
    try:
        state = get_session_state()
        
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
            custom_templates=state.get("custom_templates", {}),  # Pass custom templates
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
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create containers for dynamic content
    meta_prompts_container = st.container()
    recommendations_container = st.container()
    
    # Track progress state
    progress_state = {
        "meta_prompts": {},
        "recommendations": [],
        "current_spec": None,
        "current_template": None
    }
    
    def progress_callback(progress_data):
        """Enhanced progress callback that handles different types of progress updates"""
        nonlocal progress_state
        
        if isinstance(progress_data, dict):
            # Handle structured progress data
            status = progress_data.get("status", "")
            message = progress_data.get("message", "")
            progress = progress_data.get("progress", None)
            
            # Update progress bar and status
            if progress is not None:
                progress_bar.progress(progress)
            if message:
                status_text.text(message)
            
            # Handle different status types
            if status == "meta_prompt_ready":
                template_id = progress_data.get("template_id")
                filled_meta_prompt = progress_data.get("filled_meta_prompt")
                if template_id and filled_meta_prompt:
                    progress_state["meta_prompts"][template_id] = {
                        "name": META_PROMPT_TEMPLATES[template_id]["name"],
                        "content": filled_meta_prompt
                    }
                    display_meta_prompts_progress(meta_prompts_container, progress_state["meta_prompts"])
            
            elif status == "prompt_ready":
                template_id = progress_data.get("template_id")
                generated_prompt = progress_data.get("generated_prompt")
                if template_id and generated_prompt:
                    if template_id in progress_state["meta_prompts"]:
                        progress_state["meta_prompts"][template_id]["generated_prompt"] = generated_prompt
                    display_meta_prompts_progress(meta_prompts_container, progress_state["meta_prompts"])
            
            elif status == "processing_spec":
                # Extract spec and template info from message
                progress_state["current_spec"] = progress_data.get("spec_name", "")
                progress_state["current_template"] = progress_data.get("template_name", "")
            
            elif status == "recommendation_complete":
                # Add completed recommendation
                recommendation_data = progress_data.get("recommendation_data")
                if recommendation_data:
                    progress_state["recommendations"].append(recommendation_data)
                    display_recommendations_progress(recommendations_container, progress_state["recommendations"])
        
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
    
    try:
        # Generate recommendations with enhanced progress tracking
        results = asyncio.run(generate_recommendations_async(progress_callback))
        
        if results:
            # Store results
            from .utils import update_session_state
            update_session_state({
                "generated_recommendations": results,
                "recommendations_generated": True
            })
            
            st.success("🎉 Recommendations generated successfully!")
            st.info("🎯 Now you can select which recommendations to use for solution creation.")
            
            # Instead of going to step 3, reload the recommendation selection interface
            st.rerun()
        else:
            status_text.text("❌ Failed to generate recommendations")
            st.error("Failed to generate recommendations. Check logs for details.")
            
    except Exception as e:
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Error generating recommendations: {str(e)}")
        progress_bar.progress(0)


def display_meta_prompts_progress(container, meta_prompts):
    """Display generated meta-prompts in real-time"""
    with container.container():
        if meta_prompts:
            st.markdown("### 🧠 Generated Meta-Prompts")
            for template_id, prompt_data in meta_prompts.items():
                with st.expander(f"📝 {prompt_data['name']}", expanded=False):
                    st.markdown("**Meta-Prompt Template:**")
                    st.code(prompt_data["content"], language="text")
                    
                    if "generated_prompt" in prompt_data:
                        st.markdown("**Generated Optimization Prompt:**")
                        st.code(prompt_data["generated_prompt"], language="text")
                    else:
                        st.info("⏳ Generating optimization prompt...")


def display_recommendations_progress(container, recommendations):
    """Display generated recommendations in real-time"""
    # Clear the container first to avoid duplicates
    container.empty()
    
    with container:
        if recommendations:
            st.markdown("### 🎯 Generated Recommendations")
            
            # Show summary
            total_recs = len(recommendations)
            successful_recs = sum(1 for rec in recommendations if rec.get("success", False))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Generated", total_recs)
            with col2:
                st.metric("Successful", successful_recs)
            with col3:
                success_rate = (successful_recs / total_recs) * 100 if total_recs > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")


def get_recommendation_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Get summary statistics from recommendation results"""
    if not results or not results.get("spec_results"):
        return {"total": 0, "successful": 0, "failed": 0, "by_template": {}}
    
    summary = {
        "total": 0,
        "successful": 0, 
        "failed": 0,
        "by_template": {}
    }
    
    for spec_result in results["spec_results"]:
        for template_id, template_result in spec_result["template_results"].items():
            recommendation = template_result["recommendation"]
            summary["total"] += 1
            
            if template_id not in summary["by_template"]:
                summary["by_template"][template_id] = {"total": 0, "successful": 0, "failed": 0}
            
            summary["by_template"][template_id]["total"] += 1
            
            if recommendation.recommendation_success:
                summary["successful"] += 1
                summary["by_template"][template_id]["successful"] += 1
            else:
                summary["failed"] += 1
                summary["by_template"][template_id]["failed"] += 1
    
    return summary 


async def execute_batch_recommendations_async(evaluator: MetaArtemisEvaluator, batch_config: dict, project_specs: list) -> List[Dict[str, Any]]:
    """Execute batch recommendation creation asynchronously"""
    try:
        # Calculate total operations
        total_operations = len(batch_config["selected_constructs"]) * len(batch_config.get("selected_templates", []))
        if batch_config.get("include_baseline"):
            total_operations += len(batch_config["selected_constructs"])
        
        batch_results = []
        current_operation = 0
        
        # Process each construct
        for construct_id in batch_config["selected_constructs"]:
            # Get specs for this construct
            construct_specs = [s for s in project_specs if s["construct_id"] == construct_id]
            if not construct_specs:
                continue
            
            representative_spec = construct_specs[0]
            
            # Process baseline first if enabled
            if batch_config.get("include_baseline"):
                current_operation += 1
                
                # Send simple progress update
                if evaluator.progress_callback:
                    progress = 0.4 + (current_operation / total_operations) * 0.5
                    evaluator.progress_callback({
                        "progress": progress,
                        "message": f"Creating baseline recommendation for {representative_spec['name']} ({current_operation}/{total_operations})"
                    })
                
                try:
                    # Generate baseline recommendation
                    baseline_result = await evaluator.execute_baseline_recommendation_for_spec(
                        representative_spec
                    )
                    
                    result = {
                        "construct_id": construct_id,
                        "template_id": "baseline",
                        "spec_name": representative_spec["name"],
                        "recommendation": baseline_result,
                        "success": baseline_result.recommendation_success,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    batch_results.append(result)
                
                except Exception as e:
                    logger.error(f"Error processing {construct_id} with baseline: {str(e)}")
                    result = {
                        "construct_id": construct_id,
                        "template_id": "baseline",
                        "spec_name": representative_spec["name"],
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    batch_results.append(result)
            
            # Process each template if any are selected
            for template_id in batch_config.get("selected_templates", []):
                current_operation += 1
                template_name = META_PROMPT_TEMPLATES.get(template_id, {}).get("name", template_id)
                
                # Send simple progress update
                if evaluator.progress_callback:
                    progress = 0.4 + (current_operation / total_operations) * 0.5
                    evaluator.progress_callback({
                        "progress": progress,
                        "message": f"Creating recommendation for {representative_spec['name']} with {template_name} ({current_operation}/{total_operations})"
                    })
                
                try:
                    # Generate recommendation
                    recommendation_result = await evaluator.execute_recommendation_for_spec(
                        representative_spec, template_id
                    )
                    
                    result = {
                        "construct_id": construct_id,
                        "template_id": template_id,
                        "spec_name": representative_spec["name"],
                        "recommendation": recommendation_result,
                        "success": recommendation_result.recommendation_success,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing {construct_id} with {template_id}: {str(e)}")
                    result = {
                        "construct_id": construct_id,
                        "template_id": template_id,
                        "spec_name": representative_spec["name"],
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    batch_results.append(result)
        
        return batch_results
        
    except Exception as e:
        logger.error(f"Error in batch recommendation generation: {str(e)}")
        raise e


def display_batch_recommendation_results(batch_results: List[Dict[str, Any]], container=None):
    """Display batch recommendation results in a Streamlit container"""
    if container is None:
        container = st
        
    container.markdown("### 📊 Batch Recommendation Results")
    
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
        results_data.append({
            "Construct ID": result["construct_id"],
            "Template": result["template_id"],
            "Spec Name": result["spec_name"],
            "Status": "✅ Success" if result["success"] else "❌ Failed",
            "Error": result.get("error", ""),
            "Timestamp": result["timestamp"]
        })
    
    results_df = pd.DataFrame(results_data)
    container.dataframe(results_df, use_container_width=True)
    
    # Save results option
    if container.button("💾 Save Results", key="save_batch_results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_recommendations_{timestamp}.json"
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        container.success(f"Results saved as {filename}")


def get_top_ranked_constructs(project_id: str, evaluator: MetaArtemisEvaluator, top_n: int = 10) -> List[str]:
    """Get top-ranked constructs based on ranking tags (Rank1, Rank2, etc.)"""
    try:
        logger.info(f"🔍 Getting construct information for project {project_id}")
        
        # Get all constructs for the project
        constructs_info = evaluator.falcon_client.get_constructs_info(project_id)
        
        logger.info(f"📊 Found {len(constructs_info)} total constructs")
        
        # Look for constructs with ranking tags
        individual_ranked_constructs = []  # For specific ranks like "RANK 1", "RANK 2"
        range_ranked_constructs = []       # For range ranks like "RANK 1-10"
        
        for construct_id, construct_data in constructs_info.items():
            # Check if construct has tags
            tags = getattr(construct_data, 'tags', []) or []
            
            # Look for ranking tags
            for tag in tags:
                # Handle TagResponse objects
                tag_name = tag.name if hasattr(tag, 'name') else str(tag)
                
                if tag_name.upper().startswith('RANK'):
                    # Check if it's an individual rank (e.g., "RANK 1", "RANK 2")
                    if '-' not in tag_name:
                        try:
                            rank_str = tag_name.upper().replace('RANK', '').strip()
                            rank_number = int(rank_str)
                            individual_ranked_constructs.append((str(construct_id), rank_number, construct_data))
                            logger.info(f"🏷️ Found individual ranked construct: {str(construct_id)[:8]}... with tag '{tag_name}' (rank {rank_number})")
                        except ValueError:
                            logger.warning(f"⚠️ Could not parse individual rank from tag '{tag_name}' for construct {str(construct_id)[:8]}...")
                    
                    # Check if it's a range rank (e.g., "RANK 1-10")
                    elif '-' in tag_name and '1-10' in tag_name:
                        range_ranked_constructs.append((str(construct_id), tag_name, construct_data))
                        logger.debug(f"📝 Found range ranked construct: {str(construct_id)[:8]}... with tag '{tag_name}'")
        
        # Sort individual ranks
        individual_ranked_constructs.sort(key=lambda x: x[1])
        
        logger.info(f"🎯 Found {len(individual_ranked_constructs)} individually ranked constructs")
        logger.info(f"📋 Found {len(range_ranked_constructs)} range ranked constructs (RANK 1-10)")
        
        # Create final ranking list
        final_ranked = []
        used_ranks = set()
        
        # Add individual ranks first
        for construct_id, rank, construct_data in individual_ranked_constructs:
            if rank <= top_n:
                final_ranked.append((construct_id, rank))
                used_ranks.add(rank)
        
        # Fill missing ranks from 1 to top_n with range ranked constructs
        missing_ranks = [i for i in range(1, top_n + 1) if i not in used_ranks]
        
        if missing_ranks and range_ranked_constructs:
            logger.info(f"🔄 Filling missing ranks {missing_ranks} with range ranked constructs")
            
            # Assign range constructs to missing ranks
            for i, (construct_id, tag_name, construct_data) in enumerate(range_ranked_constructs):
                if i < len(missing_ranks):
                    rank = missing_ranks[i]
                    final_ranked.append((construct_id, rank))
                    logger.info(f"🏷️ Assigned range construct {construct_id[:8]}... to rank {rank}")
        
        # Sort final ranking by rank
        final_ranked.sort(key=lambda x: x[1])
        
        # Take only top_n
        final_ranked = final_ranked[:top_n]
        
        logger.info(f"🎯 Final top {len(final_ranked)} ranked constructs:")
        for construct_id, rank in final_ranked:
            logger.info(f"   - Rank {rank}: {construct_id[:8]}...")
        
        # Return just the construct IDs in rank order
        return [construct_id for construct_id, _ in final_ranked]
        
    except Exception as e:
        logger.error(f"❌ Error getting ranked constructs: {str(e)}")
        logger.exception("Full exception details:")
        return []


def get_top_construct_recommendations(
    project_id: str,
    project_specs: List[Dict[str, Any]],
    generated_recommendations: Optional[Dict[str, Any]] = None,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Get recommendations for the top N constructs based on ranking tags or number of specifications.
    
    Args:
        project_id: The project ID
        project_specs: List of project specifications  
        generated_recommendations: Optional generated recommendations from current session
        top_n: Number of top constructs to return
        
    Returns:
        List of recommendation dictionaries
    """
    logger.info(f"🎯 Getting top {top_n} construct recommendations for project {project_id}")
    
    # First check if we have recommendations from current batch process
    if generated_recommendations:
        logger.info("🔍 Checking for recommendations from current batch process")
        
        recommendations = []
        for spec_result in generated_recommendations.get("spec_results", []):
            spec_info = spec_result["spec_info"]
            
            for template_id, template_result in spec_result["template_results"].items():
                if template_result["recommendation"].recommendation_success:
                    rec_info = {
                        "spec_id": spec_info["spec_id"],
                        "construct_id": spec_info["construct_id"],
                        "template_id": template_id,
                        "spec_name": spec_info["name"],
                        "template_name": template_id.replace("_", " ").title(),
                        "recommendation": template_result["recommendation"],
                        "source": "current_batch"
                    }
                    recommendations.append(rec_info)
        
        if recommendations:
            logger.info(f"✅ Found {len(recommendations)} recommendations from current batch")
            return recommendations
    
    # If no current batch recommendations, get existing ones from Artemis
    logger.info("🔍 Getting existing recommendations from Artemis using proper method")
    
    try:
        # Create evaluator to get existing recommendations using the correct method
        from benchmark_evaluator_meta_artemis import MetaArtemisEvaluator, LLMType
        evaluator = MetaArtemisEvaluator(
            task_name="runtime_performance",
            meta_prompt_llm_type=LLMType("gpt-4-o"),
            code_optimization_llm_type=LLMType("gpt-4-o"),
            project_id=project_id
        )
        
        # Setup clients
        import asyncio
        asyncio.run(evaluator.setup_clients())
        
        # Get top-ranked constructs first
        top_ranked_constructs = get_top_ranked_constructs(project_id, evaluator, top_n)
        
        if not top_ranked_constructs:
            logger.warning("⚠️ No ranked constructs found, falling back to recommendation count sorting")
            # Fallback to original method if no ranking tags found
            return get_top_construct_recommendations_fallback(project_id, project_specs, evaluator, top_n)
        
        # Get existing recommendations using the proper method
        existing_recommendations = asyncio.run(evaluator.get_existing_recommendations())
        
        if not existing_recommendations:
            logger.warning("⚠️ No existing recommendations found")
            return []
        
        # Extract recommendations from the categorized structure
        all_existing_recs = []
        all_existing_recs.extend(existing_recommendations.get("meta_recommendations", []))
        all_existing_recs.extend(existing_recommendations.get("baseline_recommendations", []))
        all_existing_recs.extend(existing_recommendations.get("other_recommendations", []))
        
        logger.info(f"📊 Found {len(all_existing_recs)} total existing recommendations")
        logger.info(f"   - Meta recommendations: {len(existing_recommendations.get('meta_recommendations', []))}")
        logger.info(f"   - Baseline recommendations: {len(existing_recommendations.get('baseline_recommendations', []))}")
        logger.info(f"   - Other recommendations: {len(existing_recommendations.get('other_recommendations', []))}")
        
        # Filter recommendations to only include top-ranked constructs
        filtered_recommendations = []
        constructs_with_recs = set()
        
        for rec_info in all_existing_recs:
            if rec_info["construct_id"] in top_ranked_constructs:
                constructs_with_recs.add(rec_info["construct_id"])
                
                # Convert to our format
                template_name = "Existing Recommendation"
                template_id = "existing"
                
                # Try to determine template type from prompt info
                if rec_info.get("prompt_info") and rec_info["prompt_info"].get("name"):
                    prompt_name = rec_info["prompt_info"]["name"]
                    if "simplified" in prompt_name.lower():
                        template_name = "Simplified Template"
                        template_id = "simplified"
                    elif "standard" in prompt_name.lower():
                        template_name = "Standard Template"
                        template_id = "standard"
                    elif "enhanced" in prompt_name.lower():
                        template_name = "Enhanced Template"
                        template_id = "enhanced"
                    elif "baseline" in prompt_name.lower():
                        template_name = "Baseline"
                        template_id = "baseline"
                
                recommendation_obj = {
                    "spec_id": rec_info["spec_id"],
                    "construct_id": rec_info["construct_id"],
                    "template_id": template_id,
                    "spec_name": rec_info["spec_name"],
                    "template_name": template_name,
                    "ai_run_id": rec_info["ai_run_id"],
                    "status": rec_info["status"],
                    "created_at": rec_info["created_at"],
                    "source": "existing_artemis"
                }
                
                filtered_recommendations.append(recommendation_obj)
        
        logger.info(f"🎯 Filtered to {len(filtered_recommendations)} recommendations from top-ranked constructs")
        logger.info(f"📊 Constructs with recommendations: {len(constructs_with_recs)} out of {len(top_ranked_constructs)}")
        
        # Add placeholder entries for top-ranked constructs without recommendations
        constructs_without_recs = [c for c in top_ranked_constructs if c not in constructs_with_recs]
        
        if constructs_without_recs:
            logger.info(f"⚠️ Adding placeholders for {len(constructs_without_recs)} constructs without recommendations")
            
            for construct_id in constructs_without_recs:
                # Find construct rank for display
                construct_rank = top_ranked_constructs.index(construct_id) + 1
                
                placeholder_rec = {
                    "spec_id": f"placeholder_{construct_id}",
                    "construct_id": construct_id,
                    "template_id": "no_recommendation",
                    "spec_name": f"Rank {construct_rank} Construct",
                    "template_name": "⚠️ No Recommendation Available",
                    "ai_run_id": None,
                    "status": "pending",
                    "created_at": "Unknown",
                    "source": "placeholder"
                }
                filtered_recommendations.append(placeholder_rec)
                logger.info(f"📝 Added placeholder for Rank {construct_rank} construct: {construct_id[:8]}...")
        
        # Sort recommendations by construct rank (based on order in top_ranked_constructs)
        construct_rank_map = {construct_id: i for i, construct_id in enumerate(top_ranked_constructs)}
        filtered_recommendations.sort(key=lambda x: construct_rank_map.get(x["construct_id"], 999))
        
        logger.info(f"📋 Final total: {len(filtered_recommendations)} recommendations for {len(top_ranked_constructs)} top-ranked constructs")
        
        return filtered_recommendations
        
    except Exception as e:
        logger.error(f"❌ Error getting existing recommendations: {str(e)}")
        logger.exception("Full exception details:")
        return []


def get_top_construct_recommendations_fallback(project_id: str, project_specs: List[Dict[str, Any]], evaluator: MetaArtemisEvaluator, top_n: int) -> List[Dict[str, Any]]:
    """Fallback method using original recommendation count sorting"""
    try:
        # Get existing recommendations using the proper method
        existing_recommendations = asyncio.run(evaluator.get_existing_recommendations())
        
        if not existing_recommendations:
            logger.warning("⚠️ No existing recommendations found")
            return []
        
        # Extract recommendations from the categorized structure
        all_existing_recs = []
        all_existing_recs.extend(existing_recommendations.get("meta_recommendations", []))
        all_existing_recs.extend(existing_recommendations.get("baseline_recommendations", []))
        all_existing_recs.extend(existing_recommendations.get("other_recommendations", []))
        
        # Group recommendations by construct_id
        construct_groups = {}
        for rec_info in all_existing_recs:
            construct_id = rec_info["construct_id"]
            if construct_id not in construct_groups:
                construct_groups[construct_id] = []
            
            # Convert to our format
            template_name = "Existing Recommendation"
            template_id = "existing"
            
            # Try to determine template type from prompt info
            if rec_info.get("prompt_info") and rec_info["prompt_info"].get("name"):
                prompt_name = rec_info["prompt_info"]["name"]
                if "simplified" in prompt_name.lower():
                    template_name = "Simplified Template"
                    template_id = "simplified"
                elif "standard" in prompt_name.lower():
                    template_name = "Standard Template"
                    template_id = "standard"
                elif "enhanced" in prompt_name.lower():
                    template_name = "Enhanced Template"
                    template_id = "enhanced"
                elif "baseline" in prompt_name.lower():
                    template_name = "Baseline"
                    template_id = "baseline"
            
            recommendation_obj = {
                "spec_id": rec_info["spec_id"],
                "construct_id": rec_info["construct_id"],
                "template_id": template_id,
                "spec_name": rec_info["spec_name"],
                "template_name": template_name,
                "ai_run_id": rec_info["ai_run_id"],
                "status": rec_info["status"],
                "created_at": rec_info["created_at"],
                "source": "existing_artemis"
            }
            
            construct_groups[construct_id].append(recommendation_obj)
        
        # Select top N constructs based on number of recommendations
        sorted_constructs = sorted(
            construct_groups.keys(),
            key=lambda x: len(construct_groups[x]),
            reverse=True
        )[:top_n]
        
        # Get recommendations for top constructs
        recommendations = []
        for construct_id in sorted_constructs:
            if construct_id in construct_groups:
                recommendations.extend(construct_groups[construct_id])
        
        return recommendations
        
    except Exception as e:
        logger.error(f"❌ Error in fallback method: {str(e)}")
        return []


def display_recommendations_table(project_id: str, project_name: str, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Display recommendations in an interactive table format with project-specific selection controls.
    
    Args:
        project_id: The project ID
        project_name: The project display name
        recommendations: List of recommendation dictionaries
        
    Returns:
        List of selected recommendation dictionaries
    """
    logger.info(f"🎯 Display function called with {len(recommendations)} recommendations for {project_name}")
    
    if not recommendations:
        st.warning(f"⚠️ No recommendations found for {project_name}")
        return []
    
    # Separate regular recommendations from placeholders
    regular_recommendations = [r for r in recommendations if r.get("source") != "placeholder"]
    placeholder_recommendations = [r for r in recommendations if r.get("source") == "placeholder"]
    
    logger.info(f"📊 Regular recommendations: {len(regular_recommendations)}, Placeholder recommendations: {len(placeholder_recommendations)}")
    
    # Create expandable section for this project
    with st.expander(f"📋 {project_name} - Top Constructs Recommendations", expanded=True):
        
        # Extract unique LLM types from spec names
        llm_types = set()
        for rec in recommendations:
            spec_name = rec.get("spec_name", "")
            # Extract LLM type from spec name (e.g., "claude-v37-sonnet-cb69e" -> "claude-v37-sonnet")
            if spec_name:
                # Common LLM patterns
                if "claude" in spec_name.lower():
                    if "claude-v37-sonnet" in spec_name.lower():
                        llm_types.add("claude-v37-sonnet")
                    elif "claude" in spec_name.lower():
                        llm_types.add("claude")
                elif "gpt-4" in spec_name.lower():
                    if "gpt-4-o" in spec_name.lower():
                        llm_types.add("gpt-4-o")
                    else:
                        llm_types.add("gpt-4")
                elif "gpt" in spec_name.lower():
                    llm_types.add("gpt")
                elif "gemini" in spec_name.lower():
                    llm_types.add("gemini")
                else:
                    # Extract first part before hyphen or underscore as potential LLM type
                    parts = spec_name.split("-")
                    if len(parts) >= 2:
                        potential_llm = "-".join(parts[:2])
                        llm_types.add(potential_llm)
        
        # Filter section
        if llm_types:
            st.markdown("#### 🔍 Filter Options")
            col1, col2 = st.columns(2)
            with col1:
                llm_filter_options = ["All LLM Types"] + sorted(list(llm_types))
                selected_llm_filter = st.selectbox(
                    "Filter by LLM Type:",
                    options=llm_filter_options,
                    index=0,
                    key=f"llm_filter_{project_id}",
                    help="Filter recommendations by LLM type based on spec name"
                )
            with col2:
                st.markdown("**Available LLM Types:**")
                st.markdown(f"Found: {', '.join(sorted(llm_types))}")
        else:
            selected_llm_filter = "All LLM Types"
        
        # Apply LLM filter
        if selected_llm_filter != "All LLM Types":
            filtered_recommendations = []
            for rec in recommendations:
                spec_name = rec.get("spec_name", "").lower()
                if selected_llm_filter.lower() in spec_name:
                    filtered_recommendations.append(rec)
            recommendations = filtered_recommendations
            logger.info(f"🔍 Filtered to {len(recommendations)} recommendations for LLM type: {selected_llm_filter}")
        
        # Project-specific selection buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Select All - {project_name}", key=f"select_all_{project_id}"):
                # Only select regular recommendations (not placeholders)
                filtered_regular = [r for r in recommendations if r.get("source") != "placeholder"]
                for i, rec in enumerate(filtered_regular):
                    # Find original index in unfiltered list
                    original_index = next((j for j, orig_rec in enumerate(st.session_state.cached_project_recommendations[project_id]) 
                                         if orig_rec["spec_name"] == rec["spec_name"]), None)
                    if original_index is not None:
                        st.session_state[f"rec_select_{project_id}_{original_index}"] = True
        with col2:
            if st.button(f"Deselect All - {project_name}", key=f"deselect_all_{project_id}"):
                for i, rec in enumerate(recommendations):
                    # Find original index in unfiltered list
                    original_index = next((j for j, orig_rec in enumerate(st.session_state.cached_project_recommendations[project_id]) 
                                         if orig_rec["spec_name"] == rec["spec_name"]), None)
                    if original_index is not None:
                        st.session_state[f"rec_select_{project_id}_{original_index}"] = False
        
        # Group recommendations by construct for better organization
        construct_groups = {}
        for rec in recommendations:
            construct_id = rec["construct_id"]
            if construct_id not in construct_groups:
                construct_groups[construct_id] = []
            construct_groups[construct_id].append(rec)
        
        # Sort constructs by their appearance order (rank)
        sorted_constructs = sorted(construct_groups.keys(), key=lambda x: next(
            (i for i, rec in enumerate(recommendations) if rec["construct_id"] == x), 999
        ))
        
        logger.info(f"🏗️ Grouped into {len(construct_groups)} constructs: {[f'{c[:8]}...({len(construct_groups[c])} recs)' for c in sorted_constructs]}")
        
        # Prepare data for the table
        table_data = []
        for i, rec in enumerate(recommendations):
            # Check if this is the first recommendation for this construct
            construct_id = rec["construct_id"]
            is_first_in_group = i == 0 or recommendations[i-1]["construct_id"] != construct_id
            
            # Find construct rank
            construct_rank = None
            if "Rank" in rec["spec_name"]:
                try:
                    construct_rank = int(rec["spec_name"].split("Rank ")[1].split(" ")[0])
                except:
                    construct_rank = None
            
            # Determine if this is selectable (not a placeholder)
            is_selectable = rec.get("source") != "placeholder"
            
            # Find original index in unfiltered list for session state
            original_index = next((j for j, orig_rec in enumerate(st.session_state.cached_project_recommendations[project_id]) 
                                 if orig_rec["spec_name"] == rec["spec_name"]), i)
            
            row_data = {
                "Select": st.session_state.get(f"rec_select_{project_id}_{original_index}", False) if is_selectable else False,
                "Construct ID": construct_id[:8] + "..." if is_first_in_group else "",
                "Rank": f"Rank {construct_rank}" if construct_rank and is_first_in_group else "",
                "Spec Name": rec["spec_name"],
                "Template": rec["template_name"],
                "AI Run ID": rec["ai_run_id"][:8] + "..." if rec["ai_run_id"] else "N/A",
                "Status": rec["status"],
                "Created": rec["created_at"][:19] if rec["created_at"] != "Unknown" else "Unknown",
                "Selectable": is_selectable
            }
            table_data.append(row_data)
        
        logger.info(f"📋 Created table with {len(table_data)} rows for display")
        
        # Display interactive table
        if table_data:
            # Configure columns
            column_config = {
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select recommendations to create solutions from",
                    default=False,
                    width="small"
                ),
                "Construct ID": st.column_config.TextColumn(
                    "Construct ID",
                    help="Unique construct identifier",
                    width="medium"
                ),
                "Rank": st.column_config.TextColumn(
                    "Rank",
                    help="Construct ranking",
                    width="small"
                ),
                "Spec Name": st.column_config.TextColumn(
                    "Spec Name",
                    help="Specification name",
                    width="medium"
                ),
                "Template": st.column_config.TextColumn(
                    "Template",
                    help="Template type used",
                    width="medium"
                ),
                "AI Run ID": st.column_config.TextColumn(
                    "AI Run ID",
                    help="AI run identifier",
                    width="medium"
                ),
                "Status": st.column_config.TextColumn(
                    "Status",
                    help="Recommendation status",
                    width="small"
                ),
                "Created": st.column_config.TextColumn(
                    "Created",
                    help="Creation timestamp",
                    width="medium"
                ),
                "Selectable": st.column_config.CheckboxColumn(
                    "Selectable",
                    help="Whether this item can be selected",
                    width="small"
                )
            }
            
            # Create disabled list for non-selectable rows
            disabled_columns = ["Construct ID", "Rank", "Spec Name", "Template", "AI Run ID", "Status", "Created", "Selectable"]
            
            # Show the table
            edited_data = st.data_editor(
                table_data,
                column_config=column_config,
                disabled=disabled_columns,
                hide_index=True,
                use_container_width=True,
                key=f"recommendations_table_{project_id}"
            )
            
            # Count template distribution for info
            template_counts = {}
            for rec in recommendations:
                template = rec["template_name"]
                template_counts[template] = template_counts.get(template, 0) + 1
            
            logger.info(f"🏷️ Template distribution: {template_counts}")
            
            # Show summary with filter information
            filtered_regular = [r for r in recommendations if r.get("source") != "placeholder"]
            filtered_placeholder = [r for r in recommendations if r.get("source") == "placeholder"]
            
            if selected_llm_filter != "All LLM Types":
                st.info(f"📊 Showing {len(filtered_regular)} available recommendations and {len(filtered_placeholder)} placeholders for top-ranked constructs (filtered by {selected_llm_filter})")
            else:
                st.info(f"📊 Showing {len(filtered_regular)} available recommendations and {len(filtered_placeholder)} placeholders for top-ranked constructs")
            
            # Update session state based on edited data
            for i, (row, rec) in enumerate(zip(edited_data, recommendations)):
                # Find original index in unfiltered list
                original_index = next((j for j, orig_rec in enumerate(st.session_state.cached_project_recommendations[project_id]) 
                                     if orig_rec["spec_name"] == rec["spec_name"]), i)
                st.session_state[f"rec_select_{project_id}_{original_index}"] = row["Select"]
            
            # Extract selected recommendations
            selected_recommendations = []
            for i, (row, rec) in enumerate(zip(edited_data, recommendations)):
                if row["Select"] and rec.get("source") != "placeholder":
                    selected_recommendations.append(rec)
            
            return selected_recommendations
        else:
            st.warning("⚠️ No recommendations to display")
            return [] 