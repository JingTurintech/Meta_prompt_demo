"""
Solution execution functions for the Meta Artemis application.
Handles executing solutions and monitoring their progress.
"""

import streamlit as st
import asyncio
from loguru import logger
from typing import Dict, Any, List, Optional
from uuid import UUID
from benchmark_evaluator_meta_artemis import MetaArtemisEvaluator, LLMType
from artemis_client.falcon.client import FalconClient
from .utils import get_session_state, update_session_state


async def wait_for_solution_completion(falcon_client, solution_id: str, timeout: int = 600):
    """Wait for solution to complete execution"""
    logger.info(f"⏳ Waiting for solution {solution_id} to complete...")
    
    start_time = asyncio.get_event_loop().time()
    
    while asyncio.get_event_loop().time() - start_time < timeout:
        try:
            solution = falcon_client.get_solution(solution_id)
            status = str(solution.status).lower()
            
            logger.info(f"🔍 Solution {solution_id} status: {status}")
            
            if status in ['completed', 'finished', 'success']:
                logger.info(f"✅ Solution {solution_id} completed successfully!")
                return True, None
            elif status in ['failed', 'error', 'cancelled']:
                error_msg = f"Solution {solution_id} failed with status: {status}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg
            
            # Wait before checking again
            await asyncio.sleep(10)
            
        except Exception as e:
            logger.warning(f"Error checking solution status: {str(e)}")
            await asyncio.sleep(5)
    
    # Timeout reached
    error_msg = f"Solution {solution_id} timed out after {timeout} seconds"
    logger.error(f"⏰ {error_msg}")
    return False, error_msg


async def execute_solutions_async():
    """Execute solutions asynchronously with progress tracking"""
    state = get_session_state()
    
    try:
        # Get selected recommendations
        selected_recommendations = state.get("selected_recommendations", [])
        if not selected_recommendations:
            raise ValueError("No recommendations selected for execution")
        
        # Setup evaluator
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
            evaluation_repetitions=state["evaluation_repetitions"]
        )
        
        await evaluator.setup_clients()
        
        # Create and execute solutions
        execution_results = []
        
        for i, rec_info in enumerate(selected_recommendations):
            logger.info(f"🚀 Executing solution {i+1}/{len(selected_recommendations)}: {rec_info['spec_name']}")
            
            try:
                # Get the recommendation result
                recommendation = rec_info["recommendation"]
                spec_info = rec_info["spec_info"]
                
                # Create and execute solution
                solution_result = await evaluator.create_and_execute_solution_from_spec(
                    spec_info, recommendation
                )
                
                execution_results.append({
                    "recommendation_info": rec_info,
                    "solution_result": solution_result,
                    "success": solution_result.success if solution_result else False
                })
                
                logger.info(f"✅ Solution {i+1} executed: {'Success' if solution_result and solution_result.success else 'Failed'}")
                
            except Exception as e:
                logger.error(f"❌ Error executing solution {i+1}: {str(e)}")
                execution_results.append({
                    "recommendation_info": rec_info,
                    "solution_result": None,
                    "success": False,
                    "error": str(e)
                })
        
        return execution_results
        
    except Exception as e:
        logger.error(f"❌ Error in solution execution: {str(e)}")
        st.error(f"Error executing solutions: {str(e)}")
        return None


def execute_solutions_step3():
    """Execute solutions in Step 3 with progress display"""
    state = get_session_state()
    selected_recommendations = state.get("selected_recommendations", [])
    
    if not selected_recommendations:
        st.error("No recommendations selected for execution!")
        return
    
    st.markdown(f"### 🚀 Executing {len(selected_recommendations)} Solutions")
    
    # Show execution configuration
    with st.expander("⚙️ Execution Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Task:** {state['selected_task']}")
            st.markdown(f"**Meta-Prompt LLM:** {state['meta_prompt_llm']}")
            st.markdown(f"**Code Optimization LLM:** {state['code_optimization_llm']}")
        with col2:
            st.markdown(f"**Evaluation Repetitions:** {state['evaluation_repetitions']}")
            st.markdown(f"**Selected Templates:** {', '.join(state.get('selected_templates', []))}")
            if state.get("custom_worker_name"):
                st.markdown(f"**Worker Name:** {state['custom_worker_name']}")
    
    # Execution button
    if st.button("🚀 Start Execution", key="start_execution", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        try:
            with st.spinner("Executing solutions..."):
                status_text.text("Setting up execution environment...")
                
                # Execute solutions
                execution_results = asyncio.run(execute_solutions_async())
                
                if execution_results:
                    # Update progress
                    progress_bar.progress(100)
                    status_text.text("✅ Execution completed!")
                    
                    # Store results
                    update_session_state({
                        "execution_results": execution_results,
                        "execution_completed": True
                    })
                    
                    # Display results summary
                    successful_executions = sum(1 for result in execution_results if result["success"])
                    
                    with results_container:
                        st.success(f"🎉 Execution completed! {successful_executions}/{len(execution_results)} solutions executed successfully")
                        
                        # Results summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Solutions", len(execution_results))
                        with col2:
                            st.metric("Successfully Executed", successful_executions)
                        with col3:
                            success_rate = (successful_executions / len(execution_results)) * 100
                            st.metric("Success Rate", f"{success_rate:.1f}%")
                        
                        # Continue to results visualization
                        if st.button("➡️ View Results (Step 4)", key="continue_to_results"):
                            update_session_state({
                                "current_step": 4,
                                "current_results": {"execution_results": execution_results}
                            })
                            st.rerun()
                
                else:
                    progress_bar.progress(0)
                    status_text.text("❌ Execution failed!")
                    st.error("Failed to execute solutions. Check logs for details.")
                    
        except Exception as e:
            progress_bar.progress(0)
            status_text.text(f"❌ Error: {str(e)}")
            st.error(f"Error during execution: {str(e)}")


def get_execution_summary(execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get summary statistics from execution results"""
    if not execution_results:
        return {"total": 0, "successful": 0, "failed": 0, "success_rate": 0}
    
    total = len(execution_results)
    successful = sum(1 for result in execution_results if result.get("success", False))
    failed = total - successful
    success_rate = (successful / total) * 100 if total > 0 else 0
    
    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "success_rate": success_rate
    }


def format_execution_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format execution result for display"""
    rec_info = result.get("recommendation_info", {})
    solution_result = result.get("solution_result")
    
    formatted = {
        "spec_name": rec_info.get("spec_name", "Unknown"),
        "template_name": rec_info.get("template_name", "Unknown"),
        "construct_id": rec_info.get("construct_id", "Unknown"),
        "success": result.get("success", False),
        "error": result.get("error", ""),
    }
    
    if solution_result:
        formatted.update({
            "solution_id": solution_result.solution_id,
            "execution_time": getattr(solution_result, "execution_time", 0),
            "runtime_metrics": getattr(solution_result, "runtime_metrics", {}),
            "memory_metrics": getattr(solution_result, "memory_metrics", {}),
        })
    
    return formatted


async def get_solution_details_from_artemis(solution_ids: List[str]) -> Dict[str, Any]:
    """Get detailed solution information from Artemis"""
    logger.info(f"🔍 Getting solution details for {len(solution_ids)} solutions")
    
    try:
        state = get_session_state()
        
        # Setup evaluator to access Falcon client
        evaluator = MetaArtemisEvaluator(
            task_name="runtime_performance",
            meta_prompt_llm_type=LLMType("gpt-4-o"),
            code_optimization_llm_type=LLMType("gpt-4-o"),
            project_id=state["project_id"]
        )
        await evaluator.setup_clients()
        
        solution_details = {}
        
        for solution_id in solution_ids:
            try:
                solution = evaluator.falcon_client.get_solution(solution_id)
                
                solution_info = {
                    "solution_id": solution_id,
                    "status": str(solution.status),
                    "created_at": str(solution.created_at) if hasattr(solution, 'created_at') else "Unknown",
                    "has_results": bool(solution.results),
                    "specs": [{"spec_id": str(spec.spec_id), "construct_id": str(spec.construct_id)} for spec in solution.specs] if solution.specs else []
                }
                
                # Extract results if available
                if solution.results and hasattr(solution.results, 'values'):
                    results_summary = {}
                    for metric_name, values in solution.results.values.items():
                        if values:
                            results_summary[metric_name] = {
                                "avg": float(sum(values) / len(values)),
                                "min": float(min(values)),
                                "max": float(max(values)),
                                "count": len(values),
                                "values": values
                            }
                    
                    solution_info["results_summary"] = results_summary
                
                solution_details[solution_id] = solution_info
                
            except Exception as e:
                logger.warning(f"Could not get details for solution {solution_id}: {str(e)}")
                solution_details[solution_id] = {
                    "solution_id": solution_id,
                    "status": "error",
                    "error": str(e)
                }
        
        return solution_details
        
    except Exception as e:
        logger.error(f"❌ Error getting solution details: {str(e)}")
        return {} 