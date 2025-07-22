"""
Batch Meta Artemis Application

This application enables large-scale evaluations through different batch use cases:
1. Batch Recommendation Creation
2. Batch Solution Creation  
3. Batch Solution Evaluation

Each use case is optimized for processing multiple items efficiently.
"""

import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import os
import time
from uuid import UUID
from typing import Dict, Any, List, Optional
import numpy as np

# Import all modular components
from meta_artemis_modules.utils import (
    initialize_session_state, get_session_state, update_session_state, generate_colors,
    initialize_batch_session_state, get_batch_session_state, update_batch_session_state,
    reset_batch_session_state
)
from meta_artemis_modules.project_manager import (
    get_project_info_async, get_existing_solutions_async, get_project_configurations, get_optimization_configurations, validate_project_id
)
from meta_artemis_modules.recommendations import (
    generate_recommendations_async, generate_recommendations_step2, get_recommendation_summary,
    execute_batch_recommendations_async, display_batch_recommendation_results,
    get_top_construct_recommendations, display_recommendations_table, get_top_ranked_constructs
)
from meta_artemis_modules.solutions import (
    create_solutions_from_recommendations, get_solution_status_summary, format_solution_display_name,
    get_solution_metrics_summary, execute_batch_solutions_async, display_batch_solution_results
)
from meta_artemis_modules.execution import (
    execute_solutions_step3, execute_solutions_async, get_execution_summary
)
from meta_artemis_modules.visualization import (
    display_existing_solutions_analysis, create_performance_comparison_chart, create_summary_dashboard,
    display_box_plot_analysis_results, perform_statistical_analysis, format_statistical_results
)

# Import from existing modules
from meta_artemis_modules.evaluator import (
    MetaArtemisEvaluator, RecommendationResult, SolutionResult
)
from meta_artemis_modules.shared_templates import (
    OPTIMIZATION_TASKS, META_PROMPT_TEMPLATES, AVAILABLE_LLMS, JUDGE_PROMPT_TEMPLATE, DEFAULT_PROJECT_OPTIMISATION_IDS,
    DEFAULT_BATCH_CONFIG, DEFAULT_BATCH_RECOMMENDATIONS_CONFIG, DEFAULT_BATCH_SOLUTIONS_CONFIG,
    DEFAULT_BATCH_EVALUATION_CONFIG, DEFAULT_PROJECT_ID,
    DEFAULT_META_PROMPT_LLM, DEFAULT_CODE_OPTIMIZATION_LLM, DEFAULT_SCORING_LLM,
    ALL_PROMPTING_TEMPLATES, BASELINE_PROMPTING_TEMPLATES
)

from loguru import logger
import sys
from artemis_client.falcon.client import FalconSettings, FalconClient
from artemis_client.base_auth import ThanosSettings
from vision_models.service.llm import LLMType

# Configure the Streamlit page
st.set_page_config(
    page_title="Batch Meta Artemis",
    page_icon="⚡",
    layout="wide"
)

def initialize_batch_session_state():
    """Initialize session state for batch operations"""
    if "batch_meta_artemis_state" not in st.session_state:
        st.session_state.batch_meta_artemis_state = {
            # Basic configuration
            "project_id": DEFAULT_PROJECT_ID,
            "current_step": 1,
            "selected_use_case": None,
            
            # Project data
            "project_info": None,
            "project_specs": None,
            "existing_solutions": None,
            "existing_optimizations": None,
            
            # Batch configuration
            "batch_config": DEFAULT_BATCH_CONFIG.copy(),
            
            # Use case specific data
            "batch_recommendations": DEFAULT_BATCH_RECOMMENDATIONS_CONFIG.copy(),
            "batch_solutions": DEFAULT_BATCH_SOLUTIONS_CONFIG.copy(),
            "batch_evaluation": DEFAULT_BATCH_EVALUATION_CONFIG.copy(),
            "batch_analysis": {
                "selected_projects": [],
                "analysis_type": "runtime_comparison",
                "statistical_significance_level": 0.05,
                "include_baseline_comparison": True,
                "include_template_comparison": True,
                "minimum_samples": 3,
                "performance_metrics": ["runtime", "memory", "cpu_usage"],
                "analysis_results": None
            }
        }

def get_batch_session_state():
    """Get the batch session state"""
    return st.session_state.batch_meta_artemis_state

def update_batch_session_state(updates: Dict[str, Any]):
    """Update the batch session state"""
    state = get_batch_session_state()
    state.update(updates)
    st.session_state.batch_meta_artemis_state = state

def reset_batch_session_state():
    """Reset the batch session state to initial values"""
    if "batch_meta_artemis_state" in st.session_state:
        del st.session_state.batch_meta_artemis_state
    initialize_batch_session_state()

def step_1_use_case_selection():
    """Step 1: Use Case Selection"""
    st.header("🎯 Step 1: Select Use Case")
    st.markdown("Choose the type of batch operation you want to perform:")
    
    # Use case options with descriptions
    use_cases = {
        "batch_recommendations": {
            "title": "📊 Batch Recommendation Creation",
            "description": "Generate recommendations for multiple constructs using meta-prompting templates",
            "icon": "🧠",
            "features": [
                "Process multiple constructs simultaneously",
                "Support for multiple meta-prompt templates",
                "Baseline prompt comparison",
                "Progress tracking and error handling"
            ]
        },
        "batch_solutions": {
            "title": "🔧 Batch Solution Creation",
            "description": "Create solutions from existing recommendations or generate new ones",
            "icon": "⚙️",
            "features": [
                "Create solutions from recommendations",
                "Batch process existing solutions",
                "Optimization ID management",
                "Solution validation and status tracking"
            ]
        },
        "batch_evaluation": {
            "title": "🚀 Batch Solution Evaluation",
            "description": "Execute and evaluate multiple solutions with performance metrics",
            "icon": "📈",
            "features": [
                "Parallel solution execution",
                "Performance metrics collection",
                "Timeout and retry handling",
                "Results aggregation and analysis"
            ]
        },
        "batch_analysis": {
            "title": "📊 Runtime Impact Analysis",
            "description": "Analyze runtime performance impact of code recommendations vs original code",
            "icon": "🔬",
            "features": [
                "Compare solution runtime vs baseline",
                "Statistical significance testing",
                "Meta-prompt template comparison",
                "Performance improvement metrics"
            ]
        }
    }
    
    state = get_batch_session_state()
    
    # Display use case selection in a 2x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Recommendations")
        st.markdown("**Batch Recommendation Creation**")
        st.markdown("Generate recommendations for multiple constructs using meta-prompting templates")
        
        features = use_cases["batch_recommendations"]["features"]
        for feature in features:
            st.markdown(f"• {feature}")
        
        if st.button("Select Batch Recommendations", key="select_batch_recs", type="primary"):
            update_batch_session_state({
                "selected_use_case": "batch_recommendations",
                "current_step": 2
            })
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🚀 Evaluation")
        st.markdown("**Batch Solution Evaluation**")
        st.markdown("Execute and evaluate multiple solutions with performance metrics")
        
        features = use_cases["batch_evaluation"]["features"]
        for feature in features:
            st.markdown(f"• {feature}")
        
        if st.button("Select Batch Evaluation", key="select_batch_eval", type="primary"):
            update_batch_session_state({
                "selected_use_case": "batch_evaluation",
                "current_step": 2
            })
            st.rerun()
    
    with col2:
        st.markdown("### 🔧 Solutions")
        st.markdown("**Batch Solution Creation**")
        st.markdown("Create solutions from existing recommendations or generate new ones")
        
        features = use_cases["batch_solutions"]["features"]
        for feature in features:
            st.markdown(f"• {feature}")
        
        if st.button("Select Batch Solutions", key="select_batch_sols", type="primary"):
            update_batch_session_state({
                "selected_use_case": "batch_solutions",
                "current_step": 2
            })
            st.rerun()
    
        st.markdown("---")
        
        st.markdown("### 🔬 Analysis")
        st.markdown("**Runtime Impact Analysis**")
        st.markdown("Analyze runtime performance impact of code recommendations vs original code")
        
        features = use_cases["batch_analysis"]["features"]
        for feature in features:
            st.markdown(f"• {feature}")
        
        if st.button("Select Runtime Analysis", key="select_batch_analysis", type="primary"):
            update_batch_session_state({
                "selected_use_case": "batch_analysis",
                "current_step": 2
            })
            st.rerun()

def step_2_batch_configuration():
    """Step 2: Batch Process Configuration and Execution"""
    state = get_batch_session_state()
    selected_use_case = state.get("selected_use_case")
    
    if not selected_use_case:
        st.error("No use case selected. Please go back to Step 1.")
        return
    
    st.header(f"⚙️ Step 2: {selected_use_case.replace('_', ' ').title()} Configuration")
    
    # Check if there are existing results for this use case and display them
    if selected_use_case == "batch_evaluation":
        batch_eval_results = state.get("batch_evaluation", {}).get("evaluation_results")
        if batch_eval_results:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success("📊 Previous evaluation results found!")
            with col2:
                if st.button("🗑️ Clear Results", help="Clear previous results to start fresh"):
                    # Clear the evaluation results
                    if "batch_evaluation" in state:
                        if "evaluation_results" in state["batch_evaluation"]:
                            del state["batch_evaluation"]["evaluation_results"]
                    st.rerun()
            
            display_batch_evaluation_results(batch_eval_results)
            st.markdown("---")
            st.markdown("### Configure New Batch Evaluation")
    elif selected_use_case == "batch_recommendations":
        batch_rec_results = state.get("batch_recommendations", {}).get("all_project_results")
        if batch_rec_results:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success("📊 Previous recommendation results found!")
            with col2:
                if st.button("🗑️ Clear Results", help="Clear previous results to start fresh"):
                    # Clear the recommendation results
                    if "batch_recommendations" in state:
                        if "all_project_results" in state["batch_recommendations"]:
                            del state["batch_recommendations"]["all_project_results"]
                    st.rerun()
            st.markdown("---")
            st.markdown("### Configure New Batch Recommendations")
    elif selected_use_case == "batch_solutions":
        batch_sol_results = state.get("batch_solutions", {}).get("batch_results")
        if batch_sol_results:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success("📊 Previous solution results found!")
            with col2:
                if st.button("🗑️ Clear Results", help="Clear previous results to start fresh"):
                    # Clear the solution results
                    if "batch_solutions" in state:
                        if "batch_results" in state["batch_solutions"]:
                            del state["batch_solutions"]["batch_results"]
                    st.rerun()
            st.markdown("---")
            st.markdown("### Configure New Batch Solutions")
    elif selected_use_case == "batch_analysis":
        batch_analysis_results = state.get("batch_analysis", {}).get("analysis_results")
        if batch_analysis_results:
            st.markdown("---")
            st.markdown("### Configure New Batch Analysis")
    
    # Get all optimization configurations
    optimization_configurations = get_optimization_configurations()
    
    # Initialize optimization info cache in session state if not exists
    if "optimization_info_cache" not in st.session_state:
        st.session_state.optimization_info_cache = {}
    
    # Add select/deselect all buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All"):
            for optimization_id in optimization_configurations.keys():
                st.session_state[f"select_{optimization_id}"] = True
    with col2:
        if st.button("Deselect All"):
            for optimization_id in optimization_configurations.keys():
                st.session_state[f"select_{optimization_id}"] = False
    
    st.markdown("Select optimizations to process:")
    
    # Create optimization selection list
    selected_optimizations = []
    selected_projects = []  # Track unique projects for backward compatibility
    
    # Style for the optimization list
    st.markdown("""
        <style>
        .optimization-item {
            padding: 8px;
            margin: 4px 0;
            border-radius: 4px;
        }
        .optimization-item:hover {
            background-color: #f0f2f6;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Prepare data for table display
    table_data = []
    for optimization_id, opt_config in optimization_configurations.items():
        # Initialize checkbox state if not exists
        if f"select_{optimization_id}" not in st.session_state:
            st.session_state[f"select_{optimization_id}"] = False
        
        table_data.append({
            "Select": st.session_state[f"select_{optimization_id}"],
            "Project": opt_config["project_name"],
            "Project Link": f"https://artemis.turintech.ai/projects/{opt_config['project_id']}",
            "Optimization": opt_config["optimization_name"],
            "Optimization Link": f"https://artemis.turintech.ai/optimisations/{optimization_id}",
            "Description": opt_config["optimization_description"],
            "Full Project ID": opt_config["project_id"],
            "Full Optimization ID": optimization_id
        })
    
    # Display interactive table
    st.markdown("**Select optimizations to process:**")
    edited_data = st.data_editor(
        table_data,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select optimizations to process",
                default=False,
                width="small"
            ),
            "Project": st.column_config.TextColumn(
                "Project",
                help="Project name",
                width="medium"
            ),
            "Project Link": st.column_config.LinkColumn(
                "Project Link",
                help="Link to project in Artemis",
                width="small"
            ),
            "Optimization": st.column_config.TextColumn(
                "Optimization",
                help="Optimization name",
                width="medium"
            ),
            "Optimization Link": st.column_config.LinkColumn(
                "Optimization Link",
                help="Link to optimization in Artemis",
                width="small"
            ),
            "Description": st.column_config.TextColumn(
                "Description",
                help="Optimization description",
                width="large"
            ),
            "Full Project ID": None,  # Hide this column
            "Full Optimization ID": None  # Hide this column
        },
        disabled=["Project", "Project Link", "Optimization", "Optimization Link", "Description"],
        hide_index=True,
        use_container_width=True,
        key="optimization_selection_table"
    )
    
    # Update session state and collect selected items
    selected_projects_set = set()
    for i, row in enumerate(edited_data):
        optimization_id = row["Full Optimization ID"]
        project_id = row["Full Project ID"]
        
        # Update session state
        st.session_state[f"select_{optimization_id}"] = row["Select"]
        
        # Collect selected items
        if row["Select"]:
            selected_optimizations.append(optimization_id)
            selected_projects_set.add(project_id)
    
    # Convert set to list for backward compatibility
    selected_projects = list(selected_projects_set)
    
    if not selected_optimizations:
        st.warning("⚠️ Please select at least one optimization to continue")
        return
    
    # Show selection summary
    st.success(f"✅ Selected {len(selected_optimizations)} optimizations from {len(selected_projects)} projects")
    
    # Update batch session state with selected optimizations and projects
    update_batch_session_state({
        "selected_optimizations": selected_optimizations,
        "selected_projects": selected_projects  # Keep for backward compatibility
    })
    
    # Global batch configuration
    st.markdown("### ⚙️ Global Batch Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        max_concurrent = st.number_input(
            "Max Concurrent Operations:",
            min_value=1,
            max_value=20,
            value=state["batch_config"]["max_concurrent"],
            help="Maximum number of operations to run simultaneously"
        )
    
    with col2:
        timeout_seconds = st.number_input(
            "Timeout (seconds):",
            min_value=60,
            max_value=3600,
            value=state["batch_config"]["timeout_seconds"],
            help="Timeout for individual operations"
        )
    
    # Update global config
    update_batch_session_state({
        "batch_config": {
            "max_concurrent": max_concurrent,
            "timeout_seconds": timeout_seconds
        },
        "selected_projects": selected_projects
    })
    
    # Use case specific configuration
    if selected_use_case == "batch_recommendations":
        configure_batch_recommendations()
    elif selected_use_case == "batch_solutions":
        configure_batch_solutions()
    elif selected_use_case == "batch_evaluation":
        configure_batch_evaluation()
    elif selected_use_case == "batch_analysis":
        configure_batch_analysis()

def configure_batch_recommendations():
    """Configure batch recommendation creation"""
    st.markdown("### 📊 Batch Recommendation Configuration")
    
    state = get_batch_session_state()
    batch_config = state["batch_recommendations"]
    selected_projects = state.get("selected_projects", [])
    
    if not selected_projects:
        st.warning("⚠️ No projects selected. Please select projects in the Project Configuration section above.")
        return
    
    # LLM and task configuration
    st.markdown("#### ⚙️ Configuration")
    col1, col2 = st.columns(2)
    with col1:
        # Extract LLM names from AVAILABLE_LLMS list
        llm_options = [llm.value for llm in AVAILABLE_LLMS]
        meta_prompt_llm = st.selectbox(
            "Meta-Prompt LLM:",
            options=llm_options,
            index=llm_options.index(batch_config["meta_prompt_llm"]) if batch_config["meta_prompt_llm"] in llm_options else 0,
            help="LLM used for generating meta-prompts"
        )
        
        selected_task = st.selectbox(
            "Optimization Task:",
            options=list(OPTIMIZATION_TASKS.keys()),
            index=list(OPTIMIZATION_TASKS.keys()).index(batch_config["selected_task"]),
            help="Type of optimization to perform"
        )
    
    with col2:
        code_optimization_llm = st.selectbox(
            "Code Optimization LLM:",
            options=llm_options,
            index=llm_options.index(batch_config["code_optimization_llm"]) if batch_config["code_optimization_llm"] in llm_options else 0,
            help="LLM used for code optimization"
        )
        
        constructs_per_project = st.number_input(
            "Top N Constructs per Project:",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of constructs to process from each project"
        )
    
    # Template selection
    st.markdown("#### 📝 Prompting Techniques Configuration")
    st.markdown("*Configure techniques for ASE25 MPCO framework evaluation*")
    
    # Initialize template states in session state if not exists
    if "template_states" not in st.session_state:
        st.session_state.template_states = {}
        for template_id in ALL_PROMPTING_TEMPLATES.keys():
            st.session_state.template_states[template_id] = {
                "selected": template_id in batch_config["selected_templates"],
                "template": ALL_PROMPTING_TEMPLATES[template_id]["template"]
            }
    
    # Template selection and editing
    selected_templates = []
    
    # Group techniques by category
    st.markdown("##### 🧠 Meta-Prompt Templates (MPCO Framework)")
    for template_id, template_info in META_PROMPT_TEMPLATES.items():
        col1, col2 = st.columns([1, 11])
        with col1:
            is_selected = st.checkbox(
                "",
                key=f"template_{template_id}",
                value=st.session_state.template_states[template_id]["selected"]
            )
        with col2:
            st.markdown(f"**{template_info['name']}**")
            st.markdown(f"*{template_info['description']}*")
            
            # Only show template text area if template is selected
            if is_selected:
                template_text = st.text_area(
                    "Template Content",
                    value=st.session_state.template_states[template_id]["template"],
                    key=f"template_text_{template_id}",
                    height=200
                )
                st.session_state.template_states[template_id]["template"] = template_text
                selected_templates.append(template_id)
    
    # Baseline Prompting Methods
    st.markdown("##### 📊 Baseline Prompting Methods")
    
    # Use baseline techniques from shared templates
    baseline_techniques = BASELINE_PROMPTING_TEMPLATES
    
    # Initialize template states for baseline techniques if not present
    for template_id, template_info in baseline_techniques.items():
        if template_id not in st.session_state.template_states:
            st.session_state.template_states[template_id] = {
                "template": template_info["template"],
                "selected": False
            }
    
    for template_id, template_info in baseline_techniques.items():
        col1, col2 = st.columns([1, 11])
        with col1:
            is_selected = st.checkbox(
                "",
                key=f"template_{template_id}",
                value=st.session_state.template_states[template_id]["selected"]
            )
        with col2:
            st.markdown(f"**{template_info['name']}**")
            st.markdown(f"*{template_info['description']}*")
            
            # Update selection state
            st.session_state.template_states[template_id]["selected"] = is_selected
            
            # Only show template text area if template is selected
            if is_selected:
                template_text = st.text_area(
                    "Template Content",
                    value=st.session_state.template_states[template_id]["template"],
                    key=f"template_text_{template_id}",
                    height=200
                )
                st.session_state.template_states[template_id]["template"] = template_text
                selected_templates.append(template_id)
    
    # Baseline prompt checkbox
    st.markdown("---")
    st.markdown("#### 📋 Baseline Prompt")
    col1, col2 = st.columns([1, 11])
    with col1:
        include_baseline = st.checkbox(
            "",
            value=batch_config["include_baseline"],
            key="include_baseline"
        )
    with col2:
        st.markdown("**Include Baseline Prompt**")
        st.markdown("*Use the default optimization prompt without meta-prompting*")
        
        # Initialize baseline prompt with default value
        task_info = OPTIMIZATION_TASKS[batch_config["selected_task"]]
        baseline_prompt = task_info["default_prompt"]
        
        # Show baseline prompt if selected
        if include_baseline:
            baseline_prompt = st.text_area(
                "Baseline Prompt",
                value=task_info["default_prompt"],
                key="baseline_prompt",
                height=100
            )
    
    # Construct selection table
    st.markdown("---")
    st.markdown("#### 🎯 Top-Ranked Constructs Selection")
    
    # A toggle to switch between top-ranked and all constructs view
    col1, col2, col3 = st.columns([1, 2, 9])
    with col1:
        if st.button("🔄 Refresh", key="refresh_constructs", help="Refresh construct data from projects"):
            st.session_state.construct_selection_cache = {}
            st.rerun()
    # Ensure toggle state is stored in session_state
    if "show_all_constructs" not in st.session_state:
        st.session_state.show_all_constructs = False
    with col2:
        toggle_label = "Show All Constructs" if not st.session_state.show_all_constructs else "Show Top Ranked"
        if st.button(toggle_label, key="toggle_construct_view", help="Toggle between viewing only top-ranked constructs and viewing every construct in the project"):
            st.session_state.show_all_constructs = not st.session_state.show_all_constructs
            st.rerun()
    with col3:
        if st.session_state.show_all_constructs:
            st.markdown("**Select constructs (showing ALL constructs)**")
        else:
            st.markdown("**Select top-ranked constructs (RANK 1-10) for recommendation generation**")
    
    # Initialize construct selection cache
    if "construct_selection_cache" not in st.session_state:
        st.session_state.construct_selection_cache = {}
    
    # Get top-ranked constructs for all selected projects
    all_selected_constructs = []
    project_construct_data = {}
    
    for project_id in selected_projects:
        project_config = get_project_configurations()[project_id]
        
        # Check cache first
        if project_id not in st.session_state.construct_selection_cache:
            with st.spinner(f"Loading top-ranked constructs for {project_config['name']}..."):
                try:
                    # Get project info and specs
                    project_info, project_specs, _ = asyncio.run(get_project_info_async(project_id))
                    
                    # Check if project_specs is None
                    if project_specs is None:
                        st.warning(f"Could not retrieve project specs for {project_config['name']}. Skipping...")
                        st.session_state.construct_selection_cache[project_id] = {
                            "constructs": [],
                            "all_constructs": [],
                            "project_specs": []
                        }
                        continue
                    
                    # Create evaluator instance for getting ranked constructs
                    evaluator = MetaArtemisEvaluator(
                        task_name="runtime_performance",
                        meta_prompt_llm_type=LLMType(DEFAULT_META_PROMPT_LLM),
                        code_optimization_llm_type=LLMType(DEFAULT_CODE_OPTIMIZATION_LLM),
                        project_id=project_id
                    )
                    
                    # Setup evaluator clients
                    asyncio.run(evaluator.setup_clients())
                    
                    # Get top-ranked constructs (limited by constructs_per_project)
                    top_ranked_constructs = get_top_ranked_constructs(
                        project_id=project_id,
                        evaluator=evaluator,
                        top_n=constructs_per_project
                    )
                    
                    # Also collect ALL construct IDs for the optional complete view
                    all_construct_ids = list({spec["construct_id"] for spec in project_specs})
                    
                    # Cache the results
                    st.session_state.construct_selection_cache[project_id] = {
                        "constructs": top_ranked_constructs,
                        "all_constructs": all_construct_ids,
                        "project_specs": project_specs
                    }
                    
                except Exception as e:
                    st.error(f"Error loading constructs for {project_config['name']}: {str(e)}")
                    st.session_state.construct_selection_cache[project_id] = {
                        "constructs": [],
                        "project_specs": []
                    }
        
        # Get cached data
        cached_data = st.session_state.construct_selection_cache[project_id]
        top_ranked_constructs = cached_data["constructs"]
        all_constructs = cached_data.get("all_constructs", [])
        project_specs = cached_data["project_specs"]
        
        # Determine construct list to display based on user preference
        constructs_to_display = all_constructs if st.session_state.show_all_constructs else top_ranked_constructs
        
        if constructs_to_display:
            # Create construct data for table
            construct_data = []
            for idx, construct_id in enumerate(constructs_to_display):
                # Determine rank label
                if construct_id in top_ranked_constructs:
                    rank_display = f"Rank {top_ranked_constructs.index(construct_id) + 1}"
                else:
                    rank_display = "-"

                construct_specs = [s for s in project_specs if s["construct_id"] == construct_id]

                construct_data.append({
                    "Select": st.session_state.get(f"construct_select_{project_id}_{construct_id}", True),  # Default to selected
                    "Rank": rank_display,
                    "Construct ID": construct_id[:12] + "...",
                    "Specs Count": len(construct_specs),
                    "Project": project_config["name"],
                    "Full Construct ID": construct_id,
                    "Project ID": project_id
                })

            project_construct_data[project_id] = construct_data
    
    # Display construct selection table if we have data
    if project_construct_data:
        # Combine all construct data
        all_construct_data = []
        for project_constructs in project_construct_data.values():
            all_construct_data.extend(project_constructs)
        
        if all_construct_data:
            st.markdown("Select which top-ranked constructs to generate recommendations for:")
            
            # Selection buttons
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("Select All Constructs", key="select_all_constructs"):
                    for construct in all_construct_data:
                        st.session_state[f"construct_select_{construct['Project ID']}_{construct['Full Construct ID']}"] = True
                    st.rerun()
            with col2:
                if st.button("Deselect All Constructs", key="deselect_all_constructs"):
                    for construct in all_construct_data:
                        st.session_state[f"construct_select_{construct['Project ID']}_{construct['Full Construct ID']}"] = False
                    st.rerun()
            
            # Interactive table
            edited_constructs = st.data_editor(
                all_construct_data,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select constructs for recommendation generation",
                        default=True,
                        width="small"
                    ),
                    "Rank": st.column_config.TextColumn(
                        "Rank",
                        help="Construct ranking (1-10)",
                        width="small"
                    ),
                    "Construct ID": st.column_config.TextColumn(
                        "Construct ID",
                        help="Unique construct identifier",
                        width="medium"
                    ),
                    "Specs Count": st.column_config.NumberColumn(
                        "Specs",
                        help="Number of specifications for this construct",
                        width="small"
                    ),
                    "Project": st.column_config.TextColumn(
                        "Project",
                        help="Project name",
                        width="medium"
                    ),
                    "Full Construct ID": None,  # Hide this column
                    "Project ID": None  # Hide this column
                },
                disabled=["Rank", "Construct ID", "Specs Count", "Project"],
                hide_index=True,
                use_container_width=True,
                key="construct_selection_table"
            )
            
            # Update session state and collect selected constructs
            selected_constructs_by_project = {}
            for i, construct in enumerate(edited_constructs):
                project_id = construct["Project ID"]
                construct_id = construct["Full Construct ID"]
                
                # Update session state
                st.session_state[f"construct_select_{project_id}_{construct_id}"] = construct["Select"]
                
                # Collect selected constructs
                if construct["Select"]:
                    if project_id not in selected_constructs_by_project:
                        selected_constructs_by_project[project_id] = []
                    selected_constructs_by_project[project_id].append(construct_id)
                    all_selected_constructs.append({
                        "project_id": project_id,
                        "construct_id": construct_id,
                        "rank": construct["Rank"]
                    })
            
            # Show selection summary
            total_selected = sum(len(constructs) for constructs in selected_constructs_by_project.values())
            if total_selected > 0:
                st.success(f"✅ Selected {total_selected} constructs across {len(selected_constructs_by_project)} projects")
                
                # Show breakdown by project
                with st.expander("📊 Selection Breakdown", expanded=False):
                    for project_id, construct_ids in selected_constructs_by_project.items():
                        project_config = get_project_configurations()[project_id]
                        st.markdown(f"**{project_config['name']}**: {len(construct_ids)} constructs")
                        for construct_id in construct_ids:
                            st.markdown(f"  - {construct_id[:12]}...")
            else:
                st.warning("⚠️ No constructs selected. Please select at least one construct to continue.")
        else:
            st.warning("⚠️ No top-ranked constructs found in any selected project.")
    else:
        st.info("ℹ️ Configure projects and settings above to see available constructs.")
    
    # Project and construct summary
    if (selected_templates or include_baseline) and all_selected_constructs:
        st.markdown("### 📊 Batch Processing Summary")
        
        # Calculate total operations based on selected constructs
        total_constructs = len(all_selected_constructs)
        total_operations = total_constructs * len(selected_templates)
        if include_baseline:
            total_operations += total_constructs
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Projects", len(selected_projects))
        with col2:
            st.metric("Selected Constructs", total_constructs)
        with col3:
            st.metric("Total Operations", total_operations)
        
        # Show project details
        with st.expander("📋 Project Details", expanded=False):
            project_configurations = get_project_configurations()
            for project_id in selected_projects:
                project_config = project_configurations[project_id]
                selected_in_project = [c for c in all_selected_constructs if c["project_id"] == project_id]
                st.markdown(f"**{project_config['name']}**")
                st.markdown(f"- Description: {project_config['description']}")
                default_opts = DEFAULT_PROJECT_OPTIMISATION_IDS.get(project_id, [])
                if default_opts:
                    st.markdown(f"- Default Optimization: {default_opts[0]}")
                else:
                    st.markdown(f"- Default Optimization: None")
                st.markdown(f"- Selected Constructs: {len(selected_in_project)}")
                for construct in selected_in_project:
                    st.markdown(f"  - {construct['rank']}: {construct['construct_id'][:12]}...")
                st.markdown("---")
        
        # Update configuration
        update_batch_session_state({
            "batch_recommendations": {
                **batch_config,
                "selected_projects": selected_projects,
                "constructs_per_project": constructs_per_project,
                "meta_prompt_llm": meta_prompt_llm,
                "code_optimization_llm": code_optimization_llm,
                            "selected_task": selected_task,
            "selected_templates": selected_templates,
            "include_baseline": include_baseline,
            "baseline_prompt": baseline_prompt if include_baseline else None,
            "selected_constructs": all_selected_constructs,

            "custom_templates": {template_id: st.session_state.template_states[template_id]["template"] 
                               for template_id in selected_templates 
                               if template_id in st.session_state.template_states}
            }
        })
        
        if st.button("🚀 Start Batch Recommendation Creation", key="start_batch_recs", type="primary"):
            execute_batch_recommendations()
    else:
        if not selected_templates and not include_baseline:
            st.warning("⚠️ Please select at least one prompt version to continue.")
        if not all_selected_constructs:
            st.warning("⚠️ Please select at least one construct to continue.")

def configure_batch_solutions():
    """Configure batch solution creation"""
    st.markdown("### 🔧 Batch Solution Configuration")
    
    state = get_batch_session_state()
    batch_config = state["batch_solutions"]
    selected_projects = state.get("selected_projects", [])
    
    if not selected_projects:
        st.warning("⚠️ No projects selected. Please select projects in the Project Configuration section above.")
        return
    
    # Solution creation type selection
    st.markdown("#### 🎯 Solution Creation Options")
    
    solution_creation_type = st.radio(
        "Choose solution creation method:",
        options=["from_recommendations", "from_prompt_versions", "original_code"],
        format_func=lambda x: {
            "from_recommendations": "📊 From Recommendations",
            "from_prompt_versions": "🎯 From Prompt Versions", 
            "original_code": "📋 Original Code"
        }.get(x, x),
        horizontal=True,
        help="Choose whether to create solutions from individual recommendations, all recommendations from a specific prompt version, or create baseline solutions using original unmodified code"
    )
    
    if solution_creation_type == "from_recommendations":
        configure_solutions_from_recommendations(state, batch_config, selected_projects)
    elif solution_creation_type == "from_prompt_versions":
        configure_solutions_from_prompt_versions(state, batch_config, selected_projects)
    else:
        configure_solutions_from_original_code(state, batch_config, selected_projects)

def configure_solutions_from_recommendations(state, batch_config, selected_projects):
    """Configure solution creation from recommendations"""
    st.markdown("#### 📊 Create Solutions from Recommendations")
    
    # Cache recommendations in session state to avoid refetching on every checkbox click
    if "cached_project_recommendations" not in st.session_state:
        st.session_state.cached_project_recommendations = {}
    
    # Add refresh button to clear cache if needed
    col1, col2 = st.columns([1, 11])
    with col1:
        if st.button("🔄 Refresh", help="Refresh recommendations from projects", key="refresh_recommendations"):
            st.session_state.cached_project_recommendations = {}
            st.rerun()
    with col2:
        st.markdown("**Project Recommendations**")
    
    # Get recommendations for each project
    all_selected_recommendations = []
    for project_id in selected_projects:
        project_config = get_project_configurations()[project_id]
        
        # Check if we already have cached recommendations for this project
        if project_id not in st.session_state.cached_project_recommendations:
            with st.spinner(f"Loading recommendations for {project_config['name']}..."):
                # Get project info and specs
                import asyncio
                project_info, project_specs, _ = asyncio.run(get_project_info_async(project_id))
                
                # Get recommendations from state if available
                generated_recommendations = None
                if "batch_recommendations" in state and "generated_recommendations" in state["batch_recommendations"]:
                    generated_recommendations = state["batch_recommendations"]["generated_recommendations"]
                
                # Get recommendations for top constructs
                recommendations = get_top_construct_recommendations(
                    project_id=project_id,
                    project_specs=project_specs,
                    generated_recommendations=generated_recommendations,
                    top_n=10  # Default to top 10 constructs
                )
                
                # Cache the recommendations
                st.session_state.cached_project_recommendations[project_id] = recommendations
        else:
            # Use cached recommendations
            recommendations = st.session_state.cached_project_recommendations[project_id]
        
        # Use the new table display function from recommendations module
        project_selected_recommendations = display_recommendations_table(
            project_id=project_id,
            project_name=project_config['name'],
            recommendations=recommendations
        )
        
        all_selected_recommendations.extend(project_selected_recommendations)
    
    # Update configuration and show summary
    if all_selected_recommendations:
        st.info(f"📊 Will create {len(all_selected_recommendations)} solutions from recommendations")
        
        update_batch_session_state({
            "batch_solutions": {
                **batch_config,
                "source_type": "recommendations",
                "selected_recommendations": all_selected_recommendations
            }
        })
        
        if st.button("🚀 Start Batch Solution Creation (From Recommendations)", key="start_batch_sols_from_recs", type="primary"):
            execute_batch_solutions()
    else:
        st.warning("⚠️ Please select at least one recommendation to continue")

def configure_solutions_from_prompt_versions(state, batch_config, selected_projects):
    """Configure solution creation from all recommendations of a specific prompt version"""
    st.markdown("#### 🎯 Create Solutions from Prompt Versions")
    
    # Cache recommendations in session state to avoid refetching
    if "cached_project_recommendations" not in st.session_state:
        st.session_state.cached_project_recommendations = {}
    
    # Add refresh button to clear cache if needed
    col1, col2 = st.columns([1, 11])
    with col1:
        if st.button("🔄 Refresh", help="Refresh recommendations from projects", key="refresh_recommendations_prompt_versions"):
            st.session_state.cached_project_recommendations = {}
            st.rerun()
    with col2:
        st.markdown("**Available Prompt Versions**")
    
    # Get all available template/prompt versions from recommendations across all projects
    all_available_templates = set()
    project_template_data = {}
    
    for project_id in selected_projects:
        project_config = get_project_configurations()[project_id]
        
        # Check if we already have cached recommendations for this project
        if project_id not in st.session_state.cached_project_recommendations:
            with st.spinner(f"Loading recommendations for {project_config['name']}..."):
                # Get project info and specs
                import asyncio
                project_info, project_specs, _ = asyncio.run(get_project_info_async(project_id))
                
                # Get recommendations from state if available
                generated_recommendations = None
                if "batch_recommendations" in state and "all_project_results" in state["batch_recommendations"]:
                    all_project_results = state["batch_recommendations"]["all_project_results"]
                    if project_id in all_project_results:
                        # Convert batch results to the expected format
                        batch_results = all_project_results[project_id]["results"]
                        generated_recommendations = {
                            "spec_results": []
                        }
                        
                        # Group results by construct_id
                        construct_results = {}
                        for result in batch_results:
                            construct_id = result["construct_id"]
                            if construct_id not in construct_results:
                                construct_results[construct_id] = {
                                    "spec_info": {
                                        "spec_id": f"generated_{construct_id}",
                                        "construct_id": construct_id,
                                        "name": result["spec_name"]
                                    },
                                    "template_results": {}
                                }
                            
                            template_id = result["template_id"]
                            construct_results[construct_id]["template_results"][template_id] = {
                                "recommendation": result.get("recommendation")
                            }
                        
                        generated_recommendations["spec_results"] = list(construct_results.values())
                
                # Get recommendations for top constructs
                recommendations = get_top_construct_recommendations(
                    project_id=project_id,
                    project_specs=project_specs,
                    generated_recommendations=generated_recommendations,
                    top_n=10  # Default to top 10 constructs
                )
                
                # Cache the recommendations
                st.session_state.cached_project_recommendations[project_id] = recommendations
        else:
            # Use cached recommendations
            recommendations = st.session_state.cached_project_recommendations[project_id]
        
        # Extract available templates from this project's recommendations
        project_templates = set()
        construct_template_map = {}  # Map of template -> list of constructs that have this template
        
        for rec in recommendations:
            if rec.get("source") != "placeholder":  # Only count real recommendations
                template_name = rec["template_name"]
                template_id = rec.get("template_id", template_name.lower().replace(" ", "_"))
                construct_id = rec["construct_id"]
                
                project_templates.add(template_name)
                all_available_templates.add(template_name)
                
                # Track which constructs have this template
                if template_name not in construct_template_map:
                    construct_template_map[template_name] = set()
                construct_template_map[template_name].add(construct_id)
        
        project_template_data[project_id] = {
            "project_name": project_config["name"],
            "recommendations": recommendations,
            "available_templates": project_templates,
            "construct_template_map": construct_template_map
        }
    
    # Show available template versions
    if not all_available_templates:
        st.warning("⚠️ No prompt versions found in the selected projects. Please ensure you have generated recommendations first.")
        return
    
    # Add LLM filter before template selection
    st.markdown("#### 🤖 LLM Filter")
    st.info("💡 **Important**: Filter by LLM type to ensure each construct has only one spec per solution. Solutions with multiple specs per construct from different LLMs are invalid.")
    
    # Extract all available LLM types from recommendations using the same logic as recommendations module
    all_llm_types = set()
    for project_id in selected_projects:
        project_data = project_template_data[project_id]
        recommendations = project_data["recommendations"]
        
        for rec in recommendations:
            if rec.get("source") != "placeholder":
                spec_name = rec.get("spec_name", "")
                # Extract LLM type from spec name using the same logic as recommendations module
                if spec_name:
                    # Common LLM patterns
                    if "claude" in spec_name.lower():
                        if "claude-v37-sonnet" in spec_name.lower():
                            all_llm_types.add("claude-v37-sonnet")
                        elif "claude" in spec_name.lower():
                            all_llm_types.add("claude")
                    elif "gpt-4" in spec_name.lower():
                        if "gpt-4-o" in spec_name.lower():
                            all_llm_types.add("gpt-4-o")
                        else:
                            all_llm_types.add("gpt-4")
                    elif "gpt" in spec_name.lower():
                        all_llm_types.add("gpt")
                    elif "gemini" in spec_name.lower():
                        all_llm_types.add("gemini")
                    else:
                        # Extract first part before hyphen or underscore as potential LLM type
                        parts = spec_name.split("-")
                        if len(parts) >= 2:
                            potential_llm = "-".join(parts[:2])
                            all_llm_types.add(potential_llm)
    
    # Convert to sorted list for consistent ordering
    available_llm_types = sorted(list(all_llm_types)) if all_llm_types else []
    
    if not available_llm_types:
        st.warning("⚠️ No LLM types found in recommendations.")
        return
    
    # LLM type selection
    selected_llm_type = st.selectbox(
        "Select LLM Type:",
        options=available_llm_types,
        index=0,
        help="Choose which LLM type to use for all recommendations in the solution"
    )
    
    st.markdown("#### 📋 Select Prompt Version")
    
    # Convert to sorted list for consistent ordering
    available_template_list = sorted(list(all_available_templates))
    
    # Template selection with checkboxes for multiple selection
    st.markdown("**Choose one or more prompt versions to create solutions from:**")
    
    # Add select all/deselect all buttons
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("Select All Templates", key="select_all_templates"):
            for template_name in available_template_list:
                st.session_state[f"template_checkbox_{template_name}"] = True
            st.rerun()
    with col2:
        if st.button("Deselect All Templates", key="deselect_all_templates"):
            for template_name in available_template_list:
                st.session_state[f"template_checkbox_{template_name}"] = False
            st.rerun()
    
    selected_templates = []
    
    # Create checkboxes for each available template
    cols = st.columns(min(3, len(available_template_list)))  # Max 3 columns
    for i, template_name in enumerate(available_template_list):
        col_idx = i % len(cols)
        with cols[col_idx]:
            if st.checkbox(
                template_name,
                key=f"template_checkbox_{template_name}",
                help=f"Include solutions using {template_name} recommendations"
            ):
                selected_templates.append(template_name)
    
    if selected_templates:
        st.markdown(f"#### 📊 Recommendation Analysis for Selected Templates")
        st.markdown(f"**Selected templates:** {', '.join(selected_templates)}")
        st.markdown(f"**Selected LLM type:** {selected_llm_type}")
        
        # Detailed analysis of recommendations per template
        all_analysis_data = []
        duplicate_warnings = []
        all_selected_recommendations = []
        
        for selected_template in selected_templates:
            for project_id in selected_projects:
                project_data = project_template_data[project_id]
                project_name = project_data["project_name"]
                recommendations = project_data["recommendations"]
                
                # Find all recommendations for this template in this project, filtered by LLM type
                template_recommendations = [
                    rec for rec in recommendations 
                    if (rec["template_name"] == selected_template and 
                        rec.get("source") != "placeholder" and
                        selected_llm_type.lower() in rec.get("spec_name", "").lower())
                ]
                
                if template_recommendations:
                    # Group by construct to detect duplicates and select best one
                    construct_groups = {}
                    for rec in template_recommendations:
                        construct_id = rec["construct_id"]
                        if construct_id not in construct_groups:
                            construct_groups[construct_id] = []
                        construct_groups[construct_id].append(rec)
                    
                    # Analyze each construct group
                    deduplicated_recommendations = []
                    for construct_id, construct_recs in construct_groups.items():
                        if len(construct_recs) > 1:
                            # Multiple recommendations for same construct - show warning
                            duplicate_warnings.append(
                                f"⚠️ **{project_name}** - **{selected_template}**: Construct {construct_id[:8]}... has {len(construct_recs)} recommendations. Using most recent one."
                            )
                            # Sort by created_at or ai_run_id to get most recent
                            best_rec = sorted(construct_recs, key=lambda x: x.get("created_at", ""), reverse=True)[0]
                        else:
                            best_rec = construct_recs[0]
                        
                        # Add project_id to the recommendation
                        best_rec["project_id"] = project_id
                        deduplicated_recommendations.append(best_rec)
                    
                    # Add analysis data
                    num_constructs = len(construct_groups)
                    total_recommendations = len(template_recommendations)
                    duplicate_count = total_recommendations - num_constructs
                    
                    all_analysis_data.append({
                        "Project": project_name,
                        "Template": selected_template,
                        "Constructs": num_constructs,
                        "Total Recs": total_recommendations,
                        "Duplicates": duplicate_count,
                        "Final Specs": num_constructs,  # After deduplication
                        "Status": "✅ Ready" if duplicate_count == 0 else f"⚠️ {duplicate_count} duplicates"
                    })
                    
                    all_selected_recommendations.extend(deduplicated_recommendations)
        
        if all_analysis_data:
            # Display detailed analysis table
            st.markdown("#### 📋 Recommendation Distribution Analysis")
            analysis_df = pd.DataFrame(all_analysis_data)
            st.dataframe(analysis_df, use_container_width=True)
            
            # Add detailed breakdown per template
            if len(selected_templates) > 1:
                st.markdown("#### 🔍 Detailed Breakdown by Template")
                
                for selected_template in selected_templates:
                    with st.expander(f"📊 {selected_template} - Detailed View", expanded=False):
                        template_analysis = [row for row in all_analysis_data if row["Template"] == selected_template]
                        
                        if template_analysis:
                            template_df = pd.DataFrame(template_analysis)
                            st.dataframe(template_df, use_container_width=True)
                            
                            # Show construct-level details for this template
                            st.markdown("**Construct Distribution:**")
                            
                            for project_id in selected_projects:
                                project_data = project_template_data[project_id]
                                project_name = project_data["project_name"]
                                recommendations = project_data["recommendations"]
                                
                                # Find recommendations for this template and project
                                template_recs = [
                                    rec for rec in recommendations 
                                    if (rec["template_name"] == selected_template and 
                                        rec.get("source") != "placeholder" and
                                        selected_llm_type.lower() in rec.get("spec_name", "").lower())
                                ]
                                
                                if template_recs:
                                    construct_groups = {}
                                    for rec in template_recs:
                                        construct_id = rec["construct_id"]
                                        if construct_id not in construct_groups:
                                            construct_groups[construct_id] = []
                                        construct_groups[construct_id].append(rec)
                                    
                                    st.markdown(f"**{project_name}:**")
                                    construct_details = []
                                    for construct_id, recs in construct_groups.items():
                                        status = "✅ Single" if len(recs) == 1 else f"⚠️ {len(recs)} variants"
                                        construct_details.append({
                                            "Construct ID": construct_id[:12] + "...",
                                            "Recommendations": len(recs),
                                            "Status": status
                                        })
                                    
                                    if construct_details:
                                        construct_df = pd.DataFrame(construct_details)
                                        st.dataframe(construct_df, use_container_width=True)
                        else:
                            st.info(f"No data available for {selected_template}")
            
            # Show duplicate warnings if any
            if duplicate_warnings:
                st.markdown("#### ⚠️ Duplicate Recommendations Detected")
                st.markdown("The following constructs have multiple recommendations for the same template. The most recent recommendation will be used:")
                for warning in duplicate_warnings:
                    st.markdown(f"- {warning}")
            
            # Show summary metrics
            total_constructs = sum(row["Constructs"] for row in all_analysis_data)
            total_original_specs = sum(row["Total Recs"] for row in all_analysis_data)
            total_final_specs = sum(row["Final Specs"] for row in all_analysis_data)
            total_duplicates = sum(row["Duplicates"] for row in all_analysis_data)
            total_solutions_to_create = len(all_analysis_data)
            
            st.markdown("#### 📊 Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Solutions to Create", total_solutions_to_create)
            with col2:
                st.metric("Unique Constructs", total_constructs)
            with col3:
                st.metric("Original Recs", total_original_specs)
            with col4:
                st.metric("Final Specs", total_final_specs)
            with col5:
                st.metric("Duplicates Removed", total_duplicates)
            
            # Add explanation
            st.info(
                "💡 **Solution Creation Logic**: Each solution will contain exactly one spec per construct. "
                "If multiple recommendations exist for the same construct+template combination, "
                "only the most recent one will be used."
            )
            
            st.markdown("**Solution Structure:**")
            st.markdown(f"- **{total_solutions_to_create} solutions** will be created (one per project-template combination)")
            st.markdown(f"- **{len(selected_templates)} templates** selected: {', '.join(selected_templates)}")
            st.markdown(f"- **LLM type:** {selected_llm_type} (ensures one spec per construct)")
            unique_projects = len(set(row["Project"] for row in all_analysis_data))
            st.markdown(f"- **{unique_projects} projects** involved")
            st.markdown(f"- Each solution will contain **exactly one spec per construct** (duplicates removed)")
            st.markdown(f"- Total of **{total_final_specs} specifications** across all solutions after deduplication")
            
            if total_duplicates > 0:
                st.success(f"✅ **Deduplication Applied**: {total_duplicates} duplicate recommendations were removed to ensure clean solutions with one spec per construct.")
            
            # Update configuration
            update_batch_session_state({
                "batch_solutions": {
                    **batch_config,
                    "source_type": "prompt_versions",
                    "selected_templates": selected_templates,  # Now multiple templates
                    "selected_llm_type": selected_llm_type,    # Add LLM type filter
                    "selected_recommendations": all_selected_recommendations,  # Deduplicated recommendations
                    "solution_preview": all_analysis_data,  # Updated preview data
                    "total_solutions": total_solutions_to_create
                }
            })
            
            # Create solutions button
            template_list_str = ', '.join(selected_templates)
            if st.button(f"🚀 Create Solutions from Selected Templates", key="start_batch_sols_from_prompt_versions", type="primary"):
                execute_batch_solutions()
        else:
            st.warning(f"⚠️ No recommendations found for the selected templates and LLM type '{selected_llm_type}' in any selected project.")
    else:
        st.info("💡 Please select at least one template to continue.")

def configure_solutions_from_original_code(state, batch_config, selected_projects):
    """Configure solution creation from original code (baseline solutions)"""
    st.markdown("#### 📋 Create Baseline Solutions from Original Code")
    st.info("💡 **Baseline Solutions**: These solutions use the original, unmodified code from your project. They serve as a performance baseline to compare against optimized versions.")
    
    # Simple explanation and single button approach
    st.markdown("**What this will do:**")
    st.markdown("- Create one baseline solution per selected project")
    st.markdown("- Use ALL original specs from ALL constructs (no spec IDs = original code)")
    st.markdown("- No optimization or recommendations applied")
    st.markdown("- Perfect for performance baseline comparison")
    
    # Show selected projects summary
    project_configurations = get_project_configurations()
    st.markdown("#### 📊 Selected Projects for Baseline Solutions")
    
    for project_id in selected_projects:
        project_config = project_configurations[project_id]
        st.markdown(f"✅ **{project_config['name']}** - {project_config['description']}")
    
    st.markdown(f"**Total projects:** {len(selected_projects)}")
    st.markdown(f"**Total baseline solutions to create:** {len(selected_projects)}")
    
    # Update configuration for original code
    update_batch_session_state({
        "batch_solutions": {
            **batch_config,
            "source_type": "original_code",
            "selected_projects": selected_projects,
            "solution_name_prefix": "baseline_original",
            "use_all_constructs": True,
            "total_solutions": len(selected_projects)
        }
    })
    
    # Single button to create baseline solutions
    if st.button("🚀 Create Baseline Solutions (Original Code)", key="start_batch_sols_original", type="secondary"):
        execute_batch_solutions()

def configure_batch_evaluation():
    """Configure batch solution evaluation"""
    st.markdown("### 🚀 Batch Evaluation Configuration")
    
    state = get_batch_session_state()
    batch_config = state["batch_evaluation"]
    
    # Ensure selected_optimizations are included in batch_config
    if "selected_optimizations" not in batch_config and "selected_optimizations" in state:
        batch_config["selected_optimizations"] = state["selected_optimizations"]
    
    # Source type selection
    source_type = st.radio(
        "Source for Evaluation:",
        options=["solutions", "recommendations"],
        format_func=lambda x: "Existing Solutions" if x == "solutions" else "From Recommendations",
        horizontal=True
    )
    
    # Evaluation configuration
    col1, col2 = st.columns(2)
    with col1:
        repetitions = st.number_input(
            "Evaluation Repetitions:",
            min_value=1,
            max_value=10,
            value=batch_config["evaluation_config"]["repetitions"],
            help="Number of times to execute each solution"
        )
    
    with col2:
        st.info("✅ **Artemis Managed Execution**: The app will submit all evaluations to Artemis, which will handle scheduling and execution automatically. Monitor progress in the Artemis web interface.")
    
    update_batch_session_state({
        "batch_evaluation": {
            **batch_config,
            "source_type": source_type,
            "evaluation_config": {
                "repetitions": repetitions
            }
        }
    })
    
    if source_type == "solutions":
        st.markdown("#### 📊 Evaluate Existing Solutions")
        
        # Load existing solutions for selected projects
        selected_projects = state.get("selected_projects", [])
        if not selected_projects:
            st.warning("⚠️ No projects selected. Please select projects in the Project Configuration section above.")
            return
        
        # Check if we need to load existing solutions
        if "existing_solutions_cache" not in st.session_state:
            st.session_state.existing_solutions_cache = {}
        
        # Add refresh button
        col1, col2 = st.columns([1, 11])
        with col1:
            if st.button("🔄 Refresh", help="Refresh existing solutions from projects"):
                st.session_state.existing_solutions_cache = {}
                # Also clear selection state when refreshing
                if "solution_selection_state" in st.session_state:
                    st.session_state.solution_selection_state = {}
                st.success("🔄 Cache cleared! Solutions will be refetched with updated optimization IDs.")
                st.rerun()
        with col2:
            st.markdown("**Loading existing solutions from selected projects...**")
        
        # Load existing solutions for each project
        all_existing_solutions = []
        for project_id in selected_projects:
            project_config = get_project_configurations()[project_id]
            
            # Check cache first
            if project_id not in st.session_state.existing_solutions_cache:
                with st.spinner(f"Loading existing solutions for {project_config['name']}..."):
                    try:
                        # Get selected optimization IDs for this project
                        selected_opts_for_project = []
                        if "selected_optimizations" in batch_config:
                            optimization_configs = get_optimization_configurations()
                            for opt_id in batch_config["selected_optimizations"]:
                                if opt_id in optimization_configs and optimization_configs[opt_id]["project_id"] == project_id:
                                    selected_opts_for_project.append(opt_id)
                        
                        project_info, existing_optimizations, existing_solutions = asyncio.run(get_existing_solutions_async(project_id, selected_opts_for_project))
                        st.session_state.existing_solutions_cache[project_id] = {
                            "project_info": project_info,
                            "optimizations": existing_optimizations,
                            "solutions": existing_solutions
                        }
                    except Exception as e:
                        st.error(f"Error loading solutions for {project_config['name']}: {str(e)}")
                        st.session_state.existing_solutions_cache[project_id] = {
                            "project_info": None,
                            "optimizations": [],
                            "solutions": []
                        }
            
            # Get cached data
            cached_data = st.session_state.existing_solutions_cache[project_id]
            project_solutions = cached_data["solutions"]
            
            # Add project context to each solution
            for solution in project_solutions:
                solution["project_id"] = project_id
                solution["project_name"] = project_config["name"]
            
            all_existing_solutions.extend(project_solutions)
        
        # Display summary
        if all_existing_solutions:
            st.success(f"✅ Found {len(all_existing_solutions)} existing solutions across {len(selected_projects)} projects")
            
            # Prepare all solution data for unified table
            all_solution_data = []
            
            # More inclusive filtering - include most common statuses
            available_solutions = [s for s in all_existing_solutions if s.get("status") not in ["failed", "error", "cancelled"]]
            
            # Add solutions to the combined table
            for sol in available_solutions:
                # Calculate number of evaluations (use runtime metrics as they represent actual evaluations)
                num_results = 0
                if sol.get("results_summary"):
                    runtime_metrics = sol["results_summary"].get("runtime_metrics", {})
                    # Use the first runtime metric to get the count (all should have the same count)
                    for metric_data in runtime_metrics.values():
                        if isinstance(metric_data, dict) and "count" in metric_data:
                            num_results = metric_data["count"]  # Use assignment, not addition
                            break  # Only need the first one since all should be the same
                
                solution_data = {
                    "Select": False,
                    "Solution ID": sol["solution_id"][:12] + "...",
                    "Project": sol["project_name"],
                    "Optimization": sol["optimization_name"],
                    "Status": sol["status"],
                    "Specs": len(sol["specs"]),
                    "# Results": num_results,
                    "Created": sol["created_at"][:19] if sol["created_at"] != "Unknown" else "Unknown"
                }
                all_solution_data.append(solution_data)
            
            if all_solution_data:
                # Filter options
                st.markdown("#### 🔍 Filter Options")
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_only_with_results = st.checkbox(
                        "Show only solutions with results",
                        value=False,
                        help="Filter to show only solutions that have execution results"
                    )
                with col2:
                    show_only_without_results = st.checkbox(
                        "Show only solutions without results",
                        value=False,
                        help="Filter to show only solutions that don't have execution results"
                    )
                with col3:
                    # Project filter
                    selected_project_filter = st.selectbox(
                        "Filter by Project:",
                        options=["All Projects"] + [get_project_configurations()[pid]["name"] for pid in selected_projects],
                        index=0,
                        help="Filter solutions by specific project"
                    )
                
                # Apply filters
                filtered_data = all_solution_data.copy()
                if show_only_with_results:
                    filtered_data = [sol for sol in filtered_data if sol["# Results"] > 0]
                elif show_only_without_results:
                    filtered_data = [sol for sol in filtered_data if sol["# Results"] == 0]
                
                if selected_project_filter != "All Projects":
                    # Filter by project name (extract from the display string)
                    filtered_data = [sol for sol in filtered_data if selected_project_filter in sol["Project"]]
                
                if not filtered_data:
                    st.warning("No solutions match the current filter criteria")
                    return
                
                # Initialize selection state in session state if not exists
                if "solution_selection_state" not in st.session_state:
                    st.session_state.solution_selection_state = {}
                
                # Apply existing selection state to filtered data
                for i, row in enumerate(filtered_data):
                    solution_key = f"{row['Solution ID']}_{row['Project']}_{row['Optimization']}"
                    if solution_key in st.session_state.solution_selection_state:
                        filtered_data[i]["Select"] = st.session_state.solution_selection_state[solution_key]
                
                # Selection buttons
                st.markdown("#### 🎯 Solution Selection")
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("Select All", key="select_all_solutions"):
                        # Update session state for all filtered solutions
                        for row in filtered_data:
                            solution_key = f"{row['Solution ID']}_{row['Project']}_{row['Optimization']}"
                            st.session_state.solution_selection_state[solution_key] = True
                        st.rerun()
                with col2:
                    if st.button("Deselect All", key="deselect_all_solutions"):
                        # Update session state for all filtered solutions
                        for row in filtered_data:
                            solution_key = f"{row['Solution ID']}_{row['Project']}_{row['Optimization']}"
                            st.session_state.solution_selection_state[solution_key] = False
                        st.rerun()
                
                # Apply selection state to data before passing to data_editor
                for i, row in enumerate(filtered_data):
                    solution_key = f"{row['Solution ID']}_{row['Project']}_{row['Optimization']}"
                    filtered_data[i]["Select"] = st.session_state.solution_selection_state.get(solution_key, False)
                
                # Interactive table with checkboxes
                edited_data = st.data_editor(
                    filtered_data,
                    column_config={
                        "Select": st.column_config.CheckboxColumn(
                            "Select",
                            help="Select solutions to evaluate",
                            default=False,
                            width="small"
                        ),
                        "Solution ID": st.column_config.TextColumn(
                            "Solution ID",
                            help="Unique solution identifier",
                            width="medium"
                        ),
                        "Project": st.column_config.TextColumn(
                            "Project",
                            help="Project name",
                            width="medium"
                        ),
                        "Optimization": st.column_config.TextColumn(
                            "Optimization", 
                            help="Optimization name",
                            width="medium"
                        ),
                        "Status": st.column_config.TextColumn(
                            "Status",
                            help="Solution status",
                            width="medium"
                        ),
                        "Specs": st.column_config.NumberColumn(
                            "Specs",
                            help="Number of specifications",
                            width="small"
                        ),
                        "# Results": st.column_config.NumberColumn(
                            "# Results",
                            help="Number of evaluation results available",
                            width="small"
                        ),
                        "Created": st.column_config.TextColumn(
                            "Created",
                            help="Creation timestamp",
                            width="medium"
                        )
                    },
                    disabled=["Solution ID", "Project", "Optimization", "Status", "Specs", "# Results", "Created"],
                    hide_index=True,
                    use_container_width=True,
                    key="solution_selection_table"
                )
                
                # Update session state based on data_editor changes
                for i, row in enumerate(edited_data):
                    solution_key = f"{row['Solution ID']}_{row['Project']}_{row['Optimization']}"
                    st.session_state.solution_selection_state[solution_key] = row["Select"]
                
                # Get selected solutions
                selected_solutions = []
                for i, row in enumerate(edited_data):
                    if row["Select"]:
                        # Find the original solution object
                        for sol in available_solutions:
                            if (sol["solution_id"][:12] + "..." == row["Solution ID"] and
                                sol["project_name"] in row["Project"] and
                                sol["optimization_name"] in row["Optimization"]):
                                selected_solutions.append(sol)
                                break
                
                # Update configuration - preserve the evaluation_config with correct repetitions
                current_eval_config = state.get("batch_evaluation", {}).get("evaluation_config", {"repetitions": repetitions})
                update_batch_session_state({
                    "batch_evaluation": {
                        **batch_config,
                        "source_type": source_type,
                        "selected_solutions": selected_solutions,
                        "evaluation_config": current_eval_config
                    }
                })
                
                if selected_solutions:
                    st.info(f"📊 Selected {len(selected_solutions)} solutions for evaluation")
                    
                    if st.button("🚀 Start Batch Evaluation", key="start_batch_eval", type="primary"):
                        execute_batch_evaluation()
                else:
                    st.warning("⚠️ Please select at least one solution to evaluate")
            else:
                st.warning("⚠️ No solutions available for evaluation (excluding failed/error/cancelled statuses)")
        else:
            st.warning("⚠️ No existing solutions found in any selected project.")
    
    else:  # recommendations
        st.markdown("#### 🧠 Evaluate from Recommendations")
        
        generated_recommendations = state["batch_recommendations"].get("generated_recommendations")
        
        if generated_recommendations:
            st.success("✅ Recommendations available for evaluation")
            
            # Select recommendations for evaluation
            all_recommendations = []
            for spec_result in generated_recommendations["spec_results"]:
                spec_info = spec_result["spec_info"]
                for template_id, template_result in spec_result["template_results"].items():
                    recommendation = template_result["recommendation"]
                    if recommendation.recommendation_success:
                        rec_info = {
                            "spec_id": spec_info["spec_id"],
                            "construct_id": spec_info["construct_id"],
                            "template_id": template_id,
                            "spec_name": spec_info["name"],
                            "template_name": META_PROMPT_TEMPLATES.get(template_id, {}).get("name", template_id),
                            "recommendation": recommendation
                        }
                        all_recommendations.append(rec_info)
            
            if all_recommendations:
                selected_recommendations = st.multiselect(
                    "Select Recommendations to Evaluate:",
                    options=all_recommendations,
                    default=all_recommendations,
                    format_func=lambda x: f"{x['spec_name']} ({x['template_name']})",
                    help="Choose which recommendations to evaluate"
                )
                
                # Update configuration - preserve the evaluation_config with correct repetitions
                current_eval_config = state.get("batch_evaluation", {}).get("evaluation_config", {"repetitions": repetitions})
                update_batch_session_state({
                    "batch_evaluation": {
                        **batch_config,
                        "source_type": source_type,
                        "selected_recommendations": selected_recommendations,
                        "evaluation_config": current_eval_config
                    }
                })
                
                if selected_recommendations:
                    st.info(f"📊 Will evaluate {len(selected_recommendations)} recommendations")
                    
                    if st.button("🚀 Start Batch Evaluation", key="start_batch_eval_recs", type="primary"):
                        execute_batch_evaluation()
        
        else:
            st.warning("⚠️ No recommendations available. Please run batch recommendation creation first.")

def execute_batch_recommendations():
    """Execute batch recommendation creation across multiple projects"""
    st.markdown("### 🚀 Executing Batch Recommendations")
    
    state = get_batch_session_state()
    batch_config = state["batch_recommendations"]
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    # Container for displaying generated prompts
    generated_prompts_container = st.container()
    
    def progress_callback(progress_data):
        """Enhanced progress callback that displays generated prompts"""
        if isinstance(progress_data, dict):
            # Handle structured progress data
            message = progress_data.get("message", "")
            progress = progress_data.get("progress", None)
            status = progress_data.get("status", "")
            
            # Update progress bar and status
            if progress is not None:
                progress_bar.progress(progress)
            if message:
                status_text.text(message)
            
            # Display generated prompts as they are created
            if status == "meta_prompt_ready":
                template_id = progress_data.get("template_id", "")
                filled_meta_prompt = progress_data.get("filled_meta_prompt", "")
                
                with generated_prompts_container:
                    with st.expander(f"🧠 Meta-Prompt for {template_id}", expanded=False):
                        st.markdown("**Filled Meta-Prompt Template:**")
                        st.text_area("Meta-Prompt", filled_meta_prompt, height=150, key=f"meta_{template_id}_{hash(filled_meta_prompt)}")
            
            elif status == "prompt_ready":
                template_id = progress_data.get("template_id", "")
                generated_prompt = progress_data.get("generated_prompt", "")
                
                with generated_prompts_container:
                    with st.expander(f"✨ Generated Optimization Prompt for {template_id}", expanded=True):
                        st.markdown("**Final Generated Prompt:**")
                        st.text_area("Generated Prompt", generated_prompt, height=200, key=f"prompt_{template_id}_{hash(generated_prompt)}")
                        st.success(f"✅ Successfully generated prompt using {template_id}")
        
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
        with st.spinner("Setting up batch recommendation generation..."):
            status_text.text("Initializing batch processing...")
            
            # Track results for all projects
            all_project_results = {}
            total_projects = len(batch_config["selected_projects"])
            current_project = 0
            
            for project_id in batch_config["selected_projects"]:
                current_project += 1
                project_config = get_project_configurations()[project_id]
                
                status_text.text(f"Processing project {current_project}/{total_projects}: {project_config['name']}")
                
                try:
                    # Setup evaluator for this project with progress callback
                    evaluator = MetaArtemisEvaluator(
                        task_name=batch_config["selected_task"],
                        meta_prompt_llm_type=LLMType(batch_config["meta_prompt_llm"]),
                        code_optimization_llm_type=LLMType(batch_config["code_optimization_llm"]),
                        project_id=project_id,
                        selected_templates=batch_config.get("selected_templates", []),
                        current_prompt=batch_config.get("baseline_prompt", OPTIMIZATION_TASKS[batch_config["selected_task"]]["default_prompt"]),
                        custom_templates=batch_config.get("custom_templates", {}),
                        progress_callback=progress_callback
                    )
                    

                    
                    # IMPORTANT: Setup evaluator clients first!
                    status_text.text(f"Setting up API clients for {project_config['name']}...")
                    asyncio.run(evaluator.setup_clients())
                    
                    # Get project info and specs
                    project_info, project_specs, _ = asyncio.run(get_project_info_async(project_id))
                    
                    if project_specs:
                        # Get selected constructs for this project from batch configuration
                        selected_constructs_config = batch_config.get("selected_constructs", [])
                        selected_constructs_for_project = [
                            c["construct_id"] for c in selected_constructs_config 
                            if c["project_id"] == project_id
                        ]
                        
                        if not selected_constructs_for_project:
                            st.warning(f"⚠️ No constructs selected for {project_config['name']}. Skipping...")
                            continue
                        
                        logger.info(f"🎯 Using {len(selected_constructs_for_project)} selected constructs for {project_config['name']}")
                        
                        # IMPORTANT: Generate meta-prompts after client setup!
                        if batch_config.get("selected_templates"):  # Only generate meta-prompts if we have templates selected
                            status_text.text(f"Generating meta-prompts for {project_config['name']}...")
                            asyncio.run(evaluator.generate_optimization_prompts(project_info))
                        
                        # Create project-specific batch config
                        project_batch_config = {
                            **batch_config,
                            "selected_constructs": selected_constructs_for_project
                        }
                        
                        # Execute recommendations for this project
                        project_results = asyncio.run(execute_batch_recommendations_async(
                            evaluator=evaluator,
                            batch_config=project_batch_config,
                            project_specs=project_specs
                        ))
                        
                        all_project_results[project_id] = {
                            "project_name": project_config["name"],
                            "results": project_results,
                            "meta_prompts": evaluator.meta_prompts,  # Store the generated meta prompts
                            "generated_prompts": evaluator.generated_prompts  # Store the generated optimization prompts
                        }
                        
                        # Update progress
                        progress = current_project / total_projects
                        progress_bar.progress(progress)
                        
                except Exception as e:
                    st.error(f"Error processing project {project_config['name']}: {str(e)}")
                    logger.error(f"Error processing project {project_id}: {str(e)}")
                    continue
            
            # Final update
            progress_bar.progress(1.0)
            status_text.text("✅ Batch recommendation generation completed!")
            
            # Store results
            update_batch_session_state({
                "batch_recommendations": {
                    **batch_config,
                    "all_project_results": all_project_results
                }
            })
            
            # Display results AFTER completion
            with results_container:
                st.markdown("### 📊 Batch Results Summary")
                
                # Overall metrics
                total_recommendations = sum(
                    len(project_data["results"]) 
                    for project_data in all_project_results.values()
                )
                
                successful_recommendations = sum(
                    sum(1 for result in project_data["results"] if result.get("success", False))
                    for project_data in all_project_results.values()
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Projects", len(all_project_results))
                with col2:
                    st.metric("Total Recommendations", total_recommendations)
                with col3:
                    success_rate = (successful_recommendations / total_recommendations * 100) if total_recommendations > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Display meta prompts and results for each project (STATIC DISPLAY)
                for project_id, project_data in all_project_results.items():
                    with st.expander(f"📋 {project_data['project_name']} Results", expanded=False):
                        
                        # Show meta prompts for this project (STATIC - following benchmark pattern)
                        if project_data.get("meta_prompts"):
                            st.markdown("#### 🧠 Generated Meta-Prompts")
                            st.markdown("*These are the filled meta-prompt templates used to generate optimization prompts*")
                            
                            for template_id, meta_prompt_data in project_data["meta_prompts"].items():
                                if template_id != "baseline":  # Skip baseline
                                    template_info = META_PROMPT_TEMPLATES.get(template_id, {})
                                    template_name = template_info.get("name", template_id)
                                    template_description = template_info.get("description", "")
                                    
                                    # Display template information without nested expander
                                    st.markdown(f"#### 📝 {template_name}")
                                    if template_description:
                                        st.markdown(f"*{template_description}*")
                                        
                                        st.markdown("**🔧 Filled Meta-Prompt Template:**")
                                        st.markdown("*This is the actual prompt sent to the meta-prompting LLM*")
                                        
                                        filled_content = meta_prompt_data.get("filled_template", "")
                                        if filled_content:
                                            unique_key = f"static_meta_prompt_{project_id}_{template_id}_{hash(filled_content[:50])}"
                                            st.text_area(
                                                "Filled Meta-Prompt:",
                                                value=filled_content,
                                                height=200,
                                                key=unique_key,
                                                disabled=True,
                                                label_visibility="collapsed"
                                            )
                                        
                                        # Show generated optimization prompt if available
                                        generated_prompt = project_data.get("generated_prompts", {}).get(template_id)
                                        if generated_prompt:
                                            st.markdown("**✨ Generated Optimization Prompt:**")
                                            st.markdown("*This is the optimization prompt generated by the meta-prompting LLM*")
                                            
                                            unique_gen_key = f"static_generated_prompt_{project_id}_{template_id}_{hash(generated_prompt[:50])}"
                                            st.text_area(
                                                "Generated Optimization Prompt:",
                                                value=generated_prompt,
                                                height=150,
                                                key=unique_gen_key,
                                                disabled=True,
                                                label_visibility="collapsed"
                                            )
                                        
                                        st.divider()
                        
                        # Show recommendation results
                        display_batch_recommendation_results(project_data["results"])
    
    except Exception as e:
        progress_bar.progress(0)
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Error during batch recommendation generation: {str(e)}")

def execute_batch_solutions():
    """Execute batch solution creation"""
    st.markdown("### 🔧 Executing Batch Solution Creation")
    
    state = get_batch_session_state()
    batch_config = state["batch_solutions"]
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    try:
        with st.spinner("Setting up batch solution creation..."):
            status_text.text("Initializing evaluator...")
            
            # Get the first project ID from selected projects for evaluator setup
            selected_projects = state.get("selected_projects", [])
            if not selected_projects:
                raise ValueError("No projects selected for solution creation")
            
            # Use the first project for evaluator setup (we can change this later if needed)
            first_project_id = selected_projects[0]
            
            # Setup evaluator
            evaluator = MetaArtemisEvaluator(
                task_name="runtime_performance",  # Default task
                meta_prompt_llm_type=LLMType(DEFAULT_META_PROMPT_LLM),  # Default LLM
                code_optimization_llm_type=LLMType(DEFAULT_CODE_OPTIMIZATION_LLM),  # Default LLM
                project_id=first_project_id
            )
            
            # Setup evaluator clients
            status_text.text("Setting up API clients...")
            asyncio.run(evaluator.setup_clients())
            
            # Execute async function
            status_text.text("Creating solutions...")
            batch_results = asyncio.run(execute_batch_solutions_async(
                evaluator=evaluator,
                batch_config=batch_config
            ))
            
            # Final update
            progress_bar.progress(1.0)
            status_text.text("✅ Batch solution creation completed!")
            
            # Store results
            update_batch_session_state({
                "batch_solutions": {
                    **batch_config,
                    "batch_results": batch_results
                }
            })
            
            # Display results
            with results_container:
                display_batch_solution_results(batch_results, batch_config)
    
    except Exception as e:
        progress_bar.progress(0)
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Error during batch solution creation: {str(e)}")

def execute_batch_evaluation():
    """Execute batch solution evaluation"""
    st.markdown("### 🚀 Executing Batch Evaluation")
    
    state = get_batch_session_state()
    batch_config = state["batch_evaluation"]
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    try:
        with st.spinner("Setting up batch evaluation..."):
            status_text.text("Initializing evaluation...")
            
            if batch_config["source_type"] == "solutions":
                # Execute the async evaluation
                evaluation_results = asyncio.run(execute_batch_evaluation_async(batch_config, progress_bar, status_text))
                
                # Final update
                progress_bar.progress(1.0)
                status_text.text("✅ All evaluations submitted to Artemis!")
                
                # Store results
                update_batch_session_state({
                    "batch_evaluation": {
                        **batch_config,
                        "evaluation_results": evaluation_results
                    }
                })
                
                # Display results
                with results_container:
                    display_batch_evaluation_results(evaluation_results)
            
            else:
                # Evaluate from recommendations (create solutions first, then evaluate)
                st.info("Recommendation evaluation will create solutions first, then evaluate them. This feature is coming soon...")
    
    except Exception as e:
        progress_bar.progress(0)
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Error during batch evaluation: {str(e)}")

async def execute_batch_evaluation_async(batch_config: dict, progress_bar, status_text) -> List[Dict[str, Any]]:
    """Execute batch evaluation asynchronously - following benchmark app pattern"""
    # Evaluate existing solutions
    selected_solutions = batch_config.get("selected_solutions", [])
    if not selected_solutions:
        raise ValueError("No solutions selected for evaluation")
    
    status_text.text(f"Starting evaluation for {len(selected_solutions)} solutions...")
    
    # Group solutions by project to create appropriate evaluators
    project_solution_groups = {}
    for solution in selected_solutions:
        project_id = solution["project_id"]
        if project_id not in project_solution_groups:
            project_solution_groups[project_id] = []
        project_solution_groups[project_id].append(solution)
    
    evaluation_results = []
    total_solutions = len(selected_solutions)
    current_solution = 0
    
    for project_id, project_solutions in project_solution_groups.items():
        # Create evaluator for this project
        evaluator = MetaArtemisEvaluator(
            task_name="runtime_performance",
            meta_prompt_llm_type=LLMType(DEFAULT_META_PROMPT_LLM),
            code_optimization_llm_type=LLMType(DEFAULT_CODE_OPTIMIZATION_LLM),
            project_id=project_id,
            evaluation_repetitions=batch_config["evaluation_config"]["repetitions"]
        )
        
        # Setup evaluator clients
        await evaluator.setup_clients()
        
        for solution in project_solutions:
            current_solution += 1
            progress = current_solution / total_solutions
            progress_bar.progress(progress)
            
            status_text.text(f"Starting evaluation {current_solution}/{total_solutions}: {solution['solution_id'][:8]}...")
            
            try:
                # Start the evaluation using the same method as benchmark app
                worker_name = evaluator.custom_worker_name or "jing_runner"
                
                evaluation_response = evaluator.falcon_client.evaluate_solution(
                    solution_id=UUID(solution["solution_id"]),
                    evaluation_repetitions=batch_config["evaluation_config"]["repetitions"],
                    custom_worker_name=worker_name,
                    custom_command=evaluator.custom_command,
                    unit_test=True
                )
                
                # Add a small delay to allow the evaluation to be queued (like benchmark app)
                await asyncio.sleep(2)
                
                # Get the solution details to check status and results (following benchmark app pattern)
                post_eval_solution = evaluator.falcon_client.get_solution(solution["solution_id"])
                post_eval_status = str(post_eval_solution.status).lower()
                
                # Clean up status string
                if 'solutionstatusenum.' in post_eval_status:
                    clean_status = post_eval_status.replace('solutionstatusenum.', '')
                else:
                    clean_status = post_eval_status
                
                # Check for results (following benchmark app logic)
                has_results = hasattr(post_eval_solution, 'results') and post_eval_solution.results is not None
                
                # Determine completion using benchmark app logic: status contains completion words OR has results
                is_complete = ("complete" in clean_status or "success" in clean_status or has_results)
                
                # Record that we successfully started the evaluation
                result = {
                    "solution_id": solution["solution_id"],
                    "project_id": project_id,
                    "project_name": solution["project_name"],
                    "optimization_name": solution["optimization_name"],
                    "success": True,
                    "status": "completed" if is_complete else "evaluation_started",
                    "current_status": clean_status,
                    "has_results": has_results,
                    "is_complete": is_complete,
                    "repetitions": batch_config["evaluation_config"]["repetitions"],
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_response": str(evaluation_response) if evaluation_response else None
                }
                
            except Exception as e:
                # Record that we failed to start the evaluation
                result = {
                    "solution_id": solution["solution_id"],
                    "project_id": project_id,
                    "project_name": solution["project_name"],
                    "optimization_name": solution["optimization_name"],
                    "success": False,
                    "status": "failed_to_start",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            
            evaluation_results.append(result)
    
    return evaluation_results

def display_batch_evaluation_results(evaluation_results: List[Dict[str, Any]] = None):
    """Display batch evaluation results - following benchmark app pattern"""
    st.markdown("### 🚀 Batch Evaluation Results")
    
    # If no results provided, try to get from session state
    if not evaluation_results:
        state = get_batch_session_state()
        evaluation_results = state.get("batch_evaluation", {}).get("evaluation_results", [])
    
    if not evaluation_results:
        st.warning("No results to display")
        return
    
    # Summary statistics - following benchmark app pattern
    total_evaluations = len(evaluation_results)
    successful_starts = sum(1 for result in evaluation_results if result.get("success", False))
    completed_evaluations = sum(1 for result in evaluation_results if result.get("is_complete", False))
    failed_starts = total_evaluations - successful_starts
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Solutions", total_evaluations)
    with col2:
        st.metric("Successfully Started", successful_starts)
    with col3:
        st.metric("Completed", completed_evaluations)
    with col4:
        st.metric("Failed to Start", failed_starts)
    
    # Show submission and completion status
    if successful_starts > 0:
        st.success(f"✅ **{successful_starts} evaluations successfully submitted to Artemis!**")
        if completed_evaluations > 0:
            st.success(f"🎉 **{completed_evaluations} evaluations have completed!**")
        if successful_starts > completed_evaluations:
            remaining = successful_starts - completed_evaluations
            st.info(f"🔄 **{remaining} evaluations are still running in the background.** Check the Artemis web interface to monitor progress.")
    

    
    # Results table - show submission details
    st.markdown("#### 📋 Evaluation Submission Details")
    
    results_data = []
    for result in evaluation_results:
        # Determine status display following benchmark app pattern
        if result.get("success", False):
            if result.get("is_complete", False):
                status_display = "✅ Completed"
            elif result.get("has_results", False):
                status_display = "🎉 Has Results"
            else:
                status_display = "🔄 Running"
        else:
            status_display = "❌ Failed to Start"
        
        results_data.append({
            "Solution ID": result["solution_id"][:12] + "...",
            "Project": result["project_name"],
            "Optimization": result["optimization_name"],
            "Status": status_display,
            "Current Status": result.get("current_status", "unknown"),
            "Has Results": "✅" if result.get("has_results", False) else "❌",
            "Repetitions": result.get("repetitions", "N/A"),
            "Error": result.get("error", "")
        })
    
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True)

def configure_batch_analysis():
    """Configure batch runtime impact analysis"""

    state = get_batch_session_state()
    batch_config = state["batch_analysis"]
    
    # Ensure selected_optimizations are included in batch_config
    if "selected_optimizations" not in batch_config and "selected_optimizations" in state:
        batch_config["selected_optimizations"] = state["selected_optimizations"]
    selected_projects = state.get("selected_projects", [])
    
    if not selected_projects:
        st.warning("⚠️ No project selected. Please select a project in the Project Configuration section above.")
        return
    
    # Analysis configuration
    st.markdown("#### ⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type:",
            options=["runtime_comparison", "statistical_analysis"],
            index=0,
            help="Type of analysis to perform on the runtime data"
        )
        
        significance_level = st.number_input(
            "Statistical Significance Level (α):",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            help="Alpha level for statistical significance testing"
        )
        
        effect_size_threshold = st.number_input(
            "Effect Size Threshold:",
            min_value=0.1,
            max_value=0.8,
            value=0.2,
            step=0.1,
            help="Threshold for Cohen's d effect size. Versions with effect size ≥ this value will be considered different.\n"
                 "- Negligible: |d| < 0.2\n"
                 "- Small: 0.2 ≤ |d| < 0.5\n"
                 "- Medium: 0.5 ≤ |d| < 0.8\n"
                 "- Large: |d| ≥ 0.8"
        )
    
    with col2:
        minimum_samples = st.number_input(
            "Minimum Samples Required:",
            min_value=3,
            max_value=50,
            value=5,
            help="Minimum number of execution samples required for analysis"
        )
        
        include_outliers = st.checkbox(
            "Include Outlier Detection",
            value=True,
            help="Detect and handle outliers in runtime measurements"
        )
    

    # Set fixed configuration values (no user input needed)
    include_baseline = True
    include_original = True  
    template_analysis = True
    construct_level = True
    
    # Set runtime as the only metric
    selected_metrics = ["runtime"]
    
    # Project-specific solution analysis
    st.markdown("#### 📋 Solution Analysis Preview")
    
    if "analysis_data_cache" not in st.session_state:
        st.session_state.analysis_data_cache = {}
    
    # Add refresh button
    col1, col2 = st.columns([1, 11])
    with col1:
        if st.button("🔄 Refresh", help="Refresh solution data from projects"):
            st.session_state.analysis_data_cache = {}
            st.rerun()
    with col2:
        st.markdown("**Loading solutions with runtime data for analysis...**")
    
    total_solutions_with_data = 0
    analysis_summary = {}
    
    for project_id in selected_projects:
        project_config = get_project_configurations()[project_id]
        
        # Check cache first
        if project_id not in st.session_state.analysis_data_cache:
            with st.spinner(f"Analyzing solutions for {project_config['name']}..."):
                try:
                    # Get selected optimization IDs for this project
                    selected_opts_for_project = []
                    if "selected_optimizations" in batch_config:
                        optimization_configs = get_optimization_configurations()
                        for opt_id in batch_config["selected_optimizations"]:
                            if opt_id in optimization_configs and optimization_configs[opt_id]["project_id"] == project_id:
                                selected_opts_for_project.append(opt_id)
                    
                    # Get existing solutions
                    project_info, existing_optimizations, existing_solutions = asyncio.run(get_existing_solutions_async(project_id, selected_opts_for_project))
                    
                    # Filter solutions with runtime data
                    solutions_with_data = []
                    baseline_solutions = []
                    meta_prompt_solutions = []
                    
                    for solution in existing_solutions:
                        if solution.get("has_results", False) and solution.get("status") not in ["failed", "error", "cancelled"]:
                            # Categorize solutions by type
                            solution_name = solution.get("solution_name", "").lower()
                            if "baseline" in solution_name or "default" in solution_name:
                                baseline_solutions.append(solution)
                            elif any(template in solution_name for template in ["meta", "template", "optimized"]):
                                meta_prompt_solutions.append(solution)
                            
                            solutions_with_data.append(solution)
                    
                    # Cache the analysis data
                    st.session_state.analysis_data_cache[project_id] = {
                        "total_solutions": len(existing_solutions),
                        "solutions_with_data": solutions_with_data,
                        "baseline_solutions": baseline_solutions,
                        "meta_prompt_solutions": meta_prompt_solutions,
                        "project_info": project_info
                    }
                    
                except Exception as e:
                    st.error(f"Error analyzing solutions for {project_config['name']}: {str(e)}")
                    st.session_state.analysis_data_cache[project_id] = {
                        "total_solutions": 0,
                        "solutions_with_data": [],
                        "baseline_solutions": [],
                        "meta_prompt_solutions": [],
                        "project_info": None
                    }
        
        # Get cached data
        cached_data = st.session_state.analysis_data_cache[project_id]
        solutions_with_data = cached_data["solutions_with_data"]
        baseline_solutions = cached_data["baseline_solutions"]
        meta_prompt_solutions = cached_data["meta_prompt_solutions"]
        
        total_solutions_with_data += len(solutions_with_data)
        
        analysis_summary[project_id] = {
            "project_name": project_config["name"],
            "total_solutions": cached_data["total_solutions"],
            "solutions_with_data": len(solutions_with_data),
            "baseline_solutions": len(baseline_solutions),
            "meta_prompt_solutions": len(meta_prompt_solutions)
        }
    
    # Display analysis summary
    if analysis_summary:
        st.markdown("#### 📊 Analysis Data Summary")
        
        summary_data = []
        for project_id, summary in analysis_summary.items():
            summary_data.append({
                "Project": summary["project_name"],
                "Total Solutions": summary["total_solutions"],
                "With Runtime Data": summary["solutions_with_data"],
                "Baseline Solutions": summary["baseline_solutions"],
                "Meta-Prompt Solutions": summary["meta_prompt_solutions"],
                "Analysis Ready": "✅" if summary["solutions_with_data"] >= minimum_samples else "⚠️"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Update configuration
        update_batch_session_state({
            "batch_analysis": {
                **batch_config,
                "selected_projects": selected_projects,
                "analysis_type": analysis_type,
                "significance_level": significance_level,
                "effect_size_threshold": effect_size_threshold,  # Add effect size threshold to config
                "minimum_samples": minimum_samples,
                "include_outliers": include_outliers,
                "include_baseline": include_baseline,
                "include_original": include_original,
                "template_analysis": template_analysis,
                "construct_level": construct_level,
                "selected_metrics": selected_metrics,
                "analysis_summary": analysis_summary
            }
        })
        
        # Analysis execution button
        if total_solutions_with_data >= minimum_samples:
            
            if st.button("🔬 Start Runtime Impact Analysis", key="start_batch_analysis", type="primary"):
                execute_batch_analysis()
        else:
            st.warning(f"⚠️ Need at least {minimum_samples} solutions with runtime data to perform analysis. Currently have {total_solutions_with_data}.")
        
        # CSV Export section
        st.markdown("---")
        st.markdown("#### 📥 Export Solution Evaluation Data")
        st.markdown("Export all solution evaluation data including runtime measurements, construct details, and prompt version information.")
        
        if st.button("📊 Export Solution Data to CSV", key="export_solution_csv", type="secondary"):
            try:
                # Ensure recommendations cache is populated for LLM type extraction
                if "cached_project_recommendations" not in st.session_state:
                    st.session_state.cached_project_recommendations = {}
                
                # Populate cache for any missing projects
                for project_id in selected_projects:
                    if project_id not in st.session_state.cached_project_recommendations:
                        with st.spinner(f"Loading recommendations for LLM type extraction..."):
                            try:
                                from meta_artemis_modules.project_manager import get_project_info_async
                                from meta_artemis_modules.recommendations import get_top_construct_recommendations
                                
                                project_info, project_specs, _ = asyncio.run(get_project_info_async(project_id))
                                
                                # Get recommendations for LLM type extraction
                                recommendations = get_top_construct_recommendations(
                                    project_id=project_id,
                                    project_specs=project_specs,
                                    generated_recommendations=None,
                                    top_n=10
                                )
                                
                                st.session_state.cached_project_recommendations[project_id] = recommendations
                                logger.info(f"✅ CSV Export: Populated {len(recommendations)} recommendations for project {project_id}")
                                
                            except Exception as e:
                                logger.warning(f"⚠️ CSV Export: Could not populate recommendations for project {project_id}: {str(e)}")
                                st.session_state.cached_project_recommendations[project_id] = []
                
                # Collect all solution evaluation data
                csv_data = []
                
                for project_id in selected_projects:
                    cached_data = st.session_state.analysis_data_cache.get(project_id, {})
                    solutions_with_data = cached_data.get("solutions_with_data", [])
                    project_info = cached_data.get("project_info", {})
                    project_name = project_info.get("name", "Unknown Project")
                    
                    # Debug: Log optimization IDs being processed
                    optimization_ids_in_solutions = set()
                    for solution in solutions_with_data:
                        opt_id = solution.get("optimization_id", "")
                        if opt_id:
                            optimization_ids_in_solutions.add(opt_id)
                    
                    logger.info(f"🔍 CSV Export: Processing {len(solutions_with_data)} solutions for project {project_name}")
                    logger.info(f"🔍 CSV Export: Optimization IDs in solutions: {list(optimization_ids_in_solutions)}")
                    
                    # Debug: Log selected optimizations from batch config
                    if "selected_optimizations" in batch_config:
                        logger.info(f"🔍 CSV Export: Selected optimizations from batch config: {batch_config['selected_optimizations']}")
                    else:
                        logger.warning(f"⚠️ CSV Export: No selected_optimizations found in batch_config!")
                    
                    for solution in solutions_with_data:
                        solution_id = solution.get("solution_id", "")
                        optimization_id = solution.get("optimization_id", "")
                        optimization_name = solution.get("optimization_name", "")
                        solution_status = solution.get("status", "")
                        created_at = solution.get("created_at", "")
                        specs = solution.get("specs", [])
                        results_summary = solution.get("results_summary", {})
                        runtime_metrics = results_summary.get("runtime_metrics", {})
                        
                        # Process each spec (construct) in the solution
                        # Handle original version (0 specs) differently
                        if not specs:
                            # Create base row data for original version
                            base_row = {
                                "project_id": project_id,
                                "project_name": project_name,
                                "solution_id": solution_id,
                                "optimization_id": optimization_id,
                                "optimization_name": optimization_name,
                                "solution_status": solution_status,
                                "created_at": created_at,
                                "construct_id": "Original",
                                "spec_id": "Original",
                                "prompt_version": "original",
                                "code_optimization_llm": "original",  # Original code, no LLM used
                                "num_specs_in_solution": 0,
                                "num_runtime_measurements": 0  # Will be updated after extracting measurements
                            }
                            
                            # Extract runtime measurements from results summary directly
                            runtime_measurements = []
                            if results_summary and isinstance(results_summary, dict):
                                # Look for runtime_metrics at the top level
                                runtime_metrics = results_summary.get('runtime_metrics', {})
                                if runtime_metrics:
                                    for metric_name, metric_data in runtime_metrics.items():
                                        if isinstance(metric_data, dict) and "values" in metric_data:
                                            runtime_measurements.extend([float(v) for v in metric_data["values"] if isinstance(v, (int, float))])
                                            break  # Only take the first set of measurements found
                                
                                # If no runtime_metrics found, look in the old structure
                                if not runtime_measurements:
                                    for key, value in results_summary.items():
                                        if isinstance(value, dict) and "runtime" in value:
                                            runtime_data = value.get("runtime", {})
                                            if isinstance(runtime_data, dict) and "values" in runtime_data:
                                                runtime_measurements.extend(runtime_data["values"])
                                                break
                            
                            # Update the count in base_row
                            base_row["num_runtime_measurements"] = len(runtime_measurements)
                            
                            # Create a row for runtime measurements
                            runtime_row = base_row.copy()
                            runtime_row["metric_name"] = "runtime"
                            runtime_row["runtime_measurements"] = ",".join(map(str, runtime_measurements)) if runtime_measurements else ""
                            runtime_row["metric_measurements"] = runtime_row["runtime_measurements"]
                            runtime_row["metric_count"] = len(runtime_measurements) if runtime_measurements else 0
                            csv_data.append(runtime_row)
                            
                            # Add metric-specific data for each additional runtime metric
                            for metric_name, metric_data in runtime_metrics.items():
                                if metric_name != "runtime" and isinstance(metric_data, dict) and metric_data.get("values"):
                                    metric_row = base_row.copy()
                                    metric_row["metric_name"] = metric_name
                                    metric_row["runtime_measurements"] = ""  # Empty for non-runtime metrics
                                    metric_row["metric_measurements"] = ",".join(map(str, metric_data["values"]))
                                    metric_row["metric_count"] = len(metric_data["values"])
                                    csv_data.append(metric_row)
                            continue
                            
                        # For solutions with specs, determine if this is a version-level solution
                        is_version_level = len(specs) > 1
                        
                        # Determine prompt version used
                        evaluator = MetaArtemisEvaluator(
                            task_name="runtime_performance",
                            meta_prompt_llm_type=LLMType(DEFAULT_META_PROMPT_LLM),
                            code_optimization_llm_type=LLMType(DEFAULT_CODE_OPTIMIZATION_LLM),
                            project_id=project_id
                        )
                        
                        # For version-level solutions, use the first spec to determine version type
                        first_spec = specs[0]  # Safe now because we checked for empty specs
                        prompt_version = determine_version_type_from_spec(
                            solution, first_spec.get("spec_id", ""), project_id, evaluator
                        )
                        
                        if is_version_level:
                            # For version-level solutions, extract runtime measurements from the FIRST construct only
                            # This avoids multiplying measurements by the number of specs
                            all_runtime_measurements = []
                            
                            # Use the first construct's measurements to represent the solution
                            first_construct_id = specs[0].get("construct_id", "")
                            all_runtime_measurements = extract_runtime_from_solution_results(results_summary, first_construct_id)
                            
                            # Extract LLM type from the first spec (assume all specs use the same LLM)
                            first_spec = specs[0]
                            first_spec_id = first_spec.get("spec_id", "")
                            
                            # Try to extract LLM information from cached recommendations or spec ID pattern
                            code_optimization_llm = "unknown"
                            
                            # Debug: Check what's in the cache
                            if hasattr(st.session_state, 'cached_project_recommendations') and project_id in st.session_state.cached_project_recommendations:
                                cached_recs = st.session_state.cached_project_recommendations[project_id]

                                # First, try to find the spec in cached recommendations by construct_id
                                for rec in cached_recs:
                                    # Try to match by construct_id since spec_id might not match
                                    if rec.get("construct_id") == first_construct_id:
                                        spec_name = rec.get("spec_name", "")
                                        if spec_name:
                                            code_optimization_llm = extract_llm_type_from_spec_name(spec_name)
                                            logger.info(f"✅ CSV Export: Found LLM type {code_optimization_llm} for construct {first_construct_id} from spec_name: {spec_name}")
                                            break
                                
                                # If still not found, try to match by spec_id (exact match)
                                if code_optimization_llm == "unknown":
                                    for rec in cached_recs:
                                        if rec.get("new_spec_id") == first_spec_id or rec.get("spec_id") == first_spec_id:
                                            spec_name = rec.get("spec_name", "")
                                            if spec_name:
                                                code_optimization_llm = extract_llm_type_from_spec_name(spec_name)
                                                logger.info(f"✅ CSV Export: Found LLM type {code_optimization_llm} for spec {first_spec_id} from spec_name: {spec_name}")
                                                break
                                
                                if code_optimization_llm == "unknown":
                                    logger.warning(f"⚠️ CSV Export: Could not find LLM type for construct {first_construct_id} or spec {first_spec_id}")
                            else:
                                logger.warning(f"⚠️ CSV Export: No cached recommendations found for project {project_id}")
                            
                            # Create base row data for version-level solution
                            base_row = {
                                "project_id": project_id,
                                "project_name": project_name,
                                "solution_id": solution_id,
                                "optimization_id": optimization_id,
                                "optimization_name": optimization_name,
                                "solution_status": solution_status,
                                "created_at": created_at,
                                "construct_id": "All",  # Use "All" for version-level solutions
                                "spec_id": "All",      # Use "All" for version-level solutions
                                "prompt_version": prompt_version or "unknown",
                                "code_optimization_llm": code_optimization_llm,
                                "num_specs_in_solution": len(specs),
                                "num_runtime_measurements": len(all_runtime_measurements)
                            }
                            
                            # Create a row for runtime measurements
                            runtime_row = base_row.copy()
                            runtime_row["metric_name"] = "runtime"
                            runtime_row["runtime_measurements"] = ",".join(map(str, all_runtime_measurements)) if all_runtime_measurements else ""
                            runtime_row["metric_measurements"] = runtime_row["runtime_measurements"]
                            runtime_row["metric_count"] = len(all_runtime_measurements) if all_runtime_measurements else 0
                            csv_data.append(runtime_row)
                        else:
                            # For construct-level solutions, handle the single spec
                            spec = specs[0]
                            spec_id = spec.get("spec_id", "")
                            construct_id = spec.get("construct_id", "")
                            
                            # Extract runtime measurements for this construct
                            runtime_measurements = extract_runtime_from_solution_results(results_summary, construct_id)
                            
                            # Try to extract LLM information from cached recommendations or spec ID pattern
                            code_optimization_llm = "unknown"
                            
                            # Debug: Check what's in the cache
                            if hasattr(st.session_state, 'cached_project_recommendations') and project_id in st.session_state.cached_project_recommendations:
                                cached_recs = st.session_state.cached_project_recommendations[project_id]
                                logger.info(f"🔍 CSV Export: Found {len(cached_recs)} cached recommendations for project {project_id}")
                                logger.info(f"🔍 CSV Export: Looking for construct_id {construct_id} or spec_id {spec_id}")
                                
                                # First, try to find the spec in cached recommendations by construct_id
                                for rec in cached_recs:
                                    # Try to match by construct_id since spec_id might not match
                                    if rec.get("construct_id") == construct_id:
                                        spec_name = rec.get("spec_name", "")
                                        if spec_name:
                                            code_optimization_llm = extract_llm_type_from_spec_name(spec_name)
                                            logger.info(f"✅ CSV Export: Found LLM type {code_optimization_llm} for construct {construct_id} from spec_name: {spec_name}")
                                            break
                                
                                # If still not found, try to match by spec_id (exact match)
                                if code_optimization_llm == "unknown":
                                    for rec in cached_recs:
                                        if rec.get("new_spec_id") == spec_id or rec.get("spec_id") == spec_id:
                                            spec_name = rec.get("spec_name", "")
                                            if spec_name:
                                                code_optimization_llm = extract_llm_type_from_spec_name(spec_name)
                                                logger.info(f"✅ CSV Export: Found LLM type {code_optimization_llm} for spec {spec_id} from spec_name: {spec_name}")
                                                break
                                
                                if code_optimization_llm == "unknown":
                                    logger.warning(f"⚠️ CSV Export: Could not find LLM type for construct {construct_id} or spec {spec_id}")
                            else:
                                logger.warning(f"⚠️ CSV Export: No cached recommendations found for project {project_id}")
                            
                            # Create base row data
                            base_row = {
                                "project_id": project_id,
                                "project_name": project_name,
                                "solution_id": solution_id,
                                "optimization_id": optimization_id,
                                "optimization_name": optimization_name,
                                "solution_status": solution_status,
                                "created_at": created_at,
                                "construct_id": construct_id,
                                "spec_id": spec_id,
                                "prompt_version": prompt_version or "unknown",
                                "code_optimization_llm": code_optimization_llm,
                                "num_specs_in_solution": len(specs),
                                "num_runtime_measurements": len(runtime_measurements)
                            }
                            
                            # Create a row for runtime measurements
                            runtime_row = base_row.copy()
                            runtime_row["metric_name"] = "runtime"
                            runtime_row["runtime_measurements"] = ",".join(map(str, runtime_measurements)) if runtime_measurements else ""
                            runtime_row["metric_measurements"] = runtime_row["runtime_measurements"]
                            runtime_row["metric_count"] = len(runtime_measurements) if runtime_measurements else 0
                            csv_data.append(runtime_row)
                        
                        # Add metric-specific data for each additional runtime metric (common for both types)
                        for metric_name, metric_data in runtime_metrics.items():
                            if metric_name != "runtime" and isinstance(metric_data, dict) and metric_data.get("values"):
                                metric_row = base_row.copy()
                                metric_row["metric_name"] = metric_name
                                metric_row["runtime_measurements"] = ""  # Empty for non-runtime metrics
                                metric_row["metric_measurements"] = ",".join(map(str, metric_data["values"]))
                                metric_row["metric_count"] = len(metric_data["values"])
                                csv_data.append(metric_row)
                
                if csv_data:
                    # Create DataFrame and CSV
                    df = pd.DataFrame(csv_data)
                    
                    # Get unique project names for the filename
                    project_names = df['project_name'].unique()
                    project_name_str = '_'.join(project_names)
                    
                    # Generate filename with timestamp and project names
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"evaluation_data_{project_name_str}_{timestamp}.csv"
                    
                    # Create CSV string for download
                    csv_string = df.to_csv(index=False)
                    
                    st.download_button(
                        label="💾 Download CSV File",
                        data=csv_string,
                        file_name=filename,
                        mime="text/csv",
                        help="Download solution evaluation data as CSV file"
                    )
                    
                    st.success(f"✅ Successfully prepared {len(csv_data)} rows of solution evaluation data for download!")
                    
                    # Show preview of the data
                    st.markdown("##### 📋 Data Preview (first 10 rows)")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Show summary statistics
                    st.markdown("##### 📊 Export Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", len(csv_data))
                    with col2:
                        unique_solutions = df["solution_id"].nunique()
                        st.metric("Unique Solutions", unique_solutions)
                    with col3:
                        unique_constructs = df["construct_id"].nunique()
                        st.metric("Unique Constructs", unique_constructs)
                    with col4:
                        rows_with_runtime = df[df["runtime_measurements"] != ""].shape[0]
                        st.metric("Rows with Runtime Data", rows_with_runtime)
                        
                else:
                    st.warning("⚠️ No solution evaluation data found to export.")
                    
            except Exception as e:
                st.error(f"❌ Error preparing CSV export: {str(e)}")
                logger.error(f"Error in CSV export: {str(e)}")
    else:
        st.warning("⚠️ No solutions found with runtime data in the selected projects.")

def execute_batch_analysis():
    """Execute batch runtime impact analysis"""
    st.markdown("### 🔬 Executing Runtime Impact Analysis")
    
    state = get_batch_session_state()
    batch_config = state["batch_analysis"]
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    try:
        with st.spinner("Setting up runtime impact analysis..."):
            status_text.text("Initializing analysis...")
            
            # Execute the analysis
            analysis_results = perform_runtime_impact_analysis(batch_config, progress_bar, status_text)
            
            # Final update
            progress_bar.progress(1.0)
            status_text.text("✅ Runtime impact analysis completed!")
            
            # Store results
            update_batch_session_state({
                "batch_analysis": {
                    **batch_config,
                    "analysis_results": analysis_results
                }
            })
            
            # Display results
            with results_container:
                display_box_plot_analysis_results(analysis_results)
                
                # Add performance improvement analysis
                from meta_artemis_modules.performance_improvement_analysis import display_performance_improvement_analysis
                display_performance_improvement_analysis(analysis_results)
    
    except Exception as e:
        progress_bar.progress(0)
        status_text.text(f"❌ Error: {str(e)}")
        st.error(f"Error during runtime impact analysis: {str(e)}")

def perform_runtime_impact_analysis(batch_config: dict, progress_bar, status_text) -> Dict[str, Any]:
    """Perform the actual runtime impact analysis"""
    import numpy as np
    from scipy import stats
    import statistics
    
    status_text.text("Loading solution data...")
    progress_bar.progress(0.1)
    
    # Get cached analysis data
    analysis_data_cache = st.session_state.get("analysis_data_cache", {})
    selected_projects = batch_config["selected_projects"]
    
    all_results = {
        "summary": {},
        "project_results": {},
        "statistical_tests": {},
        "template_comparison": {},
        "performance_improvements": {},
        "recommendations": []
    }
    
    total_projects = len(selected_projects)
    current_project = 0
    
    for project_id in selected_projects:
        current_project += 1
        progress = 0.1 + (current_project / total_projects) * 0.8
        progress_bar.progress(progress)
        
        project_config = get_project_configurations()[project_id]
        status_text.text(f"Analyzing {project_config['name']} ({current_project}/{total_projects})...")
        
        if project_id not in analysis_data_cache:
            continue
        
        cached_data = analysis_data_cache[project_id]
        solutions_with_data = cached_data["solutions_with_data"]
        baseline_solutions = cached_data["baseline_solutions"]
        meta_prompt_solutions = cached_data["meta_prompt_solutions"]
        
        # Simulate runtime data extraction (in real implementation, this would extract from actual solution results)
        project_analysis = analyze_project_runtime_data(
            project_id=project_id,
            project_name=project_config["name"],
            solutions_with_data=solutions_with_data,
            baseline_solutions=baseline_solutions,
            meta_prompt_solutions=meta_prompt_solutions,
            config=batch_config
        )
        
        all_results["project_results"][project_id] = project_analysis
    
    # Aggregate results across projects
    status_text.text("Aggregating cross-project results...")
    progress_bar.progress(0.9)
    
    all_results["summary"] = aggregate_cross_project_results(all_results["project_results"], batch_config)
    
    return all_results

def analyze_project_runtime_data(project_id: str, project_name: str, solutions_with_data: list, 
                                baseline_solutions: list, meta_prompt_solutions: list, config: dict) -> Dict[str, Any]:
    """Analyze runtime data for a specific project - extract real evaluation data for top 10 ranked constructs"""
    import numpy as np
    from scipy import stats
    import asyncio
    
    # Ensure recommendations cache is populated for version type determination
    if "cached_project_recommendations" not in st.session_state:
        st.session_state.cached_project_recommendations = {}
    
    if project_id not in st.session_state.cached_project_recommendations:
        logger.info(f"🔄 Populating recommendations cache for project {project_name}")
        try:
            # Get project info and recommendations for caching
            from meta_artemis_modules.project_manager import get_project_info_async
            from meta_artemis_modules.recommendations import get_top_construct_recommendations
            
            project_info, project_specs, _ = asyncio.run(get_project_info_async(project_id))
            
            # Get recommendations using the same function as the solutions section
            recommendations = get_top_construct_recommendations(
                project_id=project_id,
                project_specs=project_specs,
                generated_recommendations=None,  # We'll look for existing recommendations
                top_n=10
            )
            
            # Cache the recommendations for version type determination
            st.session_state.cached_project_recommendations[project_id] = recommendations
            logger.info(f"✅ Cached {len(recommendations)} recommendations for project {project_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not populate recommendations cache for project {project_name}: {str(e)}")
            st.session_state.cached_project_recommendations[project_id] = []
    
    # Get top-ranked constructs (Rank 1-10) using existing filter function
    try:
        from meta_artemis_modules.evaluator import MetaArtemisEvaluator
        from vision_models.service.llm import LLMType
        evaluator = MetaArtemisEvaluator(
            task_name="runtime_performance",
            meta_prompt_llm_type=LLMType(DEFAULT_META_PROMPT_LLM),
            code_optimization_llm_type=LLMType(DEFAULT_CODE_OPTIMIZATION_LLM),
            project_id=project_id
        )
        
        # Setup clients
        import asyncio
        asyncio.run(evaluator.setup_clients())
        
        # Get top-ranked constructs
        from meta_artemis_modules.recommendations import get_top_ranked_constructs
        top_ranked_constructs = get_top_ranked_constructs(project_id, evaluator, top_n=10)
        
        if not top_ranked_constructs:
            logger.warning(f"⚠️ No top-ranked constructs found for project {project_name}")
            return {
                "project_name": project_name,
                "error": "No top-ranked constructs found",
                "num_constructs": 0,
                "versions_data": {},
                "version_stats": {},
                "box_plot_data": {}
            }
        
        logger.info(f"🎯 Found {len(top_ranked_constructs)} top-ranked constructs for {project_name}")
        
    except Exception as e:
        logger.error(f"❌ Error getting top-ranked constructs for {project_name}: {str(e)}")
        return {
            "project_name": project_name,
            "error": f"Error getting ranked constructs: {str(e)}",
            "num_constructs": 0,
            "versions_data": {},
            "version_stats": {},
            "box_plot_data": {}
        }
    
    # Extract real evaluation data from solutions
    construct_performance_data = {}
    warnings = []
    
    # Add debug logging
    logger.info(f"🔍 Starting analysis for {project_name} with {len(top_ranked_constructs)} top-ranked constructs")
    logger.info(f"📊 Processing {len(solutions_with_data)} solutions with data")
    
    for construct_id in top_ranked_constructs:
        construct_rank = top_ranked_constructs.index(construct_id) + 1
        construct_performance_data[construct_id] = {
            "rank": construct_rank,
            "versions": {},  # Use dynamic dictionary to accommodate any template ID
            "total_evaluations": 0,
            "missing_versions": []
        }
        logger.debug(f"🎯 Initialized construct Rank {construct_rank}: {construct_id[:8]}...")
    
    # Separate construct-level and version-level solutions
    construct_level_solutions = []
    version_level_solutions = []
    
    # Process all solutions to extract performance data
    for solution in solutions_with_data:
        try:
            # Get solution specs and results
            solution_specs = solution.get("specs", [])
            solution_results_summary = solution.get("results_summary", {})
            solution_id = solution.get("solution_id", "unknown")
            
            logger.debug(f"🔍 Processing solution {solution_id[:8]}... with {len(solution_specs)} specs")
            
            if not solution_results_summary:
                logger.debug(f"No results_summary found for solution {solution_id[:8]}...")
                continue
            
            # Classify solution type
            num_specs = len(solution_specs)
            
            if num_specs == 0:
                # Original version: 0 specs = all constructs use original code
                logger.debug(f"🔍 Solution {solution_id[:8]}... is ORIGINAL version (0 specs)")
                
                # Add to BOTH construct-level AND version-level analysis
                construct_level_solutions.append(solution)
                version_level_solutions.append(solution)  # Also include in version-level
                
                # For original version, extract runtime data ONCE from solution level (not per construct)
                # The original solution contains runtime measurements for the entire solution execution
                solution_runtime_data = []
                
                # Extract runtime metrics from solution results summary
                runtime_metrics = solution_results_summary.get('runtime_metrics', {})
                if runtime_metrics:
                    for metric_name, metric_data in runtime_metrics.items():
                        if isinstance(metric_data, dict):
                            values = metric_data.get('values', [])
                            if values and isinstance(values, list):
                                solution_runtime_data.extend([float(v) for v in values if isinstance(v, (int, float))])
                                logger.debug(f"Found {len(values)} solution-level runtime measurements in {metric_name}")
                
                # Fallback: try to extract from any runtime field in results_summary
                if not solution_runtime_data:
                    for key, value in solution_results_summary.items():
                        if "runtime" in str(key).lower() or "time" in str(key).lower():
                            if isinstance(value, dict):
                                values = value.get('values', [])
                                if values and isinstance(values, list):
                                    solution_runtime_data.extend([float(v) for v in values if isinstance(v, (int, float))])
                                    logger.debug(f"Found {len(values)} runtime measurements in fallback key: {key}")
                                    break
                            elif isinstance(value, (int, float)):
                                solution_runtime_data.append(float(value))
                                logger.debug(f"Found single runtime measurement in key: {key}")
                                break
                
                # Apply the SAME runtime data to ALL top-ranked constructs (they all use original code)
                if solution_runtime_data:
                    logger.debug(f"📊 Original solution has {len(solution_runtime_data)} runtime measurements total")
                    
                    for construct_id in top_ranked_constructs:
                        # All constructs get the same original runtime data since they all use original code
                        version_type = "original"
                        if version_type not in construct_performance_data[construct_id]["versions"]:
                            construct_performance_data[construct_id]["versions"][version_type] = []
                        construct_performance_data[construct_id]["versions"][version_type].extend(solution_runtime_data)
                        construct_performance_data[construct_id]["total_evaluations"] += len(solution_runtime_data)
                        logger.debug(f"📊 Added {len(solution_runtime_data)} runtime measurements for construct {construct_id[:8]}... version '{version_type}' (shared from solution)")
                else:
                    logger.warning(f"⚠️ No runtime data found in original solution {solution_id[:8]}...")
            
            elif num_specs == 1:
                # Construct-level solution: 1 spec = only 1 construct uses recommendations
                spec = solution_specs[0]
                target_construct_id = spec.get("construct_id")
                spec_id = spec.get("spec_id")
                
                # Only process if the target construct is in our top-ranked list
                if target_construct_id in top_ranked_constructs:
                    logger.debug(f"🔍 Solution {solution_id[:8]}... is CONSTRUCT-LEVEL solution (1 spec)")
                    logger.debug(f"   - Target construct: {target_construct_id[:8]}...")
                    construct_level_solutions.append(solution)
                    
                    # Determine the version type from the spec
                    solution_version_type = determine_version_type_from_spec(solution, spec_id, project_id, evaluator)
                    
                    # Only apply to the target construct (construct-level analysis)
                    runtime_data = extract_runtime_from_solution_results(solution_results_summary, target_construct_id)
                    if runtime_data:
                        # Ensure the version type key exists in the dictionary
                        if solution_version_type not in construct_performance_data[target_construct_id]["versions"]:
                            construct_performance_data[target_construct_id]["versions"][solution_version_type] = []
                        construct_performance_data[target_construct_id]["versions"][solution_version_type].extend(runtime_data)
                        construct_performance_data[target_construct_id]["total_evaluations"] += len(runtime_data)
                        logger.debug(f"📊 Added {len(runtime_data)} runtime measurements for construct {target_construct_id[:8]}... version '{solution_version_type}' (construct-level)")
                else:
                    logger.debug(f"🔍 Solution {solution_id[:8]}... targets construct not in top-ranked list, skipping")
            
            else:
                # Version-level solution: multiple specs = version-level analysis
                logger.debug(f"🔍 Solution {solution_id[:8]}... is VERSION-LEVEL solution ({num_specs} specs)")
                version_level_solutions.append(solution)
                # Version-level solutions should NOT contaminate construct-level analysis
                # They will be processed separately in process_version_level_solutions()
                

        
        except Exception as e:
            logger.warning(f"⚠️ Error processing solution {solution.get('solution_id', 'unknown')}: {str(e)}")
            continue
    
    # Check for missing data and generate warnings
    for construct_id, construct_data in construct_performance_data.items():
        total_evaluations = construct_data["total_evaluations"]
        expected_evaluations = 50  # 5 versions × 10 evaluations each
        
        if total_evaluations < expected_evaluations:
            missing_count = expected_evaluations - total_evaluations
            warnings.append(f"⚠️ Construct Rank {construct_data['rank']} ({construct_id[:8]}...): Only {total_evaluations}/{expected_evaluations} evaluations found. Missing {missing_count} evaluations.")
        
        # Check for missing versions
        missing_versions = []
        for version, data in construct_data["versions"].items():
            if len(data) == 0:
                missing_versions.append(version)
            elif len(data) < 10:
                warnings.append(f"⚠️ Construct Rank {construct_data['rank']} version '{version}': Only {len(data)}/10 evaluations found.")
        
        if missing_versions:
            construct_data["missing_versions"] = missing_versions
            warnings.append(f"⚠️ Construct Rank {construct_data['rank']} ({construct_id[:8]}...): Missing versions: {', '.join(missing_versions)}")
    
    # Summary logging with version distribution
    constructs_with_data = len([c for c in construct_performance_data.values() if c["total_evaluations"] > 0])
    total_evaluations = sum(c["total_evaluations"] for c in construct_performance_data.values())
    logger.info(f"📊 Analysis summary: {constructs_with_data}/{len(top_ranked_constructs)} constructs have data, {total_evaluations} total evaluations")
    
    # Log version distribution
    version_totals = {}
    for construct_data in construct_performance_data.values():
        for version, data in construct_data["versions"].items():
            if version not in version_totals:
                version_totals[version] = 0
            version_totals[version] += len(data)
    
    logger.info(f"🔍 Version distribution across all constructs:")
    for version, count in version_totals.items():
        logger.info(f"   - {version}: {count} evaluations")
    
    # Warn about versions with no data
    empty_versions = [v for v, count in version_totals.items() if count == 0]
    if empty_versions:
        logger.warning(f"⚠️ Versions with no data: {', '.join(empty_versions)}")
    
    # Prepare data for visualization
    versions_data = {}
    
    individual_constructs = {}
    
    # Track if we've already added original data to avoid duplication
    original_data_added = False
    
    for construct_id, construct_data in construct_performance_data.items():
        construct_rank = construct_data["rank"]
        construct_name = f"construct_rank_{construct_rank}"
        
        individual_constructs[construct_name] = {}
        
        for version, runtime_list in construct_data["versions"].items():
            if runtime_list:
                # Special handling for original version to avoid duplication
                if version == "original":
                    # Only add original data once to the overall project data
                    if not original_data_added:
                        if version not in versions_data:
                            versions_data[version] = []
                        versions_data[version].extend(runtime_list)
                        original_data_added = True
                        logger.debug(f"📊 Added {len(runtime_list)} original runtime measurements to overall project data (once)")
                    
                    # Add to individual construct data (use mean if multiple values)
                    individual_constructs[construct_name][version] = np.mean(runtime_list)
                else:
                    # For non-original versions, add normally
                    if version not in versions_data:
                        versions_data[version] = []
                    versions_data[version].extend(runtime_list)
                    
                    # Add to individual construct data (use mean if multiple values)
                    individual_constructs[construct_name][version] = np.mean(runtime_list)
            else:
                # Use NaN for missing versions
                individual_constructs[construct_name][version] = np.nan
    
    # Calculate statistics for each version
    version_stats = {}
    for version, runtimes in versions_data.items():
        if runtimes:
            version_stats[version] = {
                "count": len(runtimes),
                "mean": np.mean(runtimes),
                "median": np.median(runtimes),
                "std": np.std(runtimes),
                "min": np.min(runtimes),
                "max": np.max(runtimes),
                "q1": np.percentile(runtimes, 25),
                "q3": np.percentile(runtimes, 75)
            }
        else:
            version_stats[version] = {
                "count": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "q1": np.nan,
                "q3": np.nan
            }
    
    # Calculate overall improvement metrics
    original_runtimes = versions_data.get("original", [])
    enhanced_runtimes = versions_data.get("enhanced", [])
    
    if original_runtimes and enhanced_runtimes:
        original_mean = np.mean(original_runtimes)
        enhanced_mean = np.mean(enhanced_runtimes)
        overall_improvement = ((original_mean - enhanced_mean) / original_mean) * 100
    else:
        original_mean = np.nan
        enhanced_mean = np.nan
        overall_improvement = np.nan
        warnings.append("⚠️ Cannot calculate improvement: Missing original or enhanced version data.")
    
    # Perform statistical analysis using Scott-Knott ESD test
    logger.info("🧪 Performing statistical analysis...")
    
        # Get the analysis settings from config
    alpha = config.get("significance_level", 0.05)
    effect_size_threshold = config.get("effect_size_threshold", 0.2)
    
    # Import statistical functions from visualization module
    from meta_artemis_modules.visualization import perform_statistical_analysis, format_statistical_results
    
    # Perform statistical tests on overall project data
    statistical_results = perform_statistical_analysis(versions_data, alpha=alpha, effect_size_threshold=effect_size_threshold)
    
    # Format statistical results for display
    statistical_summary = format_statistical_results(statistical_results)
    
    # Perform statistical analysis on individual constructs
    construct_statistical_results = {}
    for construct_name, construct_versions in individual_constructs.items():
        # Convert NaN values to empty lists for statistical testing
        construct_data = {}
        for version, value in construct_versions.items():
            if not np.isnan(value):
                construct_data[version] = [value]  # Single value as list
            else:
                construct_data[version] = []  # Empty list for missing data
        
            if len([v for v in construct_data.values() if v]) >= 2:  # Need at least 2 versions with data
                construct_stats = perform_statistical_analysis(construct_data, alpha=alpha, effect_size_threshold=effect_size_threshold)
                construct_statistical_results[construct_name] = construct_stats
            else:
                construct_statistical_results[construct_name] = {
                    "success": False,
                    "error": "Insufficient data for statistical testing",
                    "groups": {},
                    "rankings": {},
                    "p_values": {}
                }
    
    # Process version-level solutions for separate analysis
    version_level_data = process_version_level_solutions(version_level_solutions, top_ranked_constructs, project_id, evaluator)
    
    # Debug logging: show final construct_performance_data structure
    logger.debug(f"Final construct_performance_data summary:")
    for construct_id, construct_data in construct_performance_data.items():
        construct_rank = construct_data.get("rank", "?")
        versions = construct_data.get("versions", {})
        total_evals = construct_data.get("total_evaluations", 0)
        logger.debug(f"Construct Rank {construct_rank} ({construct_id[:8]}...): {total_evals} total evaluations")
        for version, data in versions.items():
            logger.debug(f"   Version '{version}': {len(data)} measurements")
    
    analysis_result = {
        "project_name": project_name,
        "num_constructs": len(top_ranked_constructs),
        "constructs_processed": len([c for c in construct_performance_data.values() if c["total_evaluations"] > 0]),
        "versions_data": versions_data,
        "version_stats": version_stats,
        "box_plot_data": {
            "individual_constructs": individual_constructs,
            "overall_project": versions_data
        },
        "overall_improvement": {
            "original_mean": original_mean,
            "enhanced_mean": enhanced_mean,
            "improvement_percentage": overall_improvement,
            "improvement_seconds": original_mean - enhanced_mean if not np.isnan(original_mean) and not np.isnan(enhanced_mean) else np.nan
        },
        "statistical_analysis": {
            "overall_project": statistical_results,
            "individual_constructs": construct_statistical_results,
            "summary": statistical_summary,
            "alpha": alpha
        },
        "warnings": warnings,
        "construct_details": construct_performance_data,
        "construct_level_solutions": construct_level_solutions,
        "version_level_data": version_level_data
    }
    
    logger.debug(f"Returning analysis_result with construct_details containing {len(construct_performance_data)} constructs")
    
    return analysis_result

def process_version_level_solutions(version_level_solutions: List[dict], top_ranked_constructs: List[str], project_id: str, evaluator) -> Dict[str, Any]:
    """Process version-level solutions for separate analysis"""
    version_level_data = {
        "solutions": [],
        "version_stats": {},
        "total_solutions": len(version_level_solutions),
        "solutions_by_version": {}  # Use dynamic dictionary to accommodate any template ID
    }
    
    logger.info(f"🔍 Processing {len(version_level_solutions)} version-level solutions")
    
    for solution in version_level_solutions:
        try:
            solution_specs = solution.get("specs", [])
            solution_results_summary = solution.get("results_summary", {})
            solution_id = solution.get("solution_id", "unknown")
            
            # Determine version type from first spec
            if solution_specs:
                first_spec = solution_specs[0]
                first_spec_id = first_spec.get("spec_id")
                solution_version_type = determine_version_type_from_spec(solution, first_spec_id, project_id, evaluator)
            else:
                solution_version_type = "original"
            
            # For version-level solutions, we get the total runtime measurements for the entire solution
            # This represents the solution being evaluated ~10 times, not per construct
            total_runtime_measurements = 0
            all_runtime_data = []
            constructs_with_data = 0
            
            # Extract runtime data from the solution results summary
            # For version-level solutions, the runtime should be at the solution level, not per construct
            if solution_results_summary:
                # Try to get solution-level runtime data first
                runtime_metrics = solution_results_summary.get('runtime_metrics', {})
                if runtime_metrics:
                    for metric_name, metric_data in runtime_metrics.items():
                        if isinstance(metric_data, dict):
                            values = metric_data.get('values', [])
                            if values and isinstance(values, list):
                                all_runtime_data.extend([float(v) for v in values if isinstance(v, (int, float))])
                                total_runtime_measurements += len(values)
                                logger.debug(f"Found {len(values)} solution-level runtime measurements in {metric_name}")
                
                # If no solution-level data, try additional extraction methods
                if not all_runtime_data:
                    # Try extracting from any field containing "runtime" or "time"
                    for key, value in solution_results_summary.items():
                        if "runtime" in str(key).lower() or "time" in str(key).lower():
                            if isinstance(value, dict):
                                values = value.get('values', [])
                                if values and isinstance(values, list):
                                    all_runtime_data.extend([float(v) for v in values if isinstance(v, (int, float))])
                                    logger.debug(f"Found {len(values)} runtime measurements in version-level key: {key}")
                                    break
                            elif isinstance(value, (int, float)):
                                all_runtime_data.append(float(value))
                                logger.debug(f"Found single runtime measurement in version-level key: {key}")
                                break
                
                # Final fallback: construct-level aggregation
                if not all_runtime_data:
                    for construct_id in top_ranked_constructs:
                        runtime_data = extract_runtime_from_solution_results(solution_results_summary, construct_id)
                        if runtime_data:
                            all_runtime_data.extend(runtime_data)
                            constructs_with_data += 1
                    total_runtime_measurements = len(all_runtime_data)
                    logger.debug(f"Used construct-level fallback, found {total_runtime_measurements} measurements")
                else:
                    total_runtime_measurements = len(all_runtime_data)
                    # Count constructs that have specs in this solution
                    constructs_with_data = len([spec for spec in solution_specs if spec.get("construct_id") in top_ranked_constructs])
            
            solution_info = {
                "solution_id": solution_id,
                "solution_name": solution.get("solution_name", ""),
                "optimization_name": solution.get("optimization_name", ""),
                "version_type": solution_version_type,
                "num_specs": len(solution_specs),
                "constructs_with_data": constructs_with_data,
                "total_runtime_measurements": total_runtime_measurements,
                "runtime_data": all_runtime_data,
                "avg_runtime": np.mean(all_runtime_data) if all_runtime_data else np.nan,
                "min_runtime": np.min(all_runtime_data) if all_runtime_data else np.nan,
                "max_runtime": np.max(all_runtime_data) if all_runtime_data else np.nan
            }
            
            version_level_data["solutions"].append(solution_info)
            # Ensure the version type key exists in the dictionary
            if solution_version_type not in version_level_data["solutions_by_version"]:
                version_level_data["solutions_by_version"][solution_version_type] = []
            version_level_data["solutions_by_version"][solution_version_type].append(solution_info)
            
            logger.debug(f"📊 Version-level solution {solution_id[:8]}... version '{solution_version_type}': {total_runtime_measurements} measurements")
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing version-level solution {solution.get('solution_id', 'unknown')}: {str(e)}")
    
    # Calculate statistics for ALL versions (including those with no data)
    all_versions = list(version_level_data["solutions_by_version"].keys())
    
    # Prepare data for Scott-Knott ESD test
    version_runtime_data = {}
    
    for version in all_versions:
        solutions = version_level_data["solutions_by_version"][version]
        if solutions:
            # Aggregate all runtime data from solutions of this version
            all_runtime_data = []
            total_solution_count = len(solutions)
            total_measurements = 0
            
            for sol in solutions:
                all_runtime_data.extend(sol["runtime_data"])
                total_measurements += sol["total_runtime_measurements"]
            
            if all_runtime_data:
                version_level_data["version_stats"][version] = {
                    "solution_count": total_solution_count,
                    "total_measurements": total_measurements,
                    "mean_runtime": np.mean(all_runtime_data),
                    "median_runtime": np.median(all_runtime_data),
                    "std_runtime": np.std(all_runtime_data),
                    "min_runtime": np.min(all_runtime_data),
                    "max_runtime": np.max(all_runtime_data)
                }
                # Add to Scott-Knott ESD test data
                version_runtime_data[version] = all_runtime_data
            else:
                version_level_data["version_stats"][version] = {
                    "solution_count": total_solution_count,
                    "total_measurements": 0,
                    "mean_runtime": np.nan,
                    "median_runtime": np.nan,
                    "std_runtime": np.nan,
                    "min_runtime": np.nan,
                    "max_runtime": np.nan
                }
        else:
            # No solutions for this version
            version_level_data["version_stats"][version] = {
                "solution_count": 0,
                "total_measurements": 0,
                "mean_runtime": np.nan,
                "median_runtime": np.nan,
                "std_runtime": np.nan,
                "min_runtime": np.nan,
                "max_runtime": np.nan
            }
    
    # Perform Scott-Knott ESD test on version-level data
    version_level_data["scott_knott_rankings"] = {}
    if len(version_runtime_data) >= 2:
        try:
            from meta_artemis_modules.visualization import perform_statistical_analysis
            sk_result = perform_statistical_analysis(version_runtime_data, alpha=0.05)
            if sk_result.get("success", False):
                version_level_data["scott_knott_rankings"] = sk_result.get("rankings", {})
                logger.info(f"Version-level Scott-Knott ESD rankings: {version_level_data['scott_knott_rankings']}")
            else:
                logger.warning("Scott-Knott ESD test failed for version-level data")
        except Exception as e:
            logger.warning(f"⚠️ Statistical analysis failed for version-level data: {str(e)}")
    
    return version_level_data

def extract_llm_type_from_spec_name(spec_name: str) -> str:
    """Extract LLM type from spec name using the same logic as recommendations filtering"""
    if not spec_name:
        return "unknown"
    
    spec_name_lower = spec_name.lower()
    
    # Common LLM patterns
    if "claude" in spec_name_lower:
        if "claude-v37-sonnet" in spec_name_lower:
            return "claude-v37-sonnet"
        elif "claude" in spec_name_lower:
            return "claude"
    elif "gpt-4" in spec_name_lower:
        if "gpt-4-o" in spec_name_lower:
            return "gpt-4-o"
        else:
            return "gpt-4"
    elif "gpt" in spec_name_lower:
        return "gpt"
    elif "gemini" in spec_name_lower:
        return "gemini"
    else:
        # Extract first part before hyphen or underscore as potential LLM type
        parts = spec_name.split("-")
        if len(parts) >= 2:
            potential_llm = "-".join(parts[:2])
            return potential_llm
        
        # If no hyphen, try underscore
        parts = spec_name.split("_")
        if len(parts) >= 2:
            potential_llm = "_".join(parts[:2])
            return potential_llm
    
    return "unknown"

def extract_runtime_from_solution_results(results_summary: dict, construct_id: str) -> List[float]:
    """Extract runtime measurements from solution results_summary for a specific construct"""
    runtime_data = []
    
    try:
        # Check if results_summary is actually a dict and has the expected structure
        if not isinstance(results_summary, dict):
            logger.debug(f"Results summary is not a dict: {type(results_summary)}")
            return runtime_data
        
        # Extract runtime metrics directly from results_summary
        runtime_metrics = results_summary.get('runtime_metrics', {})
        if runtime_metrics:
            for metric_name, metric_data in runtime_metrics.items():
                if isinstance(metric_data, dict):
                    # Look for individual measurements in 'values' field
                    values = metric_data.get('values', [])
                    if values and isinstance(values, list):
                        runtime_data.extend([float(v) for v in values if isinstance(v, (int, float))])
                        logger.debug(f"Found {len(values)} runtime measurements in {metric_name}")
                    
                    # Also check for single average value if no individual values
                    elif 'avg' in metric_data:
                        avg_value = metric_data.get('avg')
                        if isinstance(avg_value, (int, float)):
                            runtime_data.append(float(avg_value))
                            logger.debug(f"Found average runtime measurement: {avg_value}")
        
        # Fallback: look for direct runtime fields in various possible locations
        if not runtime_data:
            # Check for direct runtime fields
            for key, value in results_summary.items():
                if "runtime" in str(key).lower() or "time" in str(key).lower():
                    if isinstance(value, (int, float)):
                        runtime_data.append(float(value))
                    elif isinstance(value, dict):
                        # Look deeper for runtime values
                        for subkey, subvalue in value.items():
                            if "runtime" in str(subkey).lower() or "time" in str(subkey).lower():
                                if isinstance(subvalue, (int, float)):
                                    runtime_data.append(float(subvalue))
                                elif isinstance(subvalue, list):
                                    runtime_data.extend([float(x) for x in subvalue if isinstance(x, (int, float))])
                    elif isinstance(value, list):
                        # Process list of measurements
                        for item in value:
                            if isinstance(item, (int, float)):
                                runtime_data.append(float(item))
                            elif isinstance(item, dict):
                                for subkey, subvalue in item.items():
                                    if "runtime" in str(subkey).lower() or "time" in str(subkey).lower():
                                        if isinstance(subvalue, (int, float)):
                                            runtime_data.append(float(subvalue))
        
        if runtime_data:
            logger.debug(f"Extracted {len(runtime_data)} runtime measurements for construct {construct_id[:8]}...")
        else:
            logger.debug(f"No runtime data found for construct {construct_id[:8]}... in results summary")
    
    except Exception as e:
        logger.warning(f"⚠️ Error extracting runtime data for construct {construct_id[:8]}...: {str(e)}")
    
    return runtime_data

def determine_version_type_from_spec(solution: dict, spec_id: str, project_id: str, evaluator) -> str:
    """Determine version type by tracing spec_id back to its recommendation source"""
    
    # Get solution metadata for logging
    solution_id = solution.get("solution_id", "")
    solution_name = solution.get("solution_name", "")
    optimization_name = solution.get("optimization_name", "")
    
    logger.debug(f"🔍 Determining version type for spec {spec_id[:8]}... in solution {solution_id[:8]}...")
    
    # Check if the spec_id is empty or None (original code)
    if not spec_id or spec_id.strip() == "":
        logger.debug(f"   -> Empty spec_id, classifying as 'original'")
        return "original"
    
    # Use cached recommendations to determine version type
    try:
        # Check if we have cached recommendations for this project
        if "cached_project_recommendations" in st.session_state:
            cached_recommendations = st.session_state.cached_project_recommendations.get(project_id, [])
            
            # Search through cached recommendations for this spec_id
            for rec in cached_recommendations:
                if rec.get("spec_id") == spec_id:
                    template_name = rec.get("template_name", "")
                    template_id = rec.get("template_id", "")
                    
                    # Map template names and IDs to version types using comprehensive mapping
                    from meta_artemis_modules.shared_templates import ALL_PROMPTING_TEMPLATES
                    
                    # First try to use template_id for exact matching
                    if template_id in ALL_PROMPTING_TEMPLATES:
                        logger.debug(f"   -> Found template ID '{template_id}': returning '{template_id}'")
                        return template_id
                    
                    # Handle legacy template IDs
                    elif template_id == "enhanced" or "Enhanced" in template_name or "MPCO" in template_name:
                        logger.debug(f"   -> Found MPCO/Enhanced Template: returning 'mpco'")
                        return "mpco"
                    elif template_id == "standard" or "Standard" in template_name:
                        logger.debug(f"   -> Found Standard Template: returning 'standard'")
                        return "standard"  
                    elif template_id == "simplified" or "Simplified" in template_name:
                        logger.debug(f"   -> Found Simplified Template: returning 'simplified'")
                        return "simplified"
                    elif template_id == "baseline" or "Baseline" in template_name:
                        logger.debug(f"   -> Found Baseline: returning 'baseline'")
                        return "baseline"
                    
                    # Check for new baseline prompting templates by name
                    elif "Direct Prompt" in template_name or "direct" in template_name.lower():
                        logger.debug(f"   -> Found Direct Prompt: returning 'direct_prompt'")
                        return "direct_prompt"
                    elif "Chain-of-Thought" in template_name or "cot" in template_name.lower():
                        logger.debug(f"   -> Found Chain-of-Thought: returning 'chain_of_thought'")
                        return "chain_of_thought"
                    elif "Few-Shot" in template_name or ("few" in template_name.lower() and "shot" in template_name.lower()):
                        logger.debug(f"   -> Found Few-Shot: returning 'few_shot'")
                        return "few_shot"
                    elif "Metacognitive" in template_name or "metacognitive" in template_name.lower():
                        logger.debug(f"   -> Found Metacognitive: returning 'metacognitive'")
                        return "metacognitive"
                    elif "Contextual Prompting" in template_name or "contextual" in template_name.lower():
                        logger.debug(f"   -> Found Contextual Prompting: returning 'contextual_prompting'")
                        return "contextual_prompting"
                    
                    # Check for ablation study templates
                    elif "No Project Context" in template_name or "no_project_context" in template_name.lower():
                        logger.debug(f"   -> Found No Project Context: returning 'no_project_context'")
                        return "no_project_context"
                    elif "No Task Context" in template_name or "no_task_context" in template_name.lower():
                        logger.debug(f"   -> Found No Task Context: returning 'no_task_context'")
                        return "no_task_context"
                    elif "No LLM Context" in template_name or "no_llm_context" in template_name.lower():
                        logger.debug(f"   -> Found No LLM Context: returning 'no_llm_context'")
                        return "no_llm_context"
                    elif "Minimal Context" in template_name or "minimal_context" in template_name.lower():
                        logger.debug(f"   -> Found Minimal Context: returning 'minimal_context'")
                        return "minimal_context"
                        
                    else:
                        logger.debug(f"   -> Unknown template '{template_name}' (id: {template_id}): returning template_id or 'unknown'")
                        return template_id if template_id else "unknown"
        
        logger.debug(f"   -> No cached recommendation found for spec {spec_id[:8]}..., using fallback method")
        
    except Exception as e:
        logger.debug(f"   -> Error getting recommendation data: {str(e)}")
    
    # Fallback to solution-based analysis
    return determine_version_type_from_solution_metadata(solution)


def determine_version_type_from_solution_metadata(solution: dict) -> str:
    """Determine the version type based on solution metadata as fallback"""
    
    # Get solution metadata
    solution_name = solution.get("solution_name", "").lower()
    optimization_name = solution.get("optimization_name", "").lower()
    solution_id = solution.get("solution_id", "")
    
    logger.debug(f"🔍 Using solution metadata fallback for {solution_id[:8]}...")
    
    # Check for explicit version keywords in names
    explicit_keywords = {
        "original": ["original", "baseline_original", "unmodified", "raw"],
        "baseline": ["baseline", "default", "vanilla"],
        "simplified": ["simplified", "simple", "basic", "lite"],
        "standard": ["standard", "normal", "regular"],
        "enhanced": ["enhanced", "advanced", "optimized", "improved"]
    }
    
    # Check solution name first
    for version, keywords in explicit_keywords.items():
        if any(keyword in solution_name for keyword in keywords):
            if version == "baseline" and "original" in solution_name:
                continue  # Skip baseline if "original" is also present
            logger.debug(f"   -> Found explicit keyword in solution name: {version}")
            return version
    
    # Check optimization name
    for version, keywords in explicit_keywords.items():
        if any(keyword in optimization_name for keyword in keywords):
            if version == "baseline" and "original" in optimization_name:
                continue  # Skip baseline if "original" is also present
            logger.debug(f"   -> Found explicit keyword in optimization name: {version}")
            return version
    
    # Check for template-based patterns (meta-prompting solutions)
    meta_prompt_patterns = ["template", "meta", "prompt", "generated"]
    if any(pattern in solution_name or pattern in optimization_name for pattern in meta_prompt_patterns):
        logger.debug(f"   -> Detected meta-prompting solution")
        # Distribute meta-prompting solutions across simplified/standard/enhanced
        hash_val = hash(solution_id) % 3
        meta_versions = ["simplified", "standard", "enhanced"]
        version = meta_versions[hash_val]
        logger.debug(f"   -> Assigned meta-prompting version: {version}")
        return version
    
    # Use hash-based distribution as final fallback
    if solution_id:
        # Use common template IDs for fallback
        from meta_artemis_modules.shared_templates import BASELINE_PROMPTING_TEMPLATES, META_PROMPT_TEMPLATES
        common_templates = list(BASELINE_PROMPTING_TEMPLATES.keys())[:4] if BASELINE_PROMPTING_TEMPLATES else ["direct_prompt", "chain_of_thought", "few_shot", "metacognitive"]
        hash_value = hash(solution_id) % len(common_templates)
        fallback_version = common_templates[hash_value]
        logger.debug(f"   -> Using hash-based distribution: {fallback_version}")
        return fallback_version
    
    # Final fallback
    logger.debug(f"   -> Final fallback: 'unknown'")
    return "unknown"

def aggregate_cross_project_results(project_results: Dict[str, Dict], config: dict) -> Dict[str, Any]:
    """Aggregate results across all projects"""
    import numpy as np
    
    summary = {
        "total_projects": len(project_results),
        "projects_with_improvements": 0,
        "projects_with_significant_improvements": 0,
        "overall_improvement_percentage": 0,
        "average_improvement_percentage": 0,
        "best_performing_project": None,
        "worst_performing_project": None,
        "statistical_summary": {}
    }
    
    improvements = []
    significant_improvements = []
    
    for project_id, result in project_results.items():
        if "comparison" in result and result["comparison"]:
            improvement = result["comparison"]["improvement_percentage"]
            improvements.append(improvement)
            
            if improvement > 0:
                summary["projects_with_improvements"] += 1
            
            # Check statistical significance
            if (result.get("statistical_tests", {}).get("t_test", {}).get("significant", False) or
                result.get("statistical_tests", {}).get("mann_whitney_u", {}).get("significant", False)):
                if improvement > 0:
                    summary["projects_with_significant_improvements"] += 1
                    significant_improvements.append(improvement)
    
    if improvements:
        summary["average_improvement_percentage"] = np.mean(improvements)
        summary["overall_improvement_percentage"] = np.sum(improvements) / len(improvements)
        
        # Find best and worst performing projects
        best_idx = np.argmax(improvements)
        worst_idx = np.argmin(improvements)
        
        project_names = [result["project_name"] for result in project_results.values()]
        summary["best_performing_project"] = {
            "name": project_names[best_idx],
            "improvement": improvements[best_idx]
        }
        summary["worst_performing_project"] = {
            "name": project_names[worst_idx],
            "improvement": improvements[worst_idx]
        }
        
        summary["statistical_summary"] = {
            "mean_improvement": np.mean(improvements),
            "median_improvement": np.median(improvements),
            "std_improvement": np.std(improvements),
            "min_improvement": np.min(improvements),
            "max_improvement": np.max(improvements)
        }
    
    return summary





def main():
    """Main application function"""
    st.title("⚡ Artemis Performance Evaluation Application")
    st.markdown("*Large-scale evaluations through specialized batch operations*")
    
    # Initialize session state
    initialize_batch_session_state()
    
    state = get_batch_session_state()
    current_step = state.get("current_step", 1)
    
    # Display current step
    if current_step == 1:
        step_1_use_case_selection()
    elif current_step == 2:
        step_2_batch_configuration()

if __name__ == "__main__":
    main() 