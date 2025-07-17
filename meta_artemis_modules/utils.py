"""
Utility functions for the Meta Artemis application.
"""

import streamlit as st
import colorsys
from loguru import logger
import sys
from meta_artemis_modules.shared_templates import OPTIMIZATION_TASKS
from typing import Dict, Any

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
            "project_id": "c05998b8-d588-4c8d-a4bf-06d163c1c1d8",
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


def get_session_state():
    """Get the current meta artemis session state"""
    return st.session_state.meta_artemis_state


def update_session_state(updates):
    """Update session state with given dictionary"""
    st.session_state.meta_artemis_state.update(updates)


def reset_session_state():
    """Reset session state to initial values"""
    if "meta_artemis_state" in st.session_state:
        del st.session_state.meta_artemis_state
    initialize_session_state() 


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
            "batch_evaluation": DEFAULT_BATCH_EVALUATION_CONFIG.copy()
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
    """Reset the batch session state to defaults"""
    if "batch_meta_artemis_state" in st.session_state:
        del st.session_state.batch_meta_artemis_state
    # Also clear cached recommendations
    if "cached_project_recommendations" in st.session_state:
        del st.session_state.cached_project_recommendations
    initialize_batch_session_state() 