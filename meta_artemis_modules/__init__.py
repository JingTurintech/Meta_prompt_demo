"""
Meta Artemis Modules Package

This package contains modularized components of the Meta Artemis benchmark application.
"""

# Import all functions from each module for easy access
from .utils import (
    generate_colors,
    initialize_session_state,
    get_session_state,
    update_session_state,
    reset_session_state
)

from .project_manager import (
    get_project_info_async,
    get_existing_solutions_async,
    get_project_configurations,
    validate_project_id
)

from .recommendations import (
    generate_recommendations_async,
    generate_recommendations_step2,
    display_meta_prompts_progress,
    display_recommendations_progress,
    get_recommendation_summary
)

from .solutions import (
    create_solutions_from_recommendations,
    get_solution_status_summary,
    validate_solution_data,
    format_solution_display_name,
    get_solution_metrics_summary
)

from .execution import (
    wait_for_solution_completion,
    execute_solutions_async,
    execute_solutions_step3,
    get_execution_summary,
    format_execution_result,
    get_solution_details_from_artemis
)

from .visualization import (
    display_existing_solutions_analysis,
    display_single_solution_analysis,
    create_performance_comparison_chart,
    display_execution_results,
    create_summary_dashboard,
    _display_execution_results
)

# Define what gets exported when using "from meta_artemis_modules import *"
__all__ = [
    # Utils
    'generate_colors',
    'initialize_session_state',
    'get_session_state',
    'update_session_state',
    'reset_session_state',
    
    # Project Manager
    'get_project_info_async',
    'get_existing_solutions_async',
    'get_project_configurations',
    'validate_project_id',
    
    # Recommendations
    'generate_recommendations_async',
    'generate_recommendations_step2',
    'display_meta_prompts_progress',
    'display_recommendations_progress',
    'get_recommendation_summary',
    
    # Solutions
    'create_solutions_from_recommendations',
    'get_solution_status_summary',
    'validate_solution_data',
    'format_solution_display_name',
    'get_solution_metrics_summary',
    
    # Execution
    'wait_for_solution_completion',
    'execute_solutions_async',
    'execute_solutions_step3',
    'get_execution_summary',
    'format_execution_result',
    'get_solution_details_from_artemis',
    
    # Visualization
    'display_existing_solutions_analysis',
    'display_single_solution_analysis',
    'create_performance_comparison_chart',
    'display_execution_results',
    'create_summary_dashboard',
    '_display_execution_results'
] 