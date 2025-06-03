import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import os
from meta_prompt_optimization_enhanced_by_project import (
    MetaPromptOptimizer, OPTIMIZATION_TASKS, LLMType,
    JUDGE_PROMPT_TEMPLATE, META_PROMPT_TEMPLATE
)
from loguru import logger
import sys
from artemis_client.falcon.client import FalconClient, FalconSettings
from evoml_services.clients.thanos import ThanosSettings
from typing import Dict, Any

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Define available LLMs
AVAILABLE_LLMS = [
    LLMType("gpt-4-o-mini"),
    LLMType("gemini-v15-flash"),
    LLMType("llama-3-1-8b"),
    LLMType("gpt-4-o"),
    LLMType("claude-v35-sonnet"),
    LLMType("claude-v37-sonnet")
]

def initialize_session_state():
    """Initialize all session state variables"""
    if "app_state" not in st.session_state:
        st.session_state.app_state = {
            "has_evaluated": False,
            "optimization_results": None,
            "completed_specs": [],
            "current_spec_results": None,
            "has_shown_specs_header": False,
            "baseline_prompt": None,
            "project_info": None,
            "detected_language": "unknown",
            "project_files": [],
            "evaluation_count": 0,  # Track number of evaluations
            "spec_containers": {},  # Store containers for each spec
            "current_containers": None  # Store current run's containers
        }

def create_containers():
    """Create all containers once at app start"""
    eval_count = st.session_state.app_state["evaluation_count"]
    containers = {
        "progress": st.empty(),
        "meta_prompt": st.container(),
        "prompts": st.container(),
        "specs": st.container(),  # Parent container for all specs
        "final_results": st.container()
    }
    return containers

def handle_progress_update(update_data: Dict[str, Any], containers):
    """Handle progress updates from the optimizer"""
    status = update_data.get("status")
    eval_count = st.session_state.app_state["evaluation_count"]
    
    if status == "setup":
        containers["progress"].info(update_data["message"])
    elif status == "setup_complete":
        containers["progress"].success("Setup complete!")
    elif status == "generating_meta_prompt":
        containers["progress"].info(update_data["message"])
    elif status == "meta_prompt_ready":
        with containers["meta_prompt"]:
            st.markdown("### Meta-Prompt Used")
            st.markdown("This is the actual meta-prompt used with project context:")
            st.text_area(
                "Filled Template",
                value=update_data["filled_meta_prompt"],
                height=400,
                disabled=True,
                key=f"meta_prompt_text_{eval_count}"
            )
    elif status == "generating_prompt":
        containers["progress"].info(update_data["message"])
    elif status == "prompt_ready":
        with containers["prompts"]:
            st.markdown("### Generated Prompts")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Baseline Prompt**")
                st.text_area(
                    "",
                    value=st.session_state.app_state["baseline_prompt"],
                    height=200,
                    disabled=True,
                    key=f"baseline_prompt_text_{eval_count}"
                )
            with col2:
                st.markdown("**Generated Prompt**")
                st.text_area(
                    "",
                    value=update_data["generated_prompt"],
                    height=200,
                    disabled=True,
                    key=f"generated_prompt_text_{eval_count}"
                )
        containers["progress"].success("Generated optimization prompt!")
    elif status == "processing_spec":
        progress = update_data.get("progress", 0)
        containers["progress"].progress(progress)
        containers["progress"].text(update_data["message"])
    elif status == "spec_complete":
        result = update_data.get("result")
        if result:
            spec_id = result["spec_id"]
            st.session_state.app_state["completed_specs"].append(result)
            
            # Create a new container for this spec if it doesn't exist
            if spec_id not in st.session_state.app_state["spec_containers"]:
                with containers["specs"]:
                    if not st.session_state.app_state["has_shown_specs_header"]:
                        st.markdown("### Individual Spec Results")
                        st.session_state.app_state["has_shown_specs_header"] = True
                    
                    # Create a new container for this spec
                    spec_container = st.container()
                    st.session_state.app_state["spec_containers"][spec_id] = spec_container
            
            # Display the result in the spec's container
            with st.session_state.app_state["spec_containers"][spec_id]:
                display_spec_result(
                    result,
                    len(st.session_state.app_state["completed_specs"]),
                    f"spec_{spec_id}_{eval_count}"
                )
    elif status == "complete":
        final_results = update_data.get("final_results")
        if final_results:
            with containers["final_results"]:
                display_final_results(
                    final_results,
                    f"final_{eval_count}"
                )
        containers["progress"].success("Evaluation complete!")
    elif status == "error":
        containers["progress"].error(update_data["message"])

def display_spec_result(result, spec_number, key_suffix):
    """Display results for a single spec"""
    with st.expander(f"Spec {spec_number} (ID: {result.get('spec_id', 'Unknown')})", expanded=True):
        # Display ratings
        st.markdown(f"""
        **ELO Ratings:**
        - Original: {result['original_rating']:.1f}
        - Baseline: {result['baseline_rating']:.1f}
        - Generated: {result['generated_rating']:.1f}
        """)
        
        # Display comparison results
        st.markdown("### Comparison Results")
        
        # Calculate summary statistics for this spec's comparisons
        summary = {
            'original_vs_baseline': {'wins': [0, 0], 'ties': 0},
            'original_vs_generated': {'wins': [0, 0], 'ties': 0},
            'baseline_vs_generated': {'wins': [0, 0], 'ties': 0}
        }
        
        # Process comparisons and group by pair regardless of order
        comparison_pairs = {}
        
        for comp in result.get('comparisons', []):
            if not isinstance(comp, dict) or 'comparison' not in comp or 'score' not in comp:
                continue
                
            names = comp['comparison'].lower().split(' vs ')
            order = comp.get('order', 1)
            score = comp['score']
            
            if 'original' in names:
                if names[0] == 'original':
                    normalized_names = names
                    normalized_score = score
                else:
                    normalized_names = [names[1], names[0]]
                    normalized_score = 1.0 - score if score != 0.5 else score
            else:
                if 'baseline' in names[0]:
                    normalized_names = names
                    normalized_score = score
                else:
                    normalized_names = [names[1], names[0]]
                    normalized_score = 1.0 - score if score != 0.5 else score
            
            pair_key = '_vs_'.join(normalized_names)
            if pair_key not in summary:
                summary[pair_key] = {'wins': [0, 0], 'ties': 0}
            
            if normalized_score == 0.5:
                summary[pair_key]['ties'] += 1
            elif normalized_score == 1.0:
                summary[pair_key]['wins'][0] += 1
            else:
                summary[pair_key]['wins'][1] += 1

        # Create two columns for the summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Summary")
            
            comparison_data = []
            for comp_type, stats in summary.items():
                names = comp_type.split('_vs_')
                if stats['wins'][0] > 0:
                    comparison_data.append({
                        'Comparison': f"{names[0].title()} vs {names[1].title()}", 
                        'Winner': names[0].title(),
                        'Count': stats['wins'][0]
                    })
                if stats['wins'][1] > 0:
                    comparison_data.append({
                        'Comparison': f"{names[0].title()} vs {names[1].title()}", 
                        'Winner': names[1].title(),
                        'Count': stats['wins'][1]
                    })
                if stats['ties'] > 0:
                    comparison_data.append({
                        'Comparison': f"{names[0].title()} vs {names[1].title()}", 
                        'Winner': 'Tie',
                        'Count': stats['ties']
                    })
            
            df = pd.DataFrame(comparison_data)
            if not df.empty:
                fig = px.bar(df, 
                           x='Comparison', 
                           y='Count',
                           color='Winner',
                           barmode='group',
                           color_discrete_sequence=['#e74c3c', '#2ecc71', '#3498db'])
                
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Number of Comparisons",
                    legend_title="Winner",
                    margin=dict(t=20),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                fig.update_yaxes(rangemode="tozero", dtick=1)
                fig.update_traces(width=0.2)
                # Add unique key using spec ID and a prefix
                st.plotly_chart(fig, use_container_width=True, key=f"{key_suffix}_summary_chart_{result.get('spec_id', 'unknown')}")

        with col2:
            st.markdown("#### Detailed Results")
            detailed_data = []
            for comparison in result.get('comparisons', []):
                names = comparison['comparison'].split(' vs ')
                order = comparison.get('order', 1)
                
                if comparison['score'] == 1.0:
                    winner = 'A' if order == 1 else 'B'
                elif comparison['score'] == 0.0:
                    winner = 'B' if order == 1 else 'A'
                else:
                    winner = 'TIE'
                    
                if order == 2:
                    names = names[::-1]
                    
                if winner == 'TIE':
                    result_text = "Tie"
                elif winner == 'A':
                    result_text = f"{names[0].title()} wins"
                else:
                    result_text = f"{names[1].title()} wins"
                    
                detailed_data.append({
                    "Comparison": f"{names[0].title()} vs {names[1].title()}",
                    "Result": result_text
                })
            
            st.table(pd.DataFrame(detailed_data))
        
        # Display code
        st.markdown("#### Code Comparison")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original Code**")
            st.code(result['original_code'])
        with col2:
            st.markdown("**Baseline Optimization**")
            st.code(result['baseline_code'])
        with col3:
            st.markdown("**Generated Optimization**")
            st.code(result['generated_code'])

def display_final_results(final_results, key_suffix):
    """Display final average results"""
    st.markdown("### Average ELO Ratings")
    avg_ratings = final_results['average_ratings']
    
    # Create bar chart for average ratings
    fig_avg = go.Figure(data=[
        go.Bar(
            x=['Original Code', 'Baseline Optimization', 'Generated Optimization'],
            y=[avg_ratings['original'], avg_ratings['baseline'], avg_ratings['generated']],
            text=[f"{rating:.1f}" for rating in [avg_ratings['original'], avg_ratings['baseline'], avg_ratings['generated']]],
            textposition='auto',
        )
    ])
    
    fig_avg.update_layout(
        title="Average ELO Ratings Across All Specs",
        yaxis_title="ELO Rating",
        showlegend=False
    )
    
    st.plotly_chart(fig_avg, use_container_width=True, key=f"avg_ratings_{key_suffix}")
    
    if 'results' in final_results:
        detailed_df = pd.DataFrame(final_results['results'])
        fig_progression = go.Figure()
        
        for rating_type in ['original_rating', 'baseline_rating', 'generated_rating']:
            fig_progression.add_trace(go.Scatter(
                x=list(range(len(detailed_df))),
                y=detailed_df[rating_type],
                name=rating_type.replace('_rating', '').title(),
                mode='lines+markers'
            ))
        
        fig_progression.update_layout(
            title="ELO Rating Progression Across Specs",
            xaxis_title="Spec Number",
            yaxis_title="ELO Rating",
            showlegend=True
        )
        
        st.plotly_chart(fig_progression, use_container_width=True, key=f"progression_{key_suffix}")

def save_evaluation_results(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Save evaluation results and configuration to a JSON file.
    Returns the path to the saved file.
    """
    # Get current date and time
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    
    # Create base results directory
    base_dir = "results"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create filename with datetime and project info
    project_name = config["project_info"]["name"].replace(" ", "_")
    task_name = config["task"]["name"]
    filename = f"{datetime_str}_{project_name}_{task_name}_evaluation.json"
    filepath = os.path.join(base_dir, filename)
    
    # Combine results and configuration
    full_results = {
        "configuration": config,
        "evaluation_results": results,
        "metadata": {
            "timestamp": now.isoformat(),
            "datetime": datetime_str
        }
    }
    
    # Save to file
    with open(filepath, "w") as f:
        json.dump(full_results, f, indent=2)
    
    logger.info(f"Results saved to: {filepath}")
    return filepath

def main():
    st.title("Code Optimization Meta-Prompt Evaluation")
    
    # Initialize session state
    initialize_session_state()
    
    # Create containers once
    containers = create_containers()

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Project ID input
    project_id = st.sidebar.text_input(
        "Project ID",
        value="c05998b8-d588-4c8d-a4bf-06d163c1c1d8",
        help="Enter the project ID to evaluate"
    )

    # Initialize Falcon client and get project info when project ID changes
    if project_id:
        try:
            logger.info(f"Attempting to fetch project information for ID: {project_id}")
            falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
            thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
            falcon_client = FalconClient(falcon_settings, thanos_settings)
            falcon_client.authenticate()
            
            logger.info("Successfully authenticated with Falcon API")
            project_info = falcon_client.get_project(project_id)
            logger.info(f"Raw project info response: {project_info}")
            
            # Get construct details for language detection
            construct_details = falcon_client.get_constructs_info(project_id)
            
            # Initialize language info
            detected_language = "unknown"
            project_files = []
            
            if construct_details:
                # Get the first construct's details to determine language
                first_construct = next(iter(construct_details.values()))
                detected_language = first_construct.language if hasattr(first_construct, 'language') else "unknown"
                
                # Collect files from constructs
                for construct in construct_details.values():
                    if hasattr(construct, 'file') and construct.file:
                        project_files.append(construct.file)
                
                logger.info(f"Detected language: {detected_language}")
                logger.info(f"Project files: {project_files}")
            else:
                logger.warning("No construct details available for language detection")
            
            # Store the detected info in session state separately
            st.session_state.app_state["detected_language"] = detected_language
            st.session_state.app_state["project_files"] = project_files
            st.session_state.app_state["project_info"] = project_info
            
            # Log project info
            logger.info(f"Project name: {getattr(project_info, 'name', 'Unknown')}")
            logger.info(f"Project description: {getattr(project_info, 'description', 'No description available')}")
            logger.info("Successfully stored project info in session state")
            
        except Exception as e:
            logger.error(f"Error fetching project information: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            st.error(f"Error fetching project information: {str(e)}")
            st.session_state.app_state["project_info"] = None
            st.session_state.app_state["detected_language"] = "unknown"
            st.session_state.app_state["project_files"] = []

    # Display project information if available
    if st.session_state.app_state["project_info"]:
        st.sidebar.markdown("### Project Information")
        project_info = st.session_state.app_state["project_info"]
        
        # Get the basic info
        name = getattr(project_info, 'name', 'Unknown')
        description = getattr(project_info, 'description', 'No description available')
        
        # Display all information including detected language
        st.sidebar.markdown(f"""
        **Name:** {name}
        
        **Description:** {description}
        
        **Detected Language:** {st.session_state.app_state["detected_language"]}
        
        **Files:** {len(st.session_state.app_state["project_files"])} file(s)
        """)
        
        # Optionally show file list in an expander
        if st.session_state.app_state["project_files"]:
            with st.sidebar.expander("View Project Files"):
                for file in st.session_state.app_state["project_files"]:
                    st.write(f"- {file}")

    # Task selection
    selected_task = st.sidebar.selectbox(
        "Select Optimization Task",
        options=list(OPTIMIZATION_TASKS.keys()),
        format_func=lambda x: OPTIMIZATION_TASKS[x]["description"]
    )
    
    # LLM selection
    selected_llm = st.sidebar.selectbox(
        "Select Optimization LLM",
        options=[llm.value for llm in AVAILABLE_LLMS],
        index=0  # Default to gpt-4-o-mini
    )
    
    # Judge LLM selection
    selected_judge_llm = st.sidebar.selectbox(
        "Select Judge LLM",
        options=[llm.value for llm in AVAILABLE_LLMS],
        index=0,  # Default to gpt-4-o-mini
        help="LLM that will judge code performance"
    )
    
    # Custom prompt input
    custom_baseline_prompt = st.sidebar.text_area(
        "Baseline Optimization Prompt",
        value=OPTIMIZATION_TASKS[selected_task]["default_prompt"],
        height=200
    )
    
    # Custom task description
    custom_task_description = st.sidebar.text_area(
        "Custom Task Description",
        value=OPTIMIZATION_TASKS[selected_task]["description"],
        height=100
    )
    
    # Judge prompt template
    st.sidebar.markdown("### Judge Prompt Template")
    st.sidebar.markdown("This prompt is used to compare and evaluate code optimizations.")
    
    default_judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        objective=OPTIMIZATION_TASKS[selected_task]["objective"],
        task_description=OPTIMIZATION_TASKS[selected_task]["description"],
        task_considerations=OPTIMIZATION_TASKS[selected_task]["considerations"],
        code_a="[Code A will be inserted here]",
        code_b="[Code B will be inserted here]"
    )
    
    custom_judge_prompt = st.sidebar.text_area(
        "Judge Prompt Template",
        value=default_judge_prompt,
        height=400,
        help="Customize how the judge evaluates code optimizations. Use {objective}, {task_description}, {task_considerations}, {code_a}, and {code_b} as placeholders."
    )

    # Meta-prompt template configuration
    st.sidebar.markdown("### Meta-Prompt Template")
    st.sidebar.markdown("This template is used to generate optimization prompts based on project context.")
    
    custom_meta_prompt = st.sidebar.text_area(
        "Meta-Prompt Template",
        value=META_PROMPT_TEMPLATE,
        height=400,
        help="Customize how the meta-prompt is generated. Available placeholders: {objective}, {project_name}, {project_description}, {project_languages}, {task_description}, {current_prompt}, {target_llm}"
    )
    
    # Evaluation button
    if st.sidebar.button("Run Evaluation"):
        # Increment evaluation count
        st.session_state.app_state["evaluation_count"] += 1
        
        # Create new containers for this run
        containers = create_containers()
        st.session_state.app_state["current_containers"] = containers
        
        # Reset state for new evaluation
        st.session_state.app_state["completed_specs"] = []
        st.session_state.app_state["current_spec_results"] = None
        st.session_state.app_state["has_shown_specs_header"] = False
        st.session_state.app_state["baseline_prompt"] = custom_baseline_prompt
        st.session_state.app_state["spec_containers"] = {}  # Reset spec containers
        
        with st.spinner("Running optimization evaluation..."):
            optimizer = MetaPromptOptimizer(
                project_id=project_id,
                task_name=selected_task,
                llm_type=LLMType(selected_llm),
                judge_llm_type=LLMType(selected_judge_llm),
                current_prompt=custom_baseline_prompt,
                custom_task_description=custom_task_description,
                custom_meta_prompt=custom_meta_prompt,
                progress_callback=lambda x: handle_progress_update(x, containers)
            )

            if custom_judge_prompt != default_judge_prompt:
                optimizer.custom_judge_prompt = custom_judge_prompt
            
            results = asyncio.run(optimizer.run_optimization_workflow())
            
            # Save configuration and results
            config = {
                "project_id": project_id,
                "project_info": {
                    "name": getattr(st.session_state.app_state["project_info"], 'name', 'Unknown'),
                    "description": getattr(st.session_state.app_state["project_info"], 'description', 'No description available'),
                    "detected_language": st.session_state.app_state["detected_language"],
                    "file_count": len(st.session_state.app_state["project_files"]),
                    "files": st.session_state.app_state["project_files"]
                },
                "task": {
                    "name": selected_task,
                    "description": custom_task_description
                },
                "llms": {
                    "optimization_llm": selected_llm,
                    "judge_llm": selected_judge_llm
                },
                "prompts": {
                    "baseline_prompt": custom_baseline_prompt,
                    "meta_prompt_template": custom_meta_prompt,
                    "judge_prompt_template": custom_judge_prompt
                }
            }
            
            saved_file = save_evaluation_results(results, config)
            st.success(f"Results saved to: {saved_file}")
            
            # Create download button for convenience
            with open(saved_file, "r") as f:
                results_json = f.read()
            
            st.download_button(
                label="Download Results File",
                data=results_json,
                file_name=os.path.basename(saved_file),
                mime="application/json",
                help="Download a copy of the automatically saved results file"
            )
            
            st.session_state.app_state["optimization_results"] = results
            st.session_state.app_state["has_evaluated"] = True

if __name__ == "__main__":
    main()