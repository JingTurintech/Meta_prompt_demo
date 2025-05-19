import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
import os
from benchmark_evaluator import (
    BenchmarkEvaluator, OPTIMIZATION_TASKS, LLMType,
    JUDGE_PROMPT_TEMPLATE, META_PROMPT_TEMPLATE,
    AVAILABLE_LLMS, save_evaluation_results, load_evaluation_results
)
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def initialize_session_state():
    """Initialize all session state variables"""
    if "app_state" not in st.session_state:
        st.session_state.app_state = {
            "has_evaluated": False,
            "optimization_results": None,
            "completed_snippets": [],
            "current_snippet_results": None,
            "has_shown_snippets_header": False,
            "baseline_prompt": None,
            "benchmark_info": None,
            "evaluation_count": 0,  # Track number of evaluations
            "snippet_containers": {},  # Store containers for each snippet
            "current_containers": None,  # Store current run's containers
            "saved_results": None  # Store loaded results
        }

def create_containers():
    """Create all containers once at app start"""
    eval_count = st.session_state.app_state["evaluation_count"]
    containers = {
        "progress": st.empty(),
        "meta_prompt": st.container(),
        "prompts": st.container(),
        "snippets": st.container(),  # Parent container for all snippets
        "final_results": st.container()
    }
    return containers

def handle_progress_update(update_data: dict, containers):
    """Handle progress updates from the evaluator"""
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
    elif status == "processing_snippet":
        progress = update_data.get("progress", 0)
        containers["progress"].progress(progress)
        containers["progress"].text(update_data["message"])
    elif status == "snippet_complete":
        result = update_data.get("result")
        if result:
            snippet_id = result["snippet_id"]
            st.session_state.app_state["completed_snippets"].append(result)
            
            # Create a new container for this snippet if it doesn't exist
            if snippet_id not in st.session_state.app_state["snippet_containers"]:
                with containers["snippets"]:
                    if not st.session_state.app_state["has_shown_snippets_header"]:
                        st.markdown("### Individual Snippet Results")
                        st.session_state.app_state["has_shown_snippets_header"] = True
                    
                    # Create a new container for this snippet
                    snippet_container = st.container()
                    st.session_state.app_state["snippet_containers"][snippet_id] = snippet_container
            
            # Display the result in the snippet's container
            with st.session_state.app_state["snippet_containers"][snippet_id]:
                display_snippet_result(
                    result,
                    len(st.session_state.app_state["completed_snippets"]),
                    f"snippet_{snippet_id}_{eval_count}"
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

def display_snippet_result(result, snippet_number, key_suffix):
    """Display results for a single snippet"""
    with st.expander(f"Snippet {snippet_number} (ID: {result.get('snippet_id', 'Unknown')})", expanded=True):
        # Display ratings
        st.markdown(f"""
        **ELO Ratings:**
        - Original: {result['original_rating']:.1f}
        - Baseline: {result['baseline_rating']:.1f}
        - Generated: {result['generated_rating']:.1f}
        """)
        
        # Display comparison results
        st.markdown("### Comparison Results")
        
        # Calculate summary statistics for this snippet's comparisons
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
                st.plotly_chart(fig, use_container_width=True, key=f"{key_suffix}_summary_chart")

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
        title="Average ELO Ratings Across All Snippets",
        yaxis_title="ELO Rating",
        showlegend=False
    )
    
    st.plotly_chart(fig_avg, use_container_width=True, key=f"avg_ratings_{key_suffix}")
    
    if 'results' in final_results:
        detailed_df = pd.DataFrame([
            {
                'Snippet': i + 1,
                'Original': r['original_rating'],
                'Baseline': r['baseline_rating'],
                'Generated': r['generated_rating']
            }
            for i, r in enumerate(final_results['results'])
        ])
        
        fig_progression = go.Figure()
        
        for col in ['Original', 'Baseline', 'Generated']:
            fig_progression.add_trace(go.Scatter(
                x=detailed_df['Snippet'],
                y=detailed_df[col],
                name=col,
                mode='lines+markers'
            ))
        
        fig_progression.update_layout(
            title="ELO Rating Progression Across Snippets",
            xaxis_title="Snippet Number",
            yaxis_title="ELO Rating",
            showlegend=True
        )
        
        st.plotly_chart(fig_progression, use_container_width=True, key=f"progression_{key_suffix}")

def load_saved_results():
    """Load and display saved evaluation results"""
    st.sidebar.markdown("### Load Saved Results")
    
    # Get list of saved results
    if not os.path.exists("results"):
        st.sidebar.warning("No saved results found")
        return
        
    saved_files = [f for f in os.listdir("results") if f.endswith('_evaluation.json')]
    if not saved_files:
        st.sidebar.warning("No saved results found")
        return
    
    # Sort files by timestamp (newest first)
    saved_files.sort(reverse=True)
    
    selected_file = st.sidebar.selectbox(
        "Select Saved Results",
        options=saved_files,
        format_func=lambda x: x.replace('_evaluation.json', '').replace('_', ' ')
    )
    
    if st.sidebar.button("Load Selected Results"):
        filepath = os.path.join("results", selected_file)
        results = load_evaluation_results(filepath)
        
        if results:
            st.session_state.app_state["saved_results"] = results
            st.session_state.app_state["has_evaluated"] = True
            
            # Create new containers for displaying results
            containers = create_containers()
            st.session_state.app_state["current_containers"] = containers
            
            # Get configuration data - handle both old and new format
            config = results.get("configuration", {})
            meta_prompt = config.get("meta_prompt_used") or results.get("meta_prompt_used", "Meta-prompt not available")
            baseline_prompt = config.get("baseline_prompt") or results.get("prompts", {}).get("baseline", "Baseline prompt not available")
            generated_prompt = config.get("generated_prompt") or results.get("prompts", {}).get("generated", "Generated prompt not available")
            
            # Display meta-prompt
            with containers["meta_prompt"]:
                st.markdown("### Meta-Prompt Used")
                st.markdown("This is the actual meta-prompt used with project context:")
                st.text_area(
                    "Filled Template",
                    value=meta_prompt,
                    height=400,
                    disabled=True
                )
            
            # Display prompts
            with containers["prompts"]:
                st.markdown("### Generated Prompts")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Baseline Prompt**")
                    st.text_area(
                        "",
                        value=baseline_prompt,
                        height=200,
                        disabled=True
                    )
                with col2:
                    st.markdown("**Generated Prompt**")
                    st.text_area(
                        "",
                        value=generated_prompt,
                        height=200,
                        disabled=True
                    )
            
            # Display individual snippet results
            with containers["snippets"]:
                st.markdown("### Individual Snippet Results")
                for idx, result in enumerate(results.get("results", []), 1):
                    display_snippet_result(result, idx, f"snippet_{result.get('snippet_id', str(idx))}_loaded")
            
            # Display final results
            with containers["final_results"]:
                display_final_results(results, "loaded")
            
            st.success("Results loaded successfully!")
        else:
            st.error("Failed to load results")

def main():
    st.title("Code Optimization Meta-Prompt Benchmark Evaluation")
    
    # Initialize session state
    initialize_session_state()
    
    # Create containers once
    containers = create_containers()

    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Add option to load saved results
    load_saved_results()
    
    # Add separator between load and new evaluation options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### New Evaluation")

    # Benchmark file selection
    benchmark_files = []
    if os.path.exists("benchmarks"):
        benchmark_files = [f for f in os.listdir("benchmarks") if f.endswith('.json')]
    
    if not benchmark_files:
        st.error("No benchmark files found in the 'benchmarks' directory")
        return
    
    # Set default selection to QuantLib if available
    default_index = 0
    for i, file in enumerate(benchmark_files):
        if "QuantLib" in file:
            default_index = i
            break
    
    selected_benchmarks = st.sidebar.multiselect(
        "Select Benchmark Files",
        options=benchmark_files,
        default=[benchmark_files[default_index]] if benchmark_files else [],
        format_func=lambda x: x.replace('.json', '').replace('_', ' ')
    )
    
    if not selected_benchmarks:
        st.warning("Please select at least one benchmark file to evaluate")
        return
    
    # Load and display benchmark info for selected files
    st.sidebar.markdown("### Selected Benchmarks Information")
    for benchmark_file in selected_benchmarks:
        try:
            with open(os.path.join("benchmarks", benchmark_file), 'r') as f:
                benchmark_data = json.load(f)
                
            project_info = benchmark_data["metadata"]["project_info"]
            project_id = project_info.get('project_id', 'Unknown')
            artemis_url = f"https://artemis.turintech.ai/projects/{project_id}"
            
            st.sidebar.markdown(f"""
            **{project_info.get('name', 'Unknown')}**
            - Description: {project_info.get('description', 'No description available')}
            - Language: {project_info.get('language', 'unknown')}
            - Code Snippets: {len(benchmark_data.get('code_snippets', []))}
            - Artemis URL: [@{project_id}]({artemis_url})
            ---
            """)
            
        except Exception as e:
            st.sidebar.error(f"Error loading {benchmark_file}: {str(e)}")

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
        st.session_state.app_state["completed_snippets"] = []
        st.session_state.app_state["current_snippet_results"] = None
        st.session_state.app_state["has_shown_snippets_header"] = False
        st.session_state.app_state["baseline_prompt"] = custom_baseline_prompt
        st.session_state.app_state["snippet_containers"] = {}  # Reset snippet containers
        
        with st.spinner("Running optimization evaluation..."):
            # Process each selected benchmark
            all_results = []
            for benchmark_file in selected_benchmarks:
                st.markdown(f"### Processing {benchmark_file}")
                
                evaluator = BenchmarkEvaluator(
                    task_name=selected_task,
                    llm_type=LLMType(selected_llm),
                    judge_llm_type=LLMType(selected_judge_llm),
                    current_prompt=custom_baseline_prompt,
                    custom_task_description=custom_task_description,
                    custom_meta_prompt=custom_meta_prompt,
                    progress_callback=lambda x: handle_progress_update(x, containers)
                )
                
                results = asyncio.run(evaluator.evaluate_benchmark(os.path.join("benchmarks", benchmark_file)))
                
                if results:
                    all_results.append(results)
                    
                    # Save individual results
                    saved_file = save_evaluation_results(results)
                    st.success(f"Results for {benchmark_file} saved to: {saved_file}")
                    
                    # Create download button for this benchmark's results
                    with open(saved_file, "r") as f:
                        results_json = f.read()
                    
                    st.download_button(
                        label=f"Download Results for {benchmark_file}",
                        data=results_json,
                        file_name=os.path.basename(saved_file),
                        mime="application/json",
                        help=f"Download results for {benchmark_file}"
                    )
            
            # Only save combined results if there are multiple benchmarks
            if len(all_results) > 1:
                # Combine results from all benchmarks
                combined_results = {
                    "benchmarks": all_results,
                    "overall_statistics": {
                        "total_benchmarks": len(all_results),
                        "total_snippets": sum(r["statistics"]["total_snippets"] for r in all_results),
                        "successful_snippets": sum(r["statistics"]["successful_snippets"] for r in all_results),
                        "failed_snippets": sum(r["statistics"]["failed_snippets"] for r in all_results)
                    }
                }
                
                # Save combined results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                combined_filename = f"combined_results_{timestamp}.json"
                combined_filepath = os.path.join("results", combined_filename)
                
                with open(combined_filepath, 'w') as f:
                    json.dump(combined_results, f, indent=2)
                
                st.success(f"Combined results saved to: {combined_filepath}")
                
                # Create download button for combined results
                with open(combined_filepath, "r") as f:
                    combined_json = f.read()
                
                st.download_button(
                    label="Download Combined Results",
                    data=combined_json,
                    file_name=combined_filename,
                    mime="application/json",
                    help="Download combined results from all benchmarks"
                )
                
            st.session_state.app_state["optimization_results"] = all_results[0] if len(all_results) == 1 else combined_results
            st.session_state.app_state["has_evaluated"] = True

if __name__ == "__main__":
    main() 