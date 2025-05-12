import streamlit as st
import pandas as pd
import plotly.express as px
from meta_prompt_scoring_by_project import (
    MetaPromptEvaluator, default_scoring_llms, prompt_generation_llm, evaluation_metric,
    predefined_tasks, meta_prompt_template, LLMType, AVAILABLE_SCORING_LLMS, AVAILABLE_PROMPT_GENERATION_LLMS
)
import asyncio
import io
import logging
import time
import os

# Define the main function
def main():
    # Initialize session state if not exists
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    if 'has_evaluated' not in st.session_state:
        st.session_state.has_evaluated = False
    if 'project_info' not in st.session_state:
        st.session_state.project_info = None

    st.title("Meta Prompt Evaluation Tool")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Project ID input
    project_id = st.sidebar.text_input(
        "Project ID",
        value="c05998b8-d588-4c8d-a4bf-06d163c1c1d8",
        help="Enter the project ID to evaluate"
    )

    # Text area for meta-prompt
    meta_prompt = st.sidebar.text_area(
        "Meta Prompt",
        value="""You are a prompt generation expert. We are working together to improve a task-specific prompt for a particular LLM. 
Please follow the task instructions and contexts below:

## Instructions
1. Evaluate the contexts provided below. Identify areas for improvement, considering clarity, alignment with the task, 
and the target LLM's characteristics, capabilities or limitations.
2. Propose a refined version of the prompt that better aligns with the task requirements and the LLM's strengths.
3. Your answer should only contain the refined prompt.
4. The output format instructions and placeholder for the code snippets will be given later, so do not include them in the generated prompts.

## Evaluation Questions
- Is the current prompt clear when describing the task to the LLM?
- Does the prompt align with the LLM's capabilities and address its limitations?
- How can the prompt be improved to ensure better outputs for the task?
- Does the prompt reflect whether the code comes from real-world projects?

## Context for scoring LLMs:
- For cost-efficient LLMs (e.g., gpt-4-o-mini, gemini-v15-flash, llama-3-1-8b): these models have constrained context windows and limited internal chain-of-thought, so it is essential to prompt them with instructions that are clear and succinct to avoid over-generation.
- For larger LLMs (e.g., gpt-4-o, claude-v35-sonnet, claude-v37-sonnet): Allow for more complex and extensive internal reasoning. Encourage internal verification of any assumptions related to metrics based on the task description while maintaining clarity. 
""",
        height=400
    )

    # Dropdown for predefined tasks
    selected_task_name = st.sidebar.selectbox(
        "Select Task",
        options=list(predefined_tasks.keys())
    )

    # Display the baseline prompt with ability to edit
    custom_baseline_prompt = st.sidebar.text_area(
        "Baseline Prompt",
        value=predefined_tasks[selected_task_name]["default_prompt"],
        height=100,
    )

    # Allow editing the task description
    custom_task_description = st.sidebar.text_area(
        "Task Description (Editable)",
        value=predefined_tasks[selected_task_name]["description"],
        height=100,
        help="Edit this description to change the evaluation metric (e.g., code readability, maintainability, etc.)"
    )

    # Display the selected task instruction
    st.sidebar.text_area(
        "Task-Specific Instruction",
        value=predefined_tasks[selected_task_name]["instruction"],
        height=200,
        disabled=True
    )

    # Display the task data format
    st.sidebar.text_area(
        "Task Data Format",
        value=predefined_tasks[selected_task_name]["data_format"],
        height=200,
        disabled=True
    )

    # Multi-select for scoring LLMs
    selected_scoring_llms = st.sidebar.multiselect(
        "Select Scoring LLMs",
        options=[llm.value for llm in AVAILABLE_SCORING_LLMS],
        default=[llm.value for llm in default_scoring_llms]
    )

    # Convert selected scoring LLMs back to LLMType objects
    selected_scoring_llms = [LLMType(llm) for llm in selected_scoring_llms]

    # Radio buttons for prompt generation LLM
    selected_prompt_generation_llm = st.sidebar.radio(
        "Select Prompt Generation LLM",
        options=[llm.value for llm in AVAILABLE_PROMPT_GENERATION_LLMS],
        index=[llm.value for llm in AVAILABLE_PROMPT_GENERATION_LLMS].index(prompt_generation_llm.value)
    )

    # Convert selected prompt generation LLM back to LLMType object
    selected_prompt_generation_llm = LLMType(selected_prompt_generation_llm)

    # Dropdown for evaluation metric
    selected_evaluation_metric = st.sidebar.selectbox(
        "Select Evaluation Metric",
        options=["accuracy", "cohen_kappa"]
    )

    # Button to start evaluation
    if st.sidebar.button("Evaluation Results"):
        # Clear previous detailed analysis data
        for key in list(st.session_state.keys()):
            if key.startswith('detailed_analysis_df_') or key.startswith('code_pairs_'):
                del st.session_state[key]

        with st.spinner('Evaluating...'):
            # Capture log output
            log_capture_string = io.StringIO()
            ch = logging.StreamHandler(log_capture_string)
            ch.setLevel(logging.INFO)
            logger = logging.getLogger()
            logger.addHandler(ch)
            ch.flush()  # Ensure the stream handler is flushed

            # Create an evaluator instance
            evaluator = MetaPromptEvaluator(
                meta_prompt=meta_prompt,
                meta_prompt_template=meta_prompt_template,
                task_name=selected_task_name,
                current_prompt=custom_baseline_prompt,
                scoring_llms=selected_scoring_llms,
                prompt_generation_llm=selected_prompt_generation_llm,
                evaluation_metric=selected_evaluation_metric,
                project_id=project_id,
                custom_task_description=custom_task_description
            )

            # Call the evaluation function
            results, project_info = asyncio.run(evaluator.evaluate_meta_prompts())

            # Remove the stream handler after use
            logger.removeHandler(ch)

            # Store results in session state
            st.session_state.evaluation_results = results
            st.session_state.has_evaluated = True
            st.session_state.project_info = project_info

    # Display project information if available
    if st.session_state.project_info:
        st.header("Project Information")
        project_info = st.session_state.project_info
        
        # Create two columns for project info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Project Name:** {getattr(project_info, 'name', 'N/A')}")
            st.markdown(f"**Project ID:** {getattr(project_info, 'id', 'N/A')}")
            
        with col2:
            # Get languages from constructs
            constructs = evaluator.falcon_client.get_constructs_info(project_id)
            languages = set()
            for construct in constructs.values():
                if hasattr(construct, 'language'):
                    languages.add(construct.language.value)
            languages_str = ', '.join(sorted(languages)) if languages else 'N/A'
            st.markdown(f"**Languages:** {languages_str}")
            
            # Display creation date if available
            created_at = getattr(project_info, 'created_at', None)
            created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S") if created_at else 'N/A'
            st.markdown(f"**Created At:** {created_at_str}")
            
        # Display number of constructs and specs
        st.markdown(f"**Number of Constructs:** {len(constructs)}")
        total_specs = sum(len(construct.custom_specs) for construct in constructs.values() if hasattr(construct, 'custom_specs'))
        st.markdown(f"**Total Specs:** {total_specs}")
        
        # Display git information if available
        st.markdown("### Git Information")
        st.markdown(f"**Git URL:** {getattr(project_info, 'git_url', 'N/A')}")
        st.markdown(f"**Git Branch:** {getattr(project_info, 'git_branch', 'N/A')}")
        st.markdown(f"**Git Hash:** {getattr(project_info, 'git_hash', 'N/A')}")

    # Display results only if we have them in session state
    if st.session_state.has_evaluated and st.session_state.evaluation_results:
        results = st.session_state.evaluation_results
        results_df = pd.DataFrame(results)
        results_df['generated_prompt'] = results_df['generated_prompt'].apply(lambda x: x.replace('"', ''))

        # Create a container for the results
        results_container = st.container()

        with results_container:
            for scoring_llm in results_df['scoring_llm'].unique():
                st.subheader(f"Results for {scoring_llm}")
                llm_results = results_df[results_df['scoring_llm'] == scoring_llm]
                
                # Always create new detailed analysis for current evaluation
                actual_labels = llm_results[llm_results['prompt_type'] == 'Generated']['actual_label'].iloc[0]
                generated_predictions = llm_results[llm_results['prompt_type'] == 'Generated']['predicted_label'].iloc[0]
                baseline_predictions = llm_results[llm_results['prompt_type'] == 'Baseline']['predicted_label'].iloc[0]
                code_pairs = llm_results[llm_results['prompt_type'] == 'Generated']['code_pairs'].iloc[0]
                
                # Create detailed analysis DataFrame
                detailed_analysis_df = pd.DataFrame({
                    'index': range(len(actual_labels)),
                    'Actual': actual_labels,
                    'Generated': generated_predictions,
                    'Baseline': baseline_predictions
                })
                
                # Update session state with new data
                st.session_state[f'detailed_analysis_df_{scoring_llm}'] = detailed_analysis_df
                st.session_state[f'code_pairs_{scoring_llm}'] = code_pairs

                # Display results table and charts
                table_results = llm_results.astype(str)
                st.table(table_results[['prompt_type', 'generated_prompt', 'metric_value', 'time_spent']].T.rename(
                    index={
                        'prompt_type': 'Prompt Type',
                        'generated_prompt': 'Generated Prompt',
                        'metric_value': selected_evaluation_metric.capitalize(),
                        'time_spent': 'Time (s)'
                    }
                ))

                # Only show detailed analysis if we have code pairs
                if len(code_pairs) > 0:
                    # Create a bar chart for each prompt type
                    fig = px.bar(
                        llm_results,
                        x='prompt_type',
                        y='metric_value',
                        title=f"Evaluation Metric ({selected_evaluation_metric}) for {scoring_llm}",
                        labels={"prompt_type": "Prompt Type", "metric_value": selected_evaluation_metric.capitalize()},
                        width=600
                    )

                    # Update layout to make bars thinner
                    fig.update_traces(width=0.4)
                    fig.update_layout(bargap=0.6)

                    st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{scoring_llm}")

                    st.markdown("""
                    ### Interactive Analysis
                    **Below is the detailed comparison of predictions across all code pairs. Click any point on the chart to view the corresponding code pair.**
                    """)

                    # Use session state for the detailed analysis
                    detailed_analysis_df = st.session_state[f'detailed_analysis_df_{scoring_llm}']
                    code_pairs = st.session_state[f'code_pairs_{scoring_llm}']

                    # Initialize selected index in session state if not exists
                    if f'selected_index_{scoring_llm}' not in st.session_state:
                        st.session_state[f'selected_index_{scoring_llm}'] = 0

                    # Add selectbox for manual selection
                    selected_index = st.selectbox(
                        f"Select code pair to view (LLM: {scoring_llm})",
                        range(len(detailed_analysis_df)),
                        key=f'selected_index_{scoring_llm}'
                    )

                    # Create detailed analysis figure
                    fig_detailed = px.scatter(
                        detailed_analysis_df,
                        x='index',
                        y=['Actual', 'Generated', 'Baseline'],
                        title=f'Predictions Comparison for {scoring_llm}',
                        labels={'index': 'Code Pair Index', 'value': 'Label'},
                        height=400
                    )
                    
                    # Update layout for better visualization
                    fig_detailed.update_traces(mode='lines+markers')
                    fig_detailed.update_layout(
                        xaxis_title="Code Pair Index",
                        yaxis_title="Label",
                        yaxis=dict(
                            tickmode='array',
                            tickvals=[0, 1],
                            ticktext=['Not Improved', 'Improved']
                        )
                    )

                    # Display the chart
                    st.plotly_chart(
                        fig_detailed,
                        use_container_width=True,
                        key=f"detailed_analysis_{scoring_llm}"
                    )

                    # Get current selected index from session state
                    selected_index = st.session_state[f'selected_index_{scoring_llm}']

                    # Display code pairs using columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"### Code A (Pair {selected_index})")
                        st.code(code_pairs[selected_index]['code_a'], language="python")
                    with col2:
                        st.markdown(f"### Code B (Pair {selected_index})")
                        st.code(code_pairs[selected_index]['code_b'], language="python")

                    # Display predictions for the selected pair
                    st.markdown("### Predictions for this pair:")
                    st.markdown(f"""
                    - **Actual Label**: {detailed_analysis_df['Actual'][selected_index]}
                    - **Generated Prompt Prediction**: {detailed_analysis_df['Generated'][selected_index]}
                    - **Baseline Prompt Prediction**: {detailed_analysis_df['Baseline'][selected_index]}
                    """)
                else:
                    st.warning("No code pairs found to evaluate. This could mean either:\n" +
                             "1. The project has no specs with recommendations\n" +
                             "2. There was an error retrieving the specs\n" +
                             "Please check the logs for more details.")

if __name__ == "__main__":
    main()
