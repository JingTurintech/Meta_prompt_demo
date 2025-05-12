import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from meta_prompt_optimization import (
    MetaPromptOptimizer, AVAILABLE_LLMS, OPTIMIZATION_TASKS, LLMType
)
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def main():
    st.title("Meta-Prompt Code Optimization Tool")
    
    # Initialize session state
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'has_optimized' not in st.session_state:
        st.session_state.has_optimized = False
    if 'project_info' not in st.session_state:
        st.session_state.project_info = None
        
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Project ID input
    project_id = st.sidebar.text_input(
        "Project ID",
        value="c05998b8-d588-4c8d-a4bf-06d163c1c1d8",
        help="Enter the project ID to optimize"
    )
    
    # Task selection
    selected_task = st.sidebar.selectbox(
        "Select Optimization Task",
        options=list(OPTIMIZATION_TASKS.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Display task description
    st.sidebar.markdown(f"**Task Description:**")
    st.sidebar.markdown(OPTIMIZATION_TASKS[selected_task]["description"])
    
    # Custom task description
    custom_task_description = st.sidebar.text_area(
        "Custom Task Description (Optional)",
        value="",
        help="Provide a custom description for the optimization task"
    )
    
    # LLM selection
    selected_llm = st.sidebar.selectbox(
        "Select LLM",
        options=[llm.value for llm in AVAILABLE_LLMS],
        index=[llm.value for llm in AVAILABLE_LLMS].index("gpt-4-o")
    )
    
    # Baseline prompt
    baseline_prompt = st.sidebar.text_area(
        "Baseline Prompt",
        value=OPTIMIZATION_TASKS[selected_task]["default_prompt"],
        height=150
    )
    
    # Start optimization button
    if st.sidebar.button("Start Optimization"):
        with st.spinner("Running optimization workflow..."):
            # Create optimizer instance
            optimizer = MetaPromptOptimizer(
                project_id=project_id,
                task_name=selected_task,
                llm_type=LLMType(selected_llm),
                current_prompt=baseline_prompt,
                custom_task_description=custom_task_description if custom_task_description else None
            )
            
            # Run optimization workflow
            results = asyncio.run(optimizer.run_optimization_workflow())
            
            # Store results in session state
            st.session_state.optimization_results = results
            st.session_state.has_optimized = True
            
    # Display results if available
    if st.session_state.has_optimized and st.session_state.optimization_results:
        results = st.session_state.optimization_results
        
        if 'error' in results:
            st.error(f"Error during optimization: {results['error']}")
        else:
            st.header("Optimization Results")
            
            # Display prompts
            st.subheader("Prompts")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Baseline Prompt:**")
                st.code(results['prompts']['baseline'])
            with col2:
                st.markdown("**Generated Prompt:**")
                st.code(results['prompts']['generated'])
            
            # Display ELO ratings
            st.subheader("ELO Ratings")
            
            # Create DataFrame for ratings
            ratings_data = []
            for result in results['results']:
                ratings_data.append({
                    'Spec ID': result['spec_id'],
                    'Original': result['original_rating'],
                    'Baseline': result['baseline_rating'],
                    'Generated': result['generated_rating']
                })
            
            ratings_df = pd.DataFrame(ratings_data)
            
            # Plot ELO ratings
            fig = go.Figure()
            
            # Add traces for each version
            for column in ['Original', 'Baseline', 'Generated']:
                fig.add_trace(go.Box(
                    y=ratings_df[column],
                    name=column,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ))
            
            fig.update_layout(
                title="ELO Rating Distribution",
                yaxis_title="ELO Rating",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display Bayesian ELO ratings
            st.subheader("Bayesian ELO Ratings")
            bayesian_df = pd.DataFrame({
                'Version': list(results['bayesian_ratings'].keys()),
                'Rating': list(results['bayesian_ratings'].values())
            })
            
            fig_bayesian = px.bar(
                bayesian_df,
                x='Version',
                y='Rating',
                title="Bayesian ELO Ratings"
            )
            
            st.plotly_chart(fig_bayesian, use_container_width=True)
            
            # Detailed results table
            st.subheader("Detailed Results")
            
            # Create a more detailed DataFrame
            detailed_data = []
            for result in results['results']:
                detailed_data.append({
                    'Spec ID': result['spec_id'],
                    'Original Rating': f"{result['original_rating']:.2f}",
                    'Baseline Rating': f"{result['baseline_rating']:.2f}",
                    'Generated Rating': f"{result['generated_rating']:.2f}",
                    'Baseline Status': 'Error' if 'error' in result['baseline_result'] else 'Success',
                    'Generated Status': 'Error' if 'error' in result['generated_result'] else 'Success'
                })
            
            detailed_df = pd.DataFrame(detailed_data)
            st.dataframe(detailed_df)
            
            # Download results as CSV
            csv = detailed_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name="optimization_results.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 