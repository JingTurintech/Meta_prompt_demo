import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from meta_prompt_optimization_by_project import (
    MetaPromptOptimizer, OPTIMIZATION_TASKS, LLMType,
    JUDGE_PROMPT_TEMPLATE
)
from loguru import logger
import sys

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

def main():
    st.title("Code Optimization Meta-Prompt Evaluation")
    
    # Initialize session state
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'has_evaluated' not in st.session_state:
        st.session_state.has_evaluated = False

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Project ID input
    project_id = st.sidebar.text_input(
        "Project ID",
        value="c05998b8-d588-4c8d-a4bf-06d163c1c1d8",
        help="Enter the project ID to evaluate"
    )

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
    
    # Evaluation button
    if st.sidebar.button("Run Evaluation"):
        with st.spinner("Running optimization evaluation..."):
            optimizer = MetaPromptOptimizer(
                project_id=project_id,
                task_name=selected_task,
                llm_type=LLMType(selected_llm),
                judge_llm_type=LLMType(selected_judge_llm),
                current_prompt=custom_baseline_prompt,
                custom_task_description=custom_task_description
            )

            # Store the custom judge prompt in the optimizer for use when creating the LLMJudge
            if custom_judge_prompt != default_judge_prompt:
                optimizer.custom_judge_prompt = custom_judge_prompt
            
            results = asyncio.run(optimizer.run_optimization_workflow())
            st.session_state.optimization_results = results
            st.session_state.has_evaluated = True
    
    # Display results
    if st.session_state.has_evaluated and st.session_state.optimization_results:
        results = st.session_state.optimization_results
        
        # Display prompts
        st.header("Prompts")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Baseline Prompt")
            st.text_area("", value=results['prompts']['baseline'], height=200, disabled=True)
        with col2:
            st.subheader("Generated Prompt")
            st.text_area("", value=results['prompts']['generated'], height=200, disabled=True)
        
        # Display average ELO ratings
        st.header("Average ELO Ratings")
        avg_ratings = results['average_ratings']
        
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
        
        st.plotly_chart(fig_avg, use_container_width=True)
        
        # Display detailed results
        st.header("Detailed Results")
        
        # Create DataFrame for all specs
        detailed_df = pd.DataFrame(results['results'])
        
        # Create line chart showing rating progression
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
        
        st.plotly_chart(fig_progression, use_container_width=True)
        
        # Display individual spec results
        st.header("Individual Spec Results")
        
        for idx, result in enumerate(results.get('results', [])):
            if not isinstance(result, dict):
                st.error(f"Invalid result format at index {idx}")
                continue
                    
            with st.expander(f"Spec {idx + 1} (ID: {result.get('spec_id', 'Unknown')})"):
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
                    'original_vs_baseline': {'wins': [0, 0], 'ties': 0},  # [original wins, baseline wins]
                    'original_vs_generated': {'wins': [0, 0], 'ties': 0}, # [original wins, generated wins]
                    'baseline_vs_generated': {'wins': [0, 0], 'ties': 0}  # [baseline wins, generated wins]
                }
                
                # Process comparisons and group by pair regardless of order
                comparison_pairs = {}  # Store all comparisons between each pair
                
                for comp in result.get('comparisons', []):
                    if not isinstance(comp, dict) or 'comparison' not in comp or 'score' not in comp:
                        continue
                        
                    # Get the original names and order
                    names = comp['comparison'].lower().split(' vs ')
                    order = comp.get('order', 1)
                    score = comp['score']
                    
                    # Always store pairs in a consistent order (original vs other)
                    # This ensures "original vs X" and "X vs original" are counted together
                    if 'original' in names:
                        if names[0] == 'original':
                            normalized_names = names
                            normalized_score = score
                        else:
                            normalized_names = [names[1], names[0]]  # Swap to put original first
                            normalized_score = 1.0 - score if score != 0.5 else score  # Flip score for non-ties
                    else:
                        # For baseline vs generated, always put baseline first
                        if 'baseline' in names[0]:
                            normalized_names = names
                            normalized_score = score
                        else:
                            normalized_names = [names[1], names[0]]
                            normalized_score = 1.0 - score if score != 0.5 else score
                    
                    pair_key = '_vs_'.join(normalized_names)
                    if pair_key not in summary:
                        summary[pair_key] = {'wins': [0, 0], 'ties': 0}
                    
                    # Count the result
                    if normalized_score == 0.5:
                        summary[pair_key]['ties'] += 1
                    elif normalized_score == 1.0:
                        summary[pair_key]['wins'][0] += 1  # First name wins
                    else:
                        summary[pair_key]['wins'][1] += 1  # Second name wins

                # Create two columns for the summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Summary")
                    
                    # Create bar chart for wins/losses/ties
                    comparison_data = []
                    for comp_type, stats in summary.items():
                        names = comp_type.split('_vs_')
                        # Only add non-zero counts
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
                        # Create a more informative bar chart
                        fig = px.bar(df, 
                                   x='Comparison', 
                                   y='Count',
                                   color='Winner',
                                   barmode='group',
                                   color_discrete_sequence=['#e74c3c', '#2ecc71', '#3498db'])  # Red, Green, Blue
                        
                        # Update layout for better readability
                        fig.update_layout(
                            xaxis_title="",
                            yaxis_title="Number of Comparisons",
                            legend_title="Winner",
                            margin=dict(t=20),  # Reduce top margin
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        # Ensure y-axis starts at 0 and shows whole numbers
                        fig.update_yaxes(rangemode="tozero", dtick=1)
                        # Make bars wider
                        fig.update_traces(width=0.2)
                        # Add unique key using spec ID
                        st.plotly_chart(fig, use_container_width=True, key=f"summary_chart_{result.get('spec_id', 'unknown')}")

                with col2:
                    st.markdown("#### Detailed Results")
                    detailed_data = []
                    for comparison in result.get('comparisons', []):
                        names = comparison['comparison'].split(' vs ')
                        # Get the order (1 for original order, 2 for reversed)
                        order = comparison.get('order', 1)
                        
                        # Convert score back to A/B/TIE
                        if comparison['score'] == 1.0:
                            winner = 'A' if order == 1 else 'B'  # If reversed, B means first code wins
                        elif comparison['score'] == 0.0:
                            winner = 'B' if order == 1 else 'A'  # If reversed, A means second code wins
                        else:
                            winner = 'TIE'
                            
                        # For order 2, reverse the display names
                        if order == 2:
                            names = names[::-1]
                            
                        # Determine the winner based on A/B/TIE
                        if winner == 'TIE':
                            result_text = "Tie"
                        elif winner == 'A':  # First code (in original comparison) wins
                            result_text = f"{names[0].title()} wins"
                        else:  # B wins - second code (in original comparison) wins
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

if __name__ == "__main__":
    main()