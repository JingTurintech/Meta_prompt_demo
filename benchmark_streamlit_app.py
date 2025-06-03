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
    JUDGE_PROMPT_TEMPLATE, META_PROMPT_TEMPLATES,
    AVAILABLE_LLMS, save_evaluation_results, load_evaluation_results
)
from loguru import logger
import sys
import colorsys
import numpy as np

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def generate_colors(n):
    """Generate n distinct colors using HSV color space.
    
    Args:
        n (int): Number of colors to generate
        
    Returns:
        list: List of hex color codes
    """
    colors = []
    for i in range(n):
        # Use golden ratio to space out hues evenly
        hue = i * 0.618033988749895 % 1
        # Keep saturation and value high for vibrant colors
        saturation = 0.7
        value = 0.95
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert RGB to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

def get_version_colors(versions):
    """Get color mapping for a list of versions.
    
    Args:
        versions (list): List of version names
        
    Returns:
        dict: Mapping of version names to colors
    """
    # Always put 'Original' first with gray color if it exists
    color_map = {}
    if 'original' in [v.lower() for v in versions]:
        color_map['Original'] = '#808080'  # Gray
        remaining_versions = [v for v in versions if v.lower() != 'original']
    else:
        remaining_versions = versions
    
    # Generate colors for remaining versions
    colors = generate_colors(len(remaining_versions))
    for version, color in zip(remaining_versions, colors):
        color_map[version.title()] = color
    
    return color_map

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
            "evaluation_count": 0,
            "snippet_containers": {},
            "current_containers": None,
            "saved_results": None,
            "collected_prompts": {},
            "prompts_displayed": False,
            "collected_templates": {},
            "templates_displayed": False,
            "template_order": [],
            "max_available_snippets": 0  # Track max available snippets across all benchmarks
        }
    
    # Initialize selected_templates if not present
    if "selected_templates" not in st.session_state:
        st.session_state.selected_templates = []

# Define consistent colors and order for versions
VERSION_ORDER = ["baseline", "standard", "simplified"]  # Define the fixed order
VERSION_COLORS = {
    'baseline': '#1f77b4',  # Blue
    'standard': '#17becf',  # Light blue
    'simplified': '#ff7f0e'  # Orange
}

# CSS for colored text areas
def get_colored_text_area_css():
    return """
        <style>
            /* Base style for text areas */
            .stTextArea textarea {
                background-color: transparent;
            }
            
            /* Custom classes for different versions */
            .baseline-bg textarea {
                background-color: rgba(31, 119, 180, 0.1) !important;  /* Blue with 10% opacity */
            }
            .standard-bg textarea {
                background-color: rgba(23, 190, 207, 0.1) !important;  /* Light blue with 10% opacity */
            }
            .simplified-bg textarea {
                background-color: rgba(255, 127, 14, 0.1) !important;  /* Orange with 10% opacity */
            }
        </style>
    """

def create_containers(num_benchmarks=1):
    """Create all containers once at app start"""
    containers = {
        "progress": st.empty(),
        "meta_prompt": st.container(),
        "prompts": st.container(),
        "snippets": st.container(),
        "final_results": st.container()
    }
    
    # For combined results, create containers for each benchmark
    if num_benchmarks > 1:
        for i in range(1, num_benchmarks + 1):
            containers[f"benchmark_{i}"] = st.container()
    
    return containers

def handle_progress_update(update_data: dict, containers):
    """Handle progress updates from the evaluator"""
    status = update_data.get("status")
    eval_count = st.session_state.app_state["evaluation_count"]
    timestamp = datetime.now().strftime("%H%M%S%f")
    
    if status == "setup":
        containers["progress"].info(update_data["message"])
    elif status == "setup_complete":
        containers["progress"].success("Setup complete!")
        st.session_state.app_state["collected_prompts"] = {}
        st.session_state.app_state["prompts_displayed"] = False
        st.session_state.app_state["collected_templates"] = {}
        st.session_state.app_state["templates_displayed"] = False
        # Store the selected templates order
        st.session_state.app_state["template_order"] = ["baseline"] + st.session_state.selected_templates
    elif status == "generating_meta_prompt":
        containers["progress"].info(update_data["message"])
    elif status == "meta_prompt_ready":
        template_id = update_data.get("template_id", "default")
        benchmark_idx = update_data.get("benchmark_idx", 1)  # Get benchmark index if provided
        benchmark_name = update_data.get("benchmark_name", "")  # Get benchmark name if provided
        
        # Store the template with benchmark info
        template_key = f"{benchmark_idx}_{template_id}" if benchmark_idx else template_id
        st.session_state.app_state["collected_templates"][template_key] = {
            "name": template_id.title(),
            "content": update_data["filled_meta_prompt"],
            "benchmark_name": benchmark_name
        }
        
        # Check if we have all templates for this benchmark
        template_order = st.session_state.app_state["template_order"][1:]  # Skip baseline
        benchmark_templates = [t for t in st.session_state.app_state["collected_templates"].keys() 
                             if t.startswith(f"{benchmark_idx}_")]
        
        if len(benchmark_templates) >= len(template_order):
            with containers["meta_prompt"]:
                st.markdown(f"### Filled Templates for {benchmark_name}")
                cols = st.columns(len(template_order))
                
                # Display templates side by side in selected order
                for idx, template_id in enumerate(template_order):
                    template_key = f"{benchmark_idx}_{template_id}"
                    if template_key in st.session_state.app_state["collected_templates"]:
                        template_data = st.session_state.app_state["collected_templates"][template_key]
                        with cols[idx]:
                            st.text_area(
                                label=f"**{template_data['name']}**",
                                value=template_data['content'],
                                height=400,
                                disabled=True,
                                key=f"template_{template_key}_{eval_count}_{timestamp}"
                            )
            
    elif status == "generating_prompt":
        containers["progress"].info(update_data["message"])
    elif status == "prompt_ready":
        template_id = update_data.get("template_id", "default")
        benchmark_idx = update_data.get("benchmark_idx", 1)  # Get benchmark index if provided
        benchmark_name = update_data.get("benchmark_name", "")  # Get benchmark name if provided
        
        # Store the prompt with benchmark info
        prompt_key = f"{benchmark_idx}_{template_id}" if benchmark_idx else template_id
        st.session_state.app_state["collected_prompts"][prompt_key] = {
            "name": template_id.title(),
            "content": update_data["generated_prompt"],
            "benchmark_name": benchmark_name
        }
        
        # Also store baseline prompt if not already stored
        baseline_key = f"{benchmark_idx}_baseline" if benchmark_idx else "baseline"
        if baseline_key not in st.session_state.app_state["collected_prompts"]:
            st.session_state.app_state["collected_prompts"][baseline_key] = {
                "name": "Baseline",
                "content": st.session_state.app_state["baseline_prompt"],
                "benchmark_name": benchmark_name
            }
        
        # Check if we have all prompts for this benchmark
        template_order = st.session_state.app_state["template_order"]
        benchmark_prompts = [p for p in st.session_state.app_state["collected_prompts"].keys() 
                           if p.startswith(f"{benchmark_idx}_")]
        
        if len(benchmark_prompts) >= len(template_order):
            with containers["prompts"]:
                st.markdown(f"### Generated Prompts for {benchmark_name}")
                cols = st.columns(len(template_order))
                
                # Display prompts side by side in selected order
                for idx, version_id in enumerate(template_order):
                    prompt_key = f"{benchmark_idx}_{version_id}"
                    if prompt_key in st.session_state.app_state["collected_prompts"]:
                        prompt_data = st.session_state.app_state["collected_prompts"][prompt_key]
                        with cols[idx]:
                            st.text_area(
                                label=f"**{prompt_data['name']}**",
                                value=prompt_data['content'],
                                height=200,
                                disabled=True,
                                key=f"prompt_{prompt_key}_{eval_count}_{timestamp}"
                            )
            
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
                # Create unique key for the snippet result
                unique_key = f"snippet_{snippet_id}_{eval_count}_{timestamp}"
                display_snippet_result(
                    result,
                    len(st.session_state.app_state["completed_snippets"]),
                    unique_key
                )
    elif status == "complete":
        final_results = update_data.get("final_results")
        if final_results:
            with containers["final_results"]:
                # Create unique key for final results
                unique_key = f"final_{eval_count}_{timestamp}"
                display_final_results(
                    final_results,
                    unique_key
                )
        containers["progress"].success("Evaluation complete!")
    elif status == "error":
        containers["progress"].error(update_data["message"])

# Initialize the static variable
handle_progress_update.prompts_header_shown = False

def display_snippet_result(result, snippet_number, key_suffix):
    """Display results for a single snippet"""
    # Validate result structure
    if not isinstance(result, dict):
        st.error(f"Invalid result format for snippet {snippet_number}")
        return
        
    snippet_id = result.get('snippet_id', f'Unknown-{snippet_number}')
    
    with st.expander(f"Snippet {snippet_number} (ID: {snippet_id})", expanded=False):
        # Display ratings if available
        if 'ratings' in result:
            st.markdown("### ELO Ratings")
            ratings = result['ratings']
            
            # Get dynamic color mapping for versions
            COLOR_MAP = get_version_colors(list(ratings.keys()))
            
            # Create DataFrame for ratings
            rating_data = pd.DataFrame([
                {"Version": version.title(), "Rating": rating}
                for version, rating in ratings.items()
            ])
            
            fig = px.bar(
                rating_data,
                x='Version',
                y='Rating',
                title="ELO Ratings for Different Versions",
                color='Version',
                color_discrete_map=COLOR_MAP
            )
            
            fig.update_layout(
                showlegend=False,
                margin=dict(t=40),
                yaxis_title="ELO Rating"
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"{key_suffix}_ratings_chart")
        
        # Display comparison results if available
        if 'comparisons' in result:
            st.markdown("### Comparison Results")
            
            # Calculate summary statistics
            version_wins = {
                'original': 0,
                'baseline': 0,
                **{template_id: 0 for template_id in result.get('optimized_versions', {}).keys()}
            }
            
            # Count wins for each version
            for comp in result.get('comparisons', []):
                if not isinstance(comp, dict) or 'comparison' not in comp or 'score' not in comp:
                    continue
                    
                names = comp['comparison'].split(' vs ')
                score = comp['score']
                
                if score == 1.0:  # First version wins
                    version_wins[names[0]] += 1
                elif score == 0.0:  # Second version wins
                    version_wins[names[1]] += 1
                # Ties are not counted as wins
            
            # Create two columns for the summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Summary of Wins")
                
                # Create DataFrame for wins
                wins_data = pd.DataFrame([
                    {"Version": version.title(), "Wins": wins}
                    for version, wins in version_wins.items()
                ])
                
                # Get color mapping for versions
                COLOR_MAP = get_version_colors(list(version_wins.keys()))
                
                fig = px.bar(
                    wins_data,
                    x='Version',
                    y='Wins',
                    color='Version',
                    color_discrete_map=COLOR_MAP
                )
                
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Number of Wins",
                    showlegend=False,
                    margin=dict(t=40)
                )
                fig.update_yaxes(rangemode="tozero", dtick=1)
                st.plotly_chart(fig, use_container_width=True, key=f"{key_suffix}_summary_chart")
            
            with col2:
                st.markdown("#### Detailed Results")
                detailed_data = []
                for comp in result.get('comparisons', []):
                    names = comp['comparison'].split(' vs ')
                    score = comp['score']
                    
                    if score == 1.0:
                        result_text = f"{names[0]} wins"
                    elif score == 0.0:
                        result_text = f"{names[1]} wins"
                    else:
                        result_text = "Tie"
                        
                    detailed_data.append({
                        "Comparison": comp['comparison'],
                        "Result": result_text
                    })
                
                df = pd.DataFrame(detailed_data)
                
                # Apply color styling to the table
                def color_winners(val):
                    if 'wins' not in val:
                        return ''
                    winner = val.split(' ')[0].lower()
                    colors = {version.lower(): f'background-color: {color}; color: white'
                             for version, color in COLOR_MAP.items()}
                    return colors.get(winner, '')
                
                styled_df = df.style.applymap(color_winners, subset=['Result'])
                st.table(styled_df)
        
        # Display code if available
        if 'original_code' in result or 'optimized_versions' in result:
            st.markdown("#### Code Comparison")
            versions = ["original"] + list(result.get('optimized_versions', {}).keys())
            cols = st.columns(len(versions))
            
            for col, version in zip(cols, versions):
                with col:
                    st.markdown(f"**{version.title()}**")
                    code = result['original_code'] if version == "original" else result.get('optimized_versions', {}).get(version, "Not available")
                    st.code(code)

def calculate_win_probabilities(ratings):
    """
    Calculate win probabilities between all pairs of versions based on ELO ratings
    """
    versions = list(ratings.keys())
    win_probs = {}
    
    for i, version_a in enumerate(versions):
        for j, version_b in enumerate(versions):
            if i != j:  # Don't compare a version with itself
                rating_a = ratings[version_a]
                rating_b = ratings[version_b]
                
                # ELO win probability formula
                prob_a_wins = 1 / (1 + 10**((rating_b - rating_a) / 400))
                
                key = f"{version_a} vs {version_b}"
                win_probs[key] = {
                    'probability': prob_a_wins,
                    'percentage': prob_a_wins * 100,
                    'rating_diff': rating_a - rating_b
                }
    
    return win_probs

def display_elo_interpretation(ratings, title_prefix=""):
    """
    Display ELO ratings in an easy-to-understand format with win probabilities
    """
    st.markdown(f"##### {title_prefix}ELO Rating Interpretation")
    st.markdown("*Converting ELO ratings into easy-to-understand win probabilities*")
    
    # Calculate win probabilities
    win_probs = calculate_win_probabilities(ratings)
    
    # Create a matrix of win probabilities
    versions = list(ratings.keys())
    
    # Sort versions by rating (highest first)
    sorted_versions = sorted(versions, key=lambda x: ratings[x], reverse=True)
    
    # Display rankings
    st.markdown("**📊 Performance Ranking (by ELO rating):**")
    ranking_data = []
    for i, version in enumerate(sorted_versions, 1):
        rating = ratings[version]
        ranking_data.append({
            'Rank': i,
            'Version': version.title(),
            'ELO Rating': f"{rating:.0f}",
            'Rating Interpretation': get_rating_interpretation(rating)
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)
    
    # Win probability matrix
    st.markdown("**🎯 Head-to-Head Win Probabilities:**")
    st.markdown("*Percentage chance that the row version beats the column version*")
    
    # Create matrix
    matrix_data = []
    for version_a in sorted_versions:
        row = {'Version': version_a.title()}
        for version_b in sorted_versions:
            if version_a == version_b:
                row[version_b.title()] = "—"
            else:
                key = f"{version_a} vs {version_b}"
                if key in win_probs:
                    percentage = win_probs[key]['percentage']
                    row[version_b.title()] = f"{percentage:.0f}%"
                else:
                    row[version_b.title()] = "—"
        matrix_data.append(row)
    
    matrix_df = pd.DataFrame(matrix_data)
    
    # Color the dataframe to highlight high probabilities
    def color_probabilities(val):
        if val == "—":
            return 'background-color: #f0f0f0'
        try:
            prob = float(val.replace('%', ''))
            if prob >= 70:
                return 'background-color: #d4edda; color: #155724; font-weight: bold'  # Strong advantage
            elif prob >= 60:
                return 'background-color: #fff3cd; color: #856404'  # Moderate advantage
            elif prob <= 30:
                return 'background-color: #f8d7da; color: #721c24'  # Disadvantage
            elif prob <= 40:
                return 'background-color: #ffeaa7; color: #6c5500'  # Slight disadvantage
            else:
                return 'background-color: #e2e3e5; color: #383d41'  # Close match
        except:
            return ''
    
    styled_df = matrix_df.style.applymap(color_probabilities, subset=matrix_df.columns[1:])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Explanation
    st.markdown("**💡 How to Read This:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Green (70%+)**: Strong performance advantage
        - **Yellow (60-69%)**: Moderate advantage  
        - **Gray (50-59%)**: Slight advantage
        """)
    
    with col2:
        st.markdown("""
        - **Light Orange (40-49%)**: Slight disadvantage
        - **Red (30-39%)**: Disadvantage
        - **Dark Red (<30%)**: Strong disadvantage
        """)
    
    # Key insights
    st.markdown("**🔍 Key Insights:**")
    
    # Find the best and worst performers
    best_version = sorted_versions[0]
    worst_version = sorted_versions[-1]
    
    best_rating = ratings[best_version]
    worst_rating = ratings[worst_version]
    rating_gap = best_rating - worst_rating
    
    # Calculate overall win rate of best vs worst
    best_vs_worst_prob = 1 / (1 + 10**((worst_rating - best_rating) / 400))
    
    insights = []
    insights.append(f"🥇 **{best_version.title()}** is the top performer with {best_rating:.0f} ELO")
    insights.append(f"🥉 **{worst_version.title()}** is the lowest performer with {worst_rating:.0f} ELO")
    insights.append(f"📊 Rating gap: {rating_gap:.0f} points")
    insights.append(f"⚡ **{best_version.title()}** would beat **{worst_version.title()}** approximately **{best_vs_worst_prob*100:.0f}%** of the time")
    
    # Find close competitors (within 50 points)
    close_matches = []
    for i in range(len(sorted_versions)-1):
        version_a = sorted_versions[i]
        version_b = sorted_versions[i+1]
        diff = ratings[version_a] - ratings[version_b]
        if diff <= 50:
            prob = 1 / (1 + 10**(-diff / 400))
            close_matches.append(f"🤏 **{version_a.title()}** vs **{version_b.title()}**: Very close ({prob*100:.0f}% vs {(1-prob)*100:.0f}%)")
    
    if close_matches:
        insights.append("**Close Competitions:**")
        insights.extend(close_matches)
    
    for insight in insights:
        st.markdown(f"- {insight}")

def get_rating_interpretation(rating):
    """
    Convert ELO rating to human-readable interpretation
    """
    if rating >= 1700:
        return "🏆 Excellent"
    elif rating >= 1600:
        return "🥇 Very Good"
    elif rating >= 1500:
        return "✅ Good"
    elif rating >= 1400:
        return "📊 Average"
    elif rating >= 1300:
        return "📉 Below Average"
    else:
        return "⚠️ Poor"

def compute_game_level_elo_ratings(all_results, initial_rating=1500):
    """
    Compute ELO ratings using all individual pairwise comparison results as games
    Using a simple, robust ELO implementation
    """
    # Simple ELO implementation to avoid convergence issues
    ratings = {}
    all_versions = set()
    all_comparisons = []
    comparison_stats = {}
    
    # Extract all comparison results from all snippets across all benchmarks
    for benchmark_result in all_results:
        for snippet_result in benchmark_result.get("results", []):
            all_versions.update(snippet_result["ratings"].keys())
            
            # Extract individual comparisons - each comparison is a game
            comparisons = snippet_result.get("comparisons", [])
            for comparison in comparisons:
                comp_str = comparison.get("comparison", "")
                score = comparison.get("score", 0.5)
                
                # Parse comparison string (e.g., "original vs baseline")
                if " vs " in comp_str:
                    parts = comp_str.split(" vs ")
                    if len(parts) == 2:
                        version_a, version_b = parts[0].strip(), parts[1].strip()
                        # Store as tuple: (version_a, version_b, score_for_a)
                        all_comparisons.append((version_a, version_b, score))
                        
                        # Track comparison statistics for debugging
                        comp_key = f"{version_a} vs {version_b}"
                        if comp_key not in comparison_stats:
                            comparison_stats[comp_key] = []
                        comparison_stats[comp_key].append(score)
    
    # Initialize all players with the same starting rating
    for version in all_versions:
        ratings[version] = initial_rating
    
    # Debug: Check if we have varied scores
    score_distribution = {}
    for _, _, score in all_comparisons:
        score_distribution[score] = score_distribution.get(score, 0) + 1
    
    # Get initial ratings for debugging
    initial_ratings = dict(ratings)
    
    # Simple ELO calculation with K-factor
    K = 32  # Standard ELO K-factor
    processed_games = 0
    
    for version_a, version_b, score in all_comparisons:
        # Skip if either version is not recognized
        if version_a not in all_versions or version_b not in all_versions:
            continue
        
        # Get current ratings
        rating_a = ratings[version_a]
        rating_b = ratings[version_b]
        
        # Calculate expected scores
        expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a
        
        # Actual scores based on comparison result
        if score == 1.0:  # version_a wins
            actual_a, actual_b = 1.0, 0.0
        elif score == 0.0:  # version_b wins
            actual_a, actual_b = 0.0, 1.0
        else:  # tie (score == 0.5)
            actual_a, actual_b = 0.5, 0.5
        
        # Update ratings
        new_rating_a = rating_a + K * (actual_a - expected_a)
        new_rating_b = rating_b + K * (actual_b - expected_b)
        
        # Apply bounds to prevent extreme values
        ratings[version_a] = max(0, min(3000, new_rating_a))
        ratings[version_b] = max(0, min(3000, new_rating_b))
        
        processed_games += 1
    
    # Debug information
    debug_info = {
        'total_comparisons': len(all_comparisons),
        'processed_games': processed_games,
        'score_distribution': score_distribution,
        'initial_ratings': initial_ratings,
        'final_ratings': dict(ratings),
        'comparison_stats': comparison_stats
    }
    
    return dict(ratings), len(all_comparisons), debug_info

def display_overall_combined_results(combined_results, key_suffix):
    """
    Special display function for overall combined benchmark results showing multiple ELO rating levels
    """
    # Get versions and color mapping
    avg_ratings = combined_results["overall_average_ratings"]
    COLOR_MAP = get_version_colors(list(avg_ratings.keys()))
    
    # Collect all individual snippet results
    all_snippet_results = []
    all_benchmark_results = []
    for benchmark in combined_results.get("benchmarks", []):
        all_snippet_results.extend(benchmark.get("results", []))
        all_benchmark_results.append(benchmark)
    
    total_snippets = len(all_snippet_results)
    
    # 1. Snippet-Level Box Plot (Distribution across all snippets)
    st.markdown("##### Snippet-Level ELO Rating Distribution")
    st.markdown("*Shows distribution of ELO ratings across all individual code snippets*")
    
    if all_snippet_results:
        # Create box plot data
        box_data = []
        for result in all_snippet_results:
            for version, rating in result['ratings'].items():
                box_data.append({
                    'Version': version.title(),
                    'Rating': rating
                })
        
        if box_data:
            box_df = pd.DataFrame(box_data)
            
            # Create box plot
            fig_snippet = px.box(
                box_df,
                x='Version',
                y='Rating',
                title=f"Snippet-Level ELO Rating Distribution ({total_snippets} snippets)",
                color='Version',
                color_discrete_map=COLOR_MAP,
                points="all",
                hover_data={'Rating': ':.1f'}
            )
            
            fig_snippet.update_traces(showlegend=False)
            
            # Add mean markers
            for version in avg_ratings.keys():
                version_data = box_df[box_df['Version'] == version.title()]
                if not version_data.empty:
                    mean_rating = version_data['Rating'].mean()
                    fig_snippet.add_scatter(
                        x=[version.title()],
                        y=[mean_rating],
                        mode='markers',
                        marker=dict(
                            symbol='diamond',
                            size=12,
                            color='white',
                            line=dict(color=COLOR_MAP.get(version, 'black'), width=2)
                        ),
                        name=f'Mean',
                        showlegend=False,
                        hovertemplate=f'Mean: {mean_rating:.1f}<extra></extra>'
                    )
            
            fig_snippet.update_layout(
                showlegend=False,
                yaxis_title="ELO Rating",
                margin=dict(t=40),
                boxmode='group'
            )
            
            st.plotly_chart(fig_snippet, use_container_width=True, key=f"snippet_level_{key_suffix}")
            
    # Display summary statistics for snippet-level
    if all_snippet_results:
        st.markdown("#### Summary Statistics (Snippet-Level)")
        box_data = []
        for result in all_snippet_results:
            for version, rating in result['ratings'].items():
                box_data.append({
                    'Version': version.title(),
                    'Rating': rating
                })
        
        if box_data:
            box_df = pd.DataFrame(box_data)
            stats_data = []
            for version in avg_ratings.keys():
                version_data = box_df[box_df['Version'] == version.title()]
                if not version_data.empty:
                    ratings = version_data['Rating'].values
                    stats_data.append({
                        'Version': version.title(),
                        'Mean': f"{np.mean(ratings):.1f}",
                        'Std Dev': f"{np.std(ratings):.1f}",
                        'Min': f"{np.min(ratings):.1f}",
                        'Max': f"{np.max(ratings):.1f}",
                        'Median': f"{np.median(ratings):.1f}",
                        'Count': len(ratings)
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # 2. ELO Rating Progression Chart (across all snippets)
    if all_snippet_results:
        # Create progression data
        progression_data = []
        for i, result in enumerate(all_snippet_results, 1):
            for version, rating in result['ratings'].items():
                progression_data.append({
                    'Snippet': i,
                    'Version': version.title(),
                    'Rating': rating
                })
        
        if progression_data:
            progression_df = pd.DataFrame(progression_data)
            
            fig_progression = px.line(
                progression_df,
                x='Snippet',
                y='Rating',
                color='Version',
                title=f"ELO Rating Progression Across {total_snippets} Snippets",
                markers=True,
                color_discrete_map=COLOR_MAP
            )
            
            fig_progression.update_layout(
                xaxis_title="Snippet Number",
                yaxis_title="ELO Rating",
                showlegend=True
            )
            
            st.plotly_chart(fig_progression, use_container_width=True, key=f"progression_{key_suffix}")
    
    # 3. Project-Level Box Plot (Distribution across projects)
    st.markdown("##### Project-Level ELO Rating Distribution")
    st.markdown("*Shows distribution of average ELO ratings across different projects/benchmarks*")
    
    project_box_data = []
    for benchmark in combined_results.get("benchmarks", []):
        project_name = benchmark.get("benchmark_info", {}).get("project_info", {}).get("name", "Unknown")
        benchmark_avg_ratings = benchmark.get("average_ratings", {})
        
        for version, rating in benchmark_avg_ratings.items():
            project_box_data.append({
                'Version': version.title(),
                'Rating': rating,
                'Project': project_name
            })
    
    if project_box_data:
        project_box_df = pd.DataFrame(project_box_data)
        
        fig_project = px.box(
            project_box_df,
            x='Version',
            y='Rating',
            title=f"Project-Level ELO Rating Distribution ({len(all_benchmark_results)} projects)",
            color='Version',
            color_discrete_map=COLOR_MAP,
            points="all",
            hover_data={'Rating': ':.1f', 'Project': True}
        )
        
        fig_project.update_traces(showlegend=False)
        fig_project.update_layout(
            showlegend=False,
            yaxis_title="ELO Rating",
            margin=dict(t=40),
            boxmode='group'
        )
        
        st.plotly_chart(fig_project, use_container_width=True, key=f"project_level_{key_suffix}")
    
    # 4. Game-Level Bar Chart (All pairwise comparisons)
    st.markdown("##### Game-Level ELO Rating (All Pairwise Comparisons)")
    st.markdown("*Shows ELO ratings computed using all individual pairwise comparison results as separate games*")
    
    # Check if we have comparison data
    has_comparison_data = any(
        any('comparisons' in result and result['comparisons'] for result in benchmark.get("results", []))
        for benchmark in all_benchmark_results
    )
    
    if has_comparison_data:
        game_level_ratings, total_games, debug_info = compute_game_level_elo_ratings(all_benchmark_results)
        
        # Create bar chart for game-level ratings
        game_rating_data = pd.DataFrame([
            {"Version": version.title(), "Rating": rating}
            for version, rating in game_level_ratings.items()
        ])
        
        fig_game = px.bar(
            game_rating_data,
            x='Version',
            y='Rating',
            title=f"Game-Level ELO Ratings ({total_games} total games)",
            color='Version',
            color_discrete_map=COLOR_MAP
        )
        
        # Add value labels above each bar
        fig_game.update_traces(
            texttemplate='%{y:.1f}',
            textposition='outside',
            textfont=dict(size=14),
            showlegend=False
        )
        
        fig_game.update_layout(
            showlegend=False,
            yaxis_title="ELO Rating",
            margin=dict(t=40)
        )
        
        st.plotly_chart(fig_game, use_container_width=True, key=f"game_level_{key_suffix}")
        
        # Display comparison table
        st.markdown("##### Average ELO Rating Comparison Across Levels")
        
        # Calculate true snippet-level and project-level ratings
        num_benchmarks = len(all_benchmark_results)
        total_snippets = sum(len(bench.get("results", [])) for bench in all_benchmark_results)
        
        st.markdown(f"**Calculation Details:** {num_benchmarks} benchmarks, {total_snippets} total snippets")
        
        comparison_data = []
        for version in avg_ratings.keys():
            # TRUE Snippet-Level: Average of ALL individual snippet ratings across all benchmarks
            all_snippet_ratings_for_version = []
            for bench in all_benchmark_results:
                for snippet_result in bench.get("results", []):
                    if version in snippet_result.get("ratings", {}):
                        all_snippet_ratings_for_version.append(snippet_result["ratings"][version])
            
            snippet_level_avg = sum(all_snippet_ratings_for_version) / len(all_snippet_ratings_for_version) if all_snippet_ratings_for_version else 0
            
            # TRUE Project-Level: Average of each benchmark's average rating for this version
            project_averages = []
            for bench in all_benchmark_results:
                bench_avg = bench.get("average_ratings", {}).get(version)
                if bench_avg is not None:
                    project_averages.append(bench_avg)
            
            project_level_avg = sum(project_averages) / len(project_averages) if project_averages else 0
            
            comparison_data.append({
                'Version': version.title(),
                'Snippet-Level (Overall Avg)': f"{snippet_level_avg:.1f}",
                'Project-Level (Avg)': f"{project_level_avg:.1f}",
                'Game-Level': f"{game_level_ratings.get(version, 1500):.1f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Add explanation of the differences
        st.markdown("**📖 Level Explanations:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Snippet-Level:**
            - Average of ALL individual snippet ratings
            - Treats each code snippet equally
            - Most granular view
            """)
        
        with col2:
            st.markdown("""
            **Project-Level:**
            - Average of each benchmark's average
            - Treats each project/benchmark equally  
            - Balances across different projects
            """)
        
        with col3:
            st.markdown("""
            **Game-Level:**
            - Uses all pairwise comparisons as games
            - Different ELO calculation method
            - Reflects head-to-head performance
            """)
        
        # Show when they would be different
        if num_benchmarks > 1:
            st.markdown("**💡 When Project-Level ≠ Snippet-Level:**")
            st.markdown("- When different benchmarks have different numbers of snippets")
            st.markdown("- When some benchmarks consistently rate versions differently")
            st.markdown("- Project-level gives equal weight to each benchmark regardless of snippet count")
        
        # 5. ELO Rating Interpretation (using Game-Level ratings)
        st.markdown("---")
        display_elo_interpretation(game_level_ratings, "Game-Level ")
        
    else:
        st.warning("Individual comparison data not available for game-level ELO computation.")
        # Fallback to snippet-level interpretation if game-level unavailable
        st.markdown("---")
        display_elo_interpretation(avg_ratings, "Snippet-Level ")
    


def display_final_results(final_results, key_suffix):
    """Display final results with box plots showing rating distribution"""
    avg_ratings = final_results['average_ratings']
    
    # Get dynamic color mapping for versions
    COLOR_MAP = get_version_colors(list(avg_ratings.keys()))
    
    # Determine number of snippets for title
    num_snippets = len(final_results.get('results', [])) if 'results' in final_results else ''
    title_suffix = f" Across {num_snippets} Snippets" if num_snippets else ""
    
    if 'results' in final_results and final_results['results']:
        # Create box plot showing distribution of ratings across all snippets
        box_data = []
        for result in final_results['results']:
            for version, rating in result['ratings'].items():
                box_data.append({
                    'Version': version.title(),
                    'Rating': rating
                })
        
        if box_data:
            box_df = pd.DataFrame(box_data)
            
            # Create box plot
            fig = px.box(
                box_df,
                x='Version',
                y='Rating',
                title=f"ELO Rating Distribution{title_suffix}",
                color='Version',
                color_discrete_map=COLOR_MAP,
                points="all",  # Show all individual points
                hover_data={'Rating': ':.1f'}
            )
            
            # Explicitly disable legend for all traces
            fig.update_traces(showlegend=False)
            
            # Add mean markers
            for version in avg_ratings.keys():
                version_data = box_df[box_df['Version'] == version.title()]
                if not version_data.empty:
                    mean_rating = version_data['Rating'].mean()
                    fig.add_scatter(
                        x=[version.title()],
                        y=[mean_rating],
                        mode='markers',
                        marker=dict(
                            symbol='diamond',
                            size=12,
                            color='white',
                            line=dict(color=COLOR_MAP.get(version, 'black'), width=2)
                        ),
                        name=f'Mean',
                        showlegend=False,
                        hovertemplate=f'Mean: {mean_rating:.1f}<extra></extra>'
                    )
            
            fig.update_layout(
                showlegend=False,
                yaxis_title="ELO Rating",
                margin=dict(t=40),
                boxmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"box_ratings_{key_suffix}")
            
            # Display summary statistics table
            st.markdown("#### Summary Statistics")
            stats_data = []
            for version in avg_ratings.keys():
                version_data = box_df[box_df['Version'] == version.title()]
                if not version_data.empty:
                    ratings = version_data['Rating'].values
                    stats_data.append({
                        'Version': version.title(),
                        'Mean': f"{np.mean(ratings):.1f}",
                        'Std Dev': f"{np.std(ratings):.1f}",
                        'Min': f"{np.min(ratings):.1f}",
                        'Max': f"{np.max(ratings):.1f}",
                        'Median': f"{np.median(ratings):.1f}",
                        'Count': len(ratings)
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Add ELO interpretation section
        st.markdown("---")
        display_elo_interpretation(avg_ratings)
    
    else:
        # Fallback to bar chart if no individual results available
        st.warning("Individual snippet results not available. Showing average ratings only.")
        rating_data = pd.DataFrame([
            {"Version": version.title(), "Rating": rating}
            for version, rating in avg_ratings.items()
        ])
        
        fig = px.bar(
            rating_data,
            x='Version',
            y='Rating',
            title=f"Average ELO Ratings{title_suffix}",
            color='Version',
            color_discrete_map=COLOR_MAP
        )
        
        # Add value labels above each bar
        fig.update_traces(
            texttemplate='%{y:.1f}',  # Show one decimal place
            textposition='outside',
            textfont=dict(size=14)
        )
        
        fig.update_layout(
            showlegend=False,
            yaxis_title="ELO Rating",
            margin=dict(t=40)
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"avg_ratings_{key_suffix}")
        
        # Add ELO interpretation for fallback case too
        st.markdown("---")
        display_elo_interpretation(avg_ratings)
    
    if 'results' in final_results:
        # Create progression chart
        progression_data = []
        for i, r in enumerate(final_results['results'], 1):
            for version, rating in r['ratings'].items():
                progression_data.append({
                    'Snippet': i,
                    'Version': version.title(),
                    'Rating': rating
                })
        
        progression_df = pd.DataFrame(progression_data)
        
        fig_progression = px.line(
            progression_df,
            x='Snippet',
            y='Rating',
            color='Version',
            title=f"ELO Rating Progression Across {len(final_results['results'])} Snippets",
            markers=True,
            color_discrete_map=COLOR_MAP
        )
        
        fig_progression.update_layout(
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
        
    saved_files = [f for f in os.listdir("results") if f.endswith('.json')]
    if not saved_files:
        st.sidebar.warning("No saved results found")
        return
    
    # Sort files by timestamp (newest first)
    saved_files.sort(reverse=True)
    
    selected_file = st.sidebar.selectbox(
        "Select Saved Results",
        options=saved_files,
        format_func=lambda x: x.replace('.json', '').replace('_', ' ')
    )
    
    if st.sidebar.button("Load Selected Results"):
        filepath = os.path.join("results", selected_file)
        results = load_evaluation_results(filepath)
        
        if results:
            st.session_state.app_state["saved_results"] = results
            st.session_state.app_state["has_evaluated"] = True
            
            # Check if this is a combined results file
            is_combined = "benchmarks" in results
            num_benchmarks = len(results.get("benchmarks", [])) if is_combined else 1
            
            # Create new containers for displaying results
            containers = create_containers(num_benchmarks)
            st.session_state.app_state["current_containers"] = containers
            
            # Get configuration data
            if is_combined and results.get("benchmarks"):
                # For combined results, use the first benchmark's configuration
                first_benchmark = results["benchmarks"][0]
                config = {
                    "task_name": first_benchmark.get("task_name"),
                    "task_description": first_benchmark.get("task_description"),
                    "task_objective": first_benchmark.get("task_objective"),
                    "task_considerations": first_benchmark.get("task_considerations"),
                    "llm_type": first_benchmark.get("llm_type"),
                    "judge_llm_type": first_benchmark.get("judge_llm_type"),
                    "synthesis_llm_type": first_benchmark.get("synthesis_llm_type"),  # Add synthesis LLM type
                    "baseline_prompt": first_benchmark.get("prompts", {}).get("baseline"),
                    "selected_templates": first_benchmark.get("selected_templates", []),
                    "statistics": results.get("overall_statistics", {}),
                    "evaluation_timestamp": results.get("evaluation_timestamp")
                }
            else:
                # For single benchmark results, use the configuration section
                config = results.get("configuration", {})
            
            # Display configuration in sidebar
            st.sidebar.markdown("### Evaluation Configuration")
            
            # Task Information
            st.sidebar.markdown("#### Task Details")
            st.sidebar.text(f"Task: {config.get('task_name', 'Unknown')}")
            
            description = config.get('task_description')
            if description:
                st.sidebar.text_area("Description", description, disabled=True)
            
            objective = config.get('task_objective')
            if objective:
                st.sidebar.text_area("Objective", objective, disabled=True)
            
            # LLM Configuration
            st.sidebar.markdown("#### LLM Configuration")
            st.sidebar.text(f"Optimization LLM: {str(config.get('llm_type', 'Unknown')).replace('LLMType.', '')}")
            st.sidebar.text(f"Judge LLM: {str(config.get('judge_llm_type', 'Unknown')).replace('LLMType.', '')}")
            st.sidebar.text(f"Synthesis LLM: {str(config.get('synthesis_llm_type', 'Unknown')).replace('LLMType.', '')}")
            
            # Templates Used
            st.sidebar.markdown("#### Templates Used")
            selected_templates = config.get('selected_templates', [])
            if selected_templates:
                for template in selected_templates:
                    st.sidebar.text(f"• {template}")
            else:
                st.sidebar.text("No templates specified")
            
            # Prompts
            st.sidebar.markdown("#### Prompts")
            baseline_prompt = config.get('baseline_prompt')
            if baseline_prompt:
                st.sidebar.text_area("Baseline Prompt", baseline_prompt, disabled=True)
            else:
                st.sidebar.text("No baseline prompt available")
            
            # Statistics
            st.sidebar.markdown("#### Evaluation Statistics")
            stats = config.get('statistics', {})
            st.sidebar.text(f"Total Snippets: {stats.get('total_snippets', 0)}")
            st.sidebar.text(f"Successful: {stats.get('successful_snippets', 0)}")
            st.sidebar.text(f"Failed: {stats.get('failed_snippets', 0)}")
                
            # Timestamp
            st.sidebar.markdown("#### Evaluation Time")
            st.sidebar.text(f"Timestamp: {config.get('evaluation_timestamp', 'Unknown')}")
            
            if is_combined:
                # Display results for each benchmark first
                for idx, benchmark in enumerate(results.get("benchmarks", []), 1):
                    benchmark_name = benchmark.get('benchmark_info', {}).get('project_info', {}).get('name', 'Unknown')
                    with containers[f"benchmark_{idx}"]:
                        st.markdown(f"### Results for Benchmark {idx}: {benchmark_name}")
                        
                        # Display meta-prompts for this benchmark
                        st.markdown("#### Meta-Prompts Used")
                        meta_prompts = benchmark.get("meta_prompts", {})
                        if meta_prompts:
                            cols = st.columns(len(meta_prompts))
                            for col_idx, (template_id, meta_prompt_info) in enumerate(meta_prompts.items()):
                                with cols[col_idx]:
                                    st.text_area(
                                        f"**{meta_prompt_info['name']}**",
                                        value=meta_prompt_info['filled_template'],
                                        height=400,
                                        disabled=True,
                                        key=f"loaded_template_{template_id}_{idx}"
                                    )
                        
                        # Display prompts for this benchmark
                        st.markdown("#### Generated Prompts")
                        prompts = benchmark.get("prompts", {})
                        if prompts:
                            cols = st.columns(len(prompts))
                            for col_idx, (prompt_type, prompt_content) in enumerate(prompts.items()):
                                with cols[col_idx]:
                                    st.text_area(
                                        f"**{prompt_type.title()}**",
                                        value=prompt_content,
                                        height=200,
                                        disabled=True,
                                        key=f"loaded_prompt_{prompt_type}_{idx}"
                                    )
                        
                        # Display snippet results for this benchmark
                        st.markdown("#### Individual Snippet Results")
                        for snippet_idx, result in enumerate(benchmark.get("results", []), 1):
                            display_snippet_result(result, snippet_idx, f"snippet_{result.get('snippet_id', str(snippet_idx))}_loaded_{idx}")
                        
                        # Display average ratings for this benchmark
                        st.markdown(f"#### Average Evaluation Results")
                        display_final_results(benchmark, f"loaded_{idx}")
                
                # Display overall results at the end
                st.markdown("### Overall Results Across All Benchmarks")
                
                # Display overall statistics
                st.markdown("#### Overall Statistics")
                stats = results.get("overall_statistics", {})
                st.write(f"- Total Benchmarks: {stats.get('total_benchmarks', 0)}")
                st.write(f"- Total Snippets: {stats.get('total_snippets', 0)}")
                st.write(f"- Successful Snippets: {stats.get('successful_snippets', 0)}")
                st.write(f"- Failed Snippets: {stats.get('failed_snippets', 0)}")
                
                # Display overall average ratings
                if "overall_average_ratings" in results:
                    with containers["final_results"]:
                        total_snippets = results["overall_statistics"]["successful_snippets"]
                        st.markdown(f"#### Overall Average Ratings Across {total_snippets} Snippets")
                        
                        # Collect all individual snippet results for box plot functionality
                        all_snippet_results = []
                        for benchmark in results.get("benchmarks", []):
                            all_snippet_results.extend(benchmark.get("results", []))
                        
                        display_overall_combined_results(results, "overall_combined")
            else:
                # Handle single benchmark results as before
                # Display meta-prompt if available
                with containers["meta_prompt"]:
                    st.markdown("### Meta-Prompt Used")
                    meta_prompts = results.get("meta_prompts", {})
                    if meta_prompts:
                        cols = st.columns(len(meta_prompts))
                        for idx, (template_id, meta_prompt_info) in enumerate(meta_prompts.items()):
                            with cols[idx]:
                                st.text_area(
                                    f"**{meta_prompt_info['name']}**",
                                    value=meta_prompt_info['filled_template'],
                                    height=400,
                                    disabled=True,
                                    key=f"loaded_template_{template_id}"
                                )
                    else:
                        st.info("Meta-prompt information not available in this saved result")
                
                # Display prompts
                with containers["prompts"]:
                    st.markdown("### Generated Prompts")
                    prompts = config.get("generated_prompts", {}) or results.get("prompts", {})
                    if prompts:
                        cols = st.columns(len(prompts))
                        for idx, (prompt_type, prompt_content) in enumerate(prompts.items()):
                            with cols[idx]:
                                st.text_area(
                                    f"**{prompt_type.title()}**",
                                    value=prompt_content,
                                    height=200,
                                    disabled=True,
                                    key=f"loaded_prompt_{prompt_type}"
                                )
                    else:
                        st.info("No prompts available in this saved result")
                
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
    max_available_snippets = 0
    for benchmark_file in selected_benchmarks:
        try:
            with open(os.path.join("benchmarks", benchmark_file), 'r') as f:
                benchmark_data = json.load(f)
            
            project_info = benchmark_data["metadata"]["project_info"]
            project_id = project_info.get('project_id', 'Unknown')
            artemis_url = f"https://artemis.turintech.ai/projects/{project_id}"
            num_snippets = len(benchmark_data.get('code_snippets', []))
            max_available_snippets = max(max_available_snippets, num_snippets)
            
            st.sidebar.markdown(f"""
            **{project_info.get('name', 'Unknown')}**
            - Description: {project_info.get('description', 'No description available')}
            - Language: {project_info.get('language', 'unknown')}
            - Code Snippets: {num_snippets}
            - Artemis URL: [@{project_id}]({artemis_url})
            ---
            """)
            
        except Exception as e:
            st.sidebar.error(f"Error loading {benchmark_file}: {str(e)}")
    
    # Update max available snippets in session state
    st.session_state.app_state["max_available_snippets"] = max_available_snippets
    
    # Add slider for number of snippets to evaluate
    num_snippets = st.sidebar.slider(
        "Number of Snippets to Evaluate",
        min_value=1,
        max_value=max_available_snippets,
        value=min(2, max_available_snippets),  # Default to 5 or max available if less
        help="Select how many snippets to evaluate from each benchmark. If a benchmark has fewer snippets, all available snippets will be used."
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
    
    # Synthesis LLM selection
    selected_synthesis_llm = st.sidebar.selectbox(
        "Select Prompt Synthesis LLM",
        options=[llm.value for llm in AVAILABLE_LLMS],
        index=0,  # Default to gpt-4-o-mini
        help="LLM that will generate optimization prompts from meta-prompt templates"
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
    st.sidebar.markdown("### Meta-Prompt Templates")
    st.sidebar.markdown("Select one or more templates to use for generating optimization prompts.")
    
    # Store selected templates in session state
    st.session_state.selected_templates = st.sidebar.multiselect(
        "Select Meta-Prompt Templates",
        options=list(META_PROMPT_TEMPLATES.keys()),
        default=["standard"],
        format_func=lambda x: META_PROMPT_TEMPLATES[x]["name"]
    )
    
    # Show descriptions of selected templates
    for template_id in st.session_state.selected_templates:
        template = META_PROMPT_TEMPLATES[template_id]
        with st.sidebar.expander(f"About {template['name']}"):
            st.markdown(template["description"])
    
    # Custom meta-prompt input (optional)
    if st.sidebar.checkbox("Use Custom Meta-Prompt Template"):
        custom_meta_prompt = st.sidebar.text_area(
            "Custom Meta-Prompt Template",
            value=META_PROMPT_TEMPLATES["standard"]["template"],
            height=400,
            help="Customize how the meta-prompt is generated. Available placeholders: {objective}, {project_name}, {project_description}, {project_languages}, {task_description}, {current_prompt}, {target_llm}"
        )
    else:
        custom_meta_prompt = None

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
        st.session_state.app_state["snippet_containers"] = {}
        st.session_state.app_state["collected_prompts"] = {}
        st.session_state.app_state["prompts_displayed"] = False
        st.session_state.app_state["collected_templates"] = {}
        st.session_state.app_state["templates_displayed"] = False
        
        # Clear any previous progress messages
        containers["progress"].empty()
        
        with st.spinner("Running optimization evaluation..."):
            # Process each selected benchmark
            all_results = []
            combined_results = None
            
            for benchmark_file in selected_benchmarks:
                # Use the progress container for status updates
                containers["progress"].info(f"Processing {benchmark_file}")
                
                # Load benchmark data and limit snippets
                with open(os.path.join("benchmarks", benchmark_file), 'r') as f:
                    benchmark_data = json.load(f)
                # Limit the number of snippets according to the slider
                benchmark_data['code_snippets'] = benchmark_data['code_snippets'][:num_snippets]
                
                evaluator = BenchmarkEvaluator(
                    task_name=selected_task,
                    llm_type=LLMType(selected_llm),
                    judge_llm_type=LLMType(selected_judge_llm),
                    synthesis_llm_type=LLMType(selected_synthesis_llm),
                    current_prompt=custom_baseline_prompt,
                    custom_task_description=custom_task_description,
                    custom_meta_prompt=custom_meta_prompt,
                    selected_templates=st.session_state.selected_templates,
                    progress_callback=lambda x: handle_progress_update(x, containers)
                )
                
                results = asyncio.run(evaluator.evaluate_benchmark(benchmark_data))
                
                if results:
                    all_results.append(results)
                    
                    # Save individual results
                    saved_file = save_evaluation_results(results)
                    containers["progress"].success(f"Results for {benchmark_file} saved to: {saved_file}")
                    
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
                else:
                    containers["progress"].error(f"Failed to evaluate {benchmark_file}")
            
            # Only save combined results if there are multiple benchmarks
            if len(all_results) > 1:
                # Calculate overall average ratings across all snippets
                all_versions = set()
                all_snippet_ratings = {}
                
                # First, collect all version names and their ratings from each snippet
                for result in all_results:
                    versions = result["average_ratings"].keys()
                    all_versions.update(versions)
                    for version in versions:
                        if version not in all_snippet_ratings:
                            all_snippet_ratings[version] = []
                        # Add individual snippet ratings for this version
                        for snippet_result in result["results"]:
                            all_snippet_ratings[version].append(snippet_result["ratings"][version])
                
                # Calculate overall averages across all snippets
                overall_average_ratings = {
                    version: sum(ratings) / len(ratings)
                    for version, ratings in all_snippet_ratings.items()
                }
                
                # Combine results from all benchmarks
                combined_results = {
                    "benchmarks": all_results,
                    "overall_statistics": {
                        "total_benchmarks": len(all_results),
                        "total_snippets": sum(r["statistics"]["total_snippets"] for r in all_results),
                        "successful_snippets": sum(r["statistics"]["successful_snippets"] for r in all_results),
                        "failed_snippets": sum(r["statistics"]["failed_snippets"] for r in all_results)
                    },
                    "overall_average_ratings": overall_average_ratings
                }
                
                # Create a descriptive filename using benchmark names
                benchmark_names = [os.path.splitext(bf)[0].split('_')[0] for bf in selected_benchmarks]
                combined_name = '_'.join(sorted(benchmark_names))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                combined_filename = f"combined_{combined_name}_{selected_task}_{timestamp}.json"
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
                
                # Display overall average ratings
                with containers["final_results"]:
                    total_snippets = combined_results["overall_statistics"]["successful_snippets"]
                    st.markdown(f"### Overall Average Ratings Across {total_snippets} Snippets")
                    
                    # Collect all individual snippet results for box plot functionality
                    all_snippet_results = []
                    for benchmark in combined_results.get("benchmarks", []):
                        all_snippet_results.extend(benchmark.get("results", []))
                    
                    overall_results_data = {
                        "average_ratings": overall_average_ratings,
                        "results": all_snippet_results
                    }
                    display_overall_combined_results(combined_results, f"overall_{timestamp}")
            
            st.session_state.app_state["optimization_results"] = all_results[0] if len(all_results) == 1 else combined_results
            st.session_state.app_state["has_evaluated"] = True

if __name__ == "__main__":
    main() 