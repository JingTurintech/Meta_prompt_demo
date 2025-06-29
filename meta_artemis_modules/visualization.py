"""
Visualization and results analysis functions for the Meta Artemis application.
Handles displaying results, creating charts, and analyzing performance data.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from .utils import generate_colors, get_session_state
import json
from plotly.subplots import make_subplots
from scipy import stats

# Add statistical analysis imports
import warnings
warnings.filterwarnings('ignore')

# scipy is needed for effect size calculations
SCIPY_AVAILABLE = True


def perform_statistical_analysis(versions_data: Dict[str, List[float]], alpha: float = 0.05, effect_size_threshold: float = 0.2) -> Dict[str, Any]:
    """
    Perform statistical analysis to group similar versions using:
    - Mann-Whitney U test for non-parametric statistical comparison
    - Cohen's d for effect size measurement
    
    Effect size thresholds (Cohen's d):
    - Negligible: |d| < 0.2
    - Small: 0.2 ≤ |d| < 0.5
    - Medium: 0.5 ≤ |d| < 0.8
    - Large: |d| ≥ 0.8
    
    We consider versions different if:
    1. They are statistically different (p ≤ alpha) OR
    2. They have non-negligible effect size (|d| ≥ effect_size_threshold)
    
    Args:
        versions_data: Dictionary with version names as keys and runtime lists as values
        alpha: Significance level (default 0.05)
        effect_size_threshold: Threshold for considering effect size non-negligible (default 0.2)
                             Versions with effect size ≥ this threshold will be considered different
    
    Returns:
        Dictionary containing test results, groups, and rankings
    """

    # Filter out empty versions
    filtered_data = {k: v for k, v in versions_data.items() if v and len(v) > 0}
    
    if len(filtered_data) < 2:
        logger.warning("⚠️ Need at least 2 versions with data for statistical testing")
        return {
            "success": False,
            "error": "Insufficient data for statistical testing",
            "groups": {},
            "rankings": {},
            "p_values": {}
        }
    
    try:
        version_names = list(filtered_data.keys())
        
        # Calculate means and variances
        means = {version: np.mean(data) for version, data in filtered_data.items()}
        
        # Sort versions by mean performance (lower is better)
        sorted_versions = sorted(means.keys(), key=lambda x: means[x])
        
        # Initialize groups
        current_group = 1
        groups = {}
        rankings = {}
        
        # Helper function for Cohen's d effect size
        def cohen_d(data1, data2):
            n1, n2 = len(data1), len(data2)
            var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
            pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            return (np.mean(data1) - np.mean(data2)) / pooled_se if pooled_se != 0 else 0
        
        # Group versions based on statistical similarity
        i = 0
        while i < len(sorted_versions):
            current_version = sorted_versions[i]
            current_data = filtered_data[current_version]
            
            # Start a new group
            current_group_versions = [current_version]
            
            # Compare with remaining versions
            j = i + 1
            while j < len(sorted_versions):
                next_version = sorted_versions[j]
                next_data = filtered_data[next_version]
                
                # Perform Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(
                    current_data, next_data, 
                    alternative='two-sided'
                )
                
                # Calculate effect size
                effect_size = abs(cohen_d(current_data, next_data))
                
                # If not significantly different (p > alpha) and effect size below threshold
                if p_value > alpha and effect_size < effect_size_threshold:
                    current_group_versions.append(next_version)
                    j += 1
                else:
                    break
            
            # Assign group and rank to all versions in current group
            for version in current_group_versions:
                groups[version] = current_group
                rankings[version] = current_group
            
            # Move to next unprocessed version
            i = j
            current_group += 1
        
        # Create p-value matrix based on group membership
        p_values = {}
        for v1 in version_names:
            p_values[v1] = {}
            for v2 in version_names:
                if v1 == v2:
                    p_values[v1][v2] = 1.0  # Same version
                elif groups[v1] == groups[v2]:
                    p_values[v1][v2] = 1.0  # Same group = not significantly different
                else:
                    p_values[v1][v2] = 0.0  # Different groups = significantly different
        
        result = {
            "success": True,
            "method": f"Mann-Whitney U test with Cohen's d effect size (threshold: {effect_size_threshold})",
            "groups": groups,
            "rankings": rankings,
            "p_values": p_values,
            "alpha": alpha,
            "effect_size_threshold": effect_size_threshold,
            "version_names": version_names,
            "num_groups": len(set(groups.values())),
            "total_versions": len(version_names)
        }
        
        # Log a single consolidated message with key results
        logger.debug(f"Statistical analysis completed: {len(version_names)} versions grouped into {len(set(groups.values()))} groups")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Statistical analysis failed: {str(e)}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}


def format_statistical_results(stats_results: Dict[str, Any]) -> str:
    """Format statistical analysis results for display"""
    if not stats_results or not stats_results.get("success", False):
        return "❌ Statistical analysis failed or no results available"
    
    # Get rankings and sort versions by rank
    rankings = stats_results.get("rankings", {})
    if not rankings:
        return "No ranking information available"
    
    sorted_versions = sorted(rankings.items(), key=lambda x: (x[1], x[0]))
    
    # Format results
    lines = []
    lines.append(f"📊 Statistical Analysis Results:")
    lines.append(f"Method: {stats_results.get('method', 'Unknown')}")
    lines.append(f"Significance level (α): {stats_results.get('alpha', 0.05)}")
    lines.append(f"Effect Size Threshold: {stats_results.get('effect_size_threshold', 0.2)}")
    lines.append("")
    
    # Group versions by rank
    current_rank = None
    rank_group = []
    
    for version, rank in sorted_versions:
        if current_rank != rank:
            if rank_group:
                lines.append(f"Rank {current_rank}: {', '.join(rank_group)}")
            current_rank = rank
            rank_group = [version]
        else:
            rank_group.append(version)
    
    if rank_group:
        lines.append(f"Rank {current_rank}: {', '.join(rank_group)}")
    
    return "\n".join(lines)


def display_existing_solutions_analysis(results: dict):
    """Display analysis of existing solutions"""
    solutions = results.get("solutions", [])
    
    if not solutions:
        st.info("No solutions data to analyze.")
        return
    
    st.markdown("### 📊 Solutions Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Solutions", len(solutions))
    
    with col2:
        solutions_with_results = sum(1 for s in solutions if s.get("has_results"))
        st.metric("With Results", solutions_with_results)
    
    with col3:
        avg_metrics = sum(s.get("results_summary", {}).get("total_metrics", 0) for s in solutions)
        avg_metrics = avg_metrics / len(solutions) if solutions else 0
        st.metric("Avg Metrics", f"{avg_metrics:.1f}")
    
    with col4:
        total_measurements = sum(
            s.get("detailed_metrics", {}).get("total_measurements", 0) 
            for s in solutions
        )
        st.metric("Total Measurements", total_measurements)
    
    # Status distribution
    status_counts = {}
    for solution in solutions:
        status = solution.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    
    if status_counts:
        st.markdown("#### 📈 Status Distribution")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Solution Status Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=list(status_counts.keys()),
                y=list(status_counts.values()),
                title="Solutions by Status",
                labels={"x": "Status", "y": "Count"}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Performance metrics analysis
    solutions_with_metrics = [s for s in solutions if s.get("has_results") and s.get("results_summary")]
    
    if solutions_with_metrics:
        st.markdown("#### ⚡ Performance Analysis")
        
        # Runtime metrics
        runtime_data = []
        for solution in solutions_with_metrics:
            runtime_metrics = solution.get("results_summary", {}).get("runtime_metrics", {})
            for metric_name, metric_data in runtime_metrics.items():
                runtime_data.append({
                    "Solution ID": solution.get("solution_id", "N/A")[:8],
                    "Metric": metric_name,
                    "Average (s)": metric_data.get("avg", 0),
                    "Min (s)": metric_data.get("min", 0),
                    "Max (s)": metric_data.get("max", 0),
                    "Std Dev": metric_data.get("std", 0)
                })
        
        if runtime_data:
            with st.expander("🏃‍♂️ Runtime Metrics", expanded=True):
                runtime_df = pd.DataFrame(runtime_data)
                st.dataframe(runtime_df, use_container_width=True)
                
                # Runtime visualization
                if len(runtime_df) > 1:
                    fig_runtime = px.bar(
                        runtime_df,
                        x="Solution ID",
                        y="Average (s)",
                        color="Metric",
                        title="Average Runtime by Solution",
                        barmode="group"
                    )
                    st.plotly_chart(fig_runtime, use_container_width=True)
        
        # Memory metrics
        memory_data = []
        for solution in solutions_with_metrics:
            memory_metrics = solution.get("results_summary", {}).get("memory_metrics", {})
            for metric_name, metric_data in memory_metrics.items():
                memory_data.append({
                    "Solution ID": solution.get("solution_id", "N/A")[:8],
                    "Metric": metric_name,
                    "Average (bytes)": metric_data.get("avg", 0),
                    "Min (bytes)": metric_data.get("min", 0),
                    "Max (bytes)": metric_data.get("max", 0),
                    "Std Dev": metric_data.get("std", 0)
                })
        
        if memory_data:
            with st.expander("💾 Memory Metrics", expanded=False):
                memory_df = pd.DataFrame(memory_data)
                st.dataframe(memory_df, use_container_width=True)
                
                # Memory visualization
                if len(memory_df) > 1:
                    fig_memory = px.bar(
                        memory_df,
                        x="Solution ID",
                        y="Average (bytes)",
                        color="Metric",
                        title="Average Memory Usage by Solution",
                        barmode="group"
                    )
                    st.plotly_chart(fig_memory, use_container_width=True)
    
    # Detailed solution table
    st.markdown("#### 📋 Detailed Solutions")
    
    table_data = []
    for solution in solutions:
        metrics_summary = solution.get("detailed_metrics", {})
        table_data.append({
            "Solution ID": solution.get("solution_id", "N/A"),
            "Optimization": solution.get("optimization_name", "N/A"),
            "Status": solution.get("status", "Unknown"),
            "Has Results": "✅" if solution.get("has_results") else "❌",
            "Runtime Metrics": metrics_summary.get("runtime_count", 0),
            "Memory Metrics": metrics_summary.get("memory_count", 0),
            "Total Measurements": metrics_summary.get("total_measurements", 0),
            "Created": solution.get("created_at", "Unknown")
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)


def display_single_solution_analysis(analysis_data: dict):
    """Display detailed analysis for a single solution"""
    solution = analysis_data.get("solution", {})
    
    if not solution:
        st.error("No solution data provided for analysis.")
        return
    
    st.markdown(f"### 🔍 Solution Analysis: {solution.get('solution_id', 'N/A')}")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Status", solution.get("status", "Unknown"))
    
    with col2:
        st.metric("Has Results", "✅ Yes" if solution.get("has_results") else "❌ No")
    
    with col3:
        created_at = solution.get("created_at", "Unknown")
        if created_at != "Unknown":
            try:
                # Format date if it's a datetime string
                from datetime import datetime
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_at = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        st.metric("Created", created_at)
    
    # Solution details
    with st.expander("📋 Solution Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information:**")
            st.markdown(f"- **Solution ID:** {solution.get('solution_id', 'N/A')}")
            st.markdown(f"- **Optimization ID:** {solution.get('optimization_id', 'N/A')}")
            st.markdown(f"- **Optimization Name:** {solution.get('optimization_name', 'N/A')}")
        
        with col2:
            specs = solution.get("specs", [])
            st.markdown(f"**Specifications ({len(specs)}):**")
            for spec in specs[:5]:  # Show first 5
                st.markdown(f"- Spec: {spec.get('spec_id', 'N/A')}")
            if len(specs) > 5:
                st.markdown(f"- ... and {len(specs) - 5} more")
    
    # Performance metrics
    if solution.get("has_results") and solution.get("results_summary"):
        results_summary = solution["results_summary"]
        
        st.markdown("#### ⚡ Performance Metrics")
        
        # Runtime metrics
        runtime_metrics = results_summary.get("runtime_metrics", {})
        if runtime_metrics:
            st.markdown("##### 🏃‍♂️ Runtime Performance")
            
            runtime_data = []
            for metric_name, metric_data in runtime_metrics.items():
                runtime_data.append({
                    "Metric": metric_name,
                    "Average": f"{metric_data.get('avg', 0):.4f}s",
                    "Min": f"{metric_data.get('min', 0):.4f}s",
                    "Max": f"{metric_data.get('max', 0):.4f}s",
                    "Std Dev": f"{metric_data.get('std', 0):.4f}s",
                    "Measurements": metric_data.get('count', 0)
                })
            
            if runtime_data:
                runtime_df = pd.DataFrame(runtime_data)
                st.dataframe(runtime_df, use_container_width=True)
                
                # Runtime visualization
                if len(runtime_metrics) > 0:
                    values = [metric_data.get('avg', 0) for metric_data in runtime_metrics.values()]
                    labels = list(runtime_metrics.keys())
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=labels,
                        y=values,
                        name="Average Runtime",
                        marker_color=generate_colors(len(labels))
                    ))
                    
                    fig.update_layout(
                        title="Runtime Metrics Comparison",
                        xaxis_title="Metric",
                        yaxis_title="Time (seconds)",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Memory metrics
        memory_metrics = results_summary.get("memory_metrics", {})
        if memory_metrics:
            with st.expander("💾 Memory Performance", expanded=False):
                memory_data = []
                for metric_name, metric_data in memory_metrics.items():
                    memory_data.append({
                        "Metric": metric_name,
                        "Average": f"{metric_data.get('avg', 0):,.0f} bytes",
                        "Min": f"{metric_data.get('min', 0):,.0f} bytes",
                        "Max": f"{metric_data.get('max', 0):,.0f} bytes",
                        "Std Dev": f"{metric_data.get('std', 0):,.0f} bytes",
                        "Measurements": metric_data.get('count', 0)
                    })
                
                if memory_data:
                    memory_df = pd.DataFrame(memory_data)
                    st.dataframe(memory_df, use_container_width=True)
        
        # CPU metrics
        cpu_metrics = results_summary.get("cpu_metrics", {})
        if cpu_metrics:
            with st.expander("🖥️ CPU Performance", expanded=False):
                cpu_data = []
                for metric_name, metric_data in cpu_metrics.items():
                    cpu_data.append({
                        "Metric": metric_name,
                        "Average": f"{metric_data.get('avg', 0):.4f}",
                        "Min": f"{metric_data.get('min', 0):.4f}",
                        "Max": f"{metric_data.get('max', 0):.4f}",
                        "Std Dev": f"{metric_data.get('std', 0):.4f}",
                        "Measurements": metric_data.get('count', 0)
                    })
                
                if cpu_data:
                    cpu_df = pd.DataFrame(cpu_data)
                    st.dataframe(cpu_df, use_container_width=True)
    
    else:
        st.info("No performance metrics available for this solution.")


def create_performance_comparison_chart(solutions: List[Dict[str, Any]], metric_type: str = "runtime") -> Optional[go.Figure]:
    """Create a performance comparison chart for multiple solutions"""
    
    solutions_with_metrics = [
        s for s in solutions 
        if s.get("has_results") and s.get("results_summary")
    ]
    
    if not solutions_with_metrics:
        return None
    
    data = []
    for solution in solutions_with_metrics:
        solution_id = solution.get("solution_id", "N/A")[:8]
        results_summary = solution.get("results_summary", {})
        
        metrics = results_summary.get(f"{metric_type}_metrics", {})
        
        for metric_name, metric_data in metrics.items():
            data.append({
                "Solution": solution_id,
                "Metric": metric_name,
                "Average": metric_data.get("avg", 0),
                "Min": metric_data.get("min", 0),
                "Max": metric_data.get("max", 0)
            })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x="Solution",
        y="Average",
        color="Metric",
        title=f"{metric_type.title()} Performance Comparison",
        barmode="group",
        hover_data=["Min", "Max"]
    )
    
    unit = "seconds" if metric_type == "runtime" else "bytes" if metric_type == "memory" else "units"
    fig.update_layout(
        xaxis_title="Solution ID",
        yaxis_title=f"Average ({unit})",
        showlegend=True
    )
    
    return fig


def display_execution_results(results: Dict[str, Any]):
    """Display execution results with comprehensive analysis"""
    execution_results = results.get("execution_results", [])
    
    if not execution_results:
        st.info("No execution results to display.")
        return
    
    st.markdown("### 🚀 Execution Results Analysis")
    
    # Summary metrics
    successful = sum(1 for r in execution_results if r.get("success", False))
    total = len(execution_results)
    success_rate = (successful / total) * 100 if total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Executions", total)
    with col2:
        st.metric("Successful", successful, delta=f"{success_rate:.1f}%")
    with col3:
        failed = total - successful
        st.metric("Failed", failed, delta=f"{100-success_rate:.1f}%")
    
    # Results breakdown
    if execution_results:
        st.markdown("#### 📊 Execution Breakdown")
        
        results_data = []
        for result in execution_results:
            rec_info = result.get("recommendation_info", {})
            solution_result = result.get("solution_result")
            
            row = {
                "Spec Name": rec_info.get("spec_name", "Unknown"),
                "Template": rec_info.get("template_name", "Unknown"),
                "Construct": rec_info.get("construct_id", "Unknown"),
                "Success": "✅" if result.get("success") else "❌",
                "Error": result.get("error", "")
            }
            
            if solution_result:
                row.update({
                    "Solution ID": getattr(solution_result, "solution_id", "N/A"),
                    "Execution Time": f"{getattr(solution_result, 'execution_time', 0):.2f}s"
                })
            
            results_data.append(row)
        
        if results_data:
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)
        
        # Performance analysis for successful executions
        successful_results = [r for r in execution_results if r.get("success") and r.get("solution_result")]
        
        if successful_results:
            st.markdown("#### ⚡ Performance Analysis")
            
            perf_data = []
            for result in successful_results:
                solution_result = result["solution_result"]
                rec_info = result["recommendation_info"]
                
                runtime_metrics = getattr(solution_result, "runtime_metrics", {})
                memory_metrics = getattr(solution_result, "memory_metrics", {})
                
                perf_data.append({
                    "Solution": rec_info.get("spec_name", "Unknown"),
                    "Template": rec_info.get("template_name", "Unknown"),
                    "Execution Time": getattr(solution_result, "execution_time", 0),
                    "Runtime Metrics": len(runtime_metrics),
                    "Memory Metrics": len(memory_metrics)
                })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                
                # Execution time comparison
                if len(perf_df) > 1:
                    fig_exec_time = px.bar(
                        perf_df,
                        x="Solution",
                        y="Execution Time",
                        color="Template",
                        title="Execution Time Comparison"
                    )
                    st.plotly_chart(fig_exec_time, use_container_width=True)
                
                st.dataframe(perf_df, use_container_width=True)


def create_summary_dashboard(all_results: Dict[str, Any]):
    """Create a comprehensive dashboard with all results"""
    st.markdown("## 📊 Meta Artemis Results Dashboard")
    
    # Check what types of results we have
    has_solutions = "solutions" in all_results
    has_execution = "execution_results" in all_results
    
    if not has_solutions and not has_execution:
        st.info("No results data available for dashboard.")
        return
    
    # Dashboard tabs
    if has_solutions and has_execution:
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Solutions", "🚀 Executions"])
        
        with tab1:
            # Combined overview
            st.markdown("### 🎯 Analysis Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if has_solutions:
                    solutions = all_results["solutions"]
                    st.metric("Total Solutions", len(solutions))
                    solutions_with_results = sum(1 for s in solutions if s.get("has_results"))
                    st.metric("Solutions with Results", solutions_with_results)
            
            with col2:
                if has_execution:
                    exec_results = all_results["execution_results"]
                    successful_exec = sum(1 for r in exec_results if r.get("success"))
                    st.metric("Successful Executions", successful_exec)
                    success_rate = (successful_exec / len(exec_results)) * 100 if exec_results else 0
                    st.metric("Execution Success Rate", f"{success_rate:.1f}%")
        
        with tab2:
            if has_solutions:
                display_existing_solutions_analysis(all_results)
        
        with tab3:
            if has_execution:
                display_execution_results(all_results)
    
    elif has_solutions:
        display_existing_solutions_analysis(all_results)
    
    elif has_execution:
        display_execution_results(all_results)


def _display_execution_results(results):
    """Internal function to display execution results"""
    # This is for backward compatibility with the original module structure
    display_execution_results(results)


def create_box_plot_with_points(data: pd.DataFrame, x_col: str, y_col: str, color_col: str, 
                               title: str, color_map: dict = None, height: int = 500) -> go.Figure:
    """Create a box plot with individual data points shown"""
    fig = go.Figure()
    
    # Get unique categories
    categories = data[x_col].unique()
    
    # Default color map if not provided
    if color_map is None:
        colors = px.colors.qualitative.Set3
        color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
    
    # Add box plots and scatter points for each category
    for category in categories:
        category_data = data[data[x_col] == category]
        color = color_map.get(category, '#1f77b4')
        
        # Add box plot
        fig.add_trace(go.Box(
            y=category_data[y_col],
            name=category,
            boxpoints='all',  # Show all points
            jitter=0.3,       # Add some jitter to points
            pointpos=-1.8,    # Position points to the left of box
            marker=dict(
                color=color,
                size=4,
                opacity=0.6
            ),
            line=dict(color=color),
            fillcolor=color,
            opacity=0.7
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        showlegend=False,
        height=height,
        hovermode='closest'
    )
    
    return fig


def create_construct_box_plots_grid(individual_constructs: Dict[str, Dict[str, float]], 
                                   project_name: str) -> Optional[go.Figure]:
    """Create individual box plots for each construct showing the five optimization versions"""
    try:
        if not individual_constructs:
            logger.warning("No individual construct data provided")
            return None
        
        # Get the actual construct data from the analysis results
        # individual_constructs contains averaged data, but we need the raw data for box plots
        # This function should receive the raw construct data instead
        logger.warning("create_construct_box_plots_grid received averaged data instead of raw data for box plots")
        return None
        
    except Exception as e:
        logger.error(f"Error creating construct box plots: {str(e)}")
        return None


def create_individual_construct_box_plots(construct_details: Dict[str, Dict], project_name: str) -> Optional[go.Figure]:
    """Create individual box plots for each construct showing the five optimization versions with statistical analysis"""
    try:
        if not construct_details:
            logger.warning("No construct details provided")
            return None
        
        # Filter constructs that have data
        constructs_with_data = {
            construct_id: details for construct_id, details in construct_details.items()
            if details.get("total_evaluations", 0) > 0
        }
        
        if not constructs_with_data:
            logger.warning("No constructs with evaluation data found")
            return None
        
        # Perform statistical analysis for each construct
        construct_rankings = {}
        for construct_id, details in constructs_with_data.items():
            construct_versions = details.get("versions", {})
            
            # Prepare data for statistical analysis
            versions_data = {}
            for version, data in construct_versions.items():
                if data:  # Only include versions with data
                    versions_data[version] = data
            
            if len(versions_data) >= 2:  # Need at least 2 versions for statistical testing
                try:
                    sk_result = perform_statistical_analysis(versions_data, alpha=0.05)
                    if sk_result.get("success", False):
                        construct_rankings[construct_id] = {
                            "rankings": sk_result.get("rankings", {}),
                            "groups": sk_result.get("groups", {}),
                            "success": True
                        }
                        logger.debug(f"Statistical analysis for Construct {details['rank']}: {sk_result.get('rankings', {})}")
                    else:
                        construct_rankings[construct_id] = {"success": False}
                except Exception as e:
                    logger.warning(f"Statistical analysis failed for Construct {details['rank']}: {str(e)}")
                    construct_rankings[construct_id] = {"success": False}
            else:
                construct_rankings[construct_id] = {"success": False}
        
        # Calculate grid dimensions
        num_constructs = len(constructs_with_data)
        cols = min(3, num_constructs)  # Max 3 columns
        rows = (num_constructs + cols - 1) // cols  # Calculate rows needed
        
        # Create simple subplot titles without best version info
        subplot_titles = []
        for construct_id, details in constructs_with_data.items():
            construct_rank = details['rank']
            subplot_titles.append(f"Construct {construct_rank}")
        
        # Create subplots with reduced vertical spacing
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.12,  # Reduced spacing since no subtitles
            horizontal_spacing=0.08
        )
        
        # Version colors for consistency
        version_colors = {
            "original": "#ff7f7f",     # Light red
            "baseline": "#ffb347",     # Light orange  
            "simplified": "#87ceeb",   # Light blue
            "standard": "#98fb98",     # Light green
            "enhanced": "#90ee90"      # Green
        }
        
        versions = ["original", "baseline", "simplified", "standard", "enhanced"]
        
        # Plot each construct
        for idx, (construct_id, details) in enumerate(constructs_with_data.items()):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            construct_rank = details.get("rank", idx + 1)
            construct_versions = details.get("versions", {})
            
            # Get statistical analysis results for this construct
            sk_rankings = {}
            best_version = None
            if construct_id in construct_rankings and construct_rankings[construct_id].get("success", False):
                sk_rankings = construct_rankings[construct_id]["rankings"]
                # Find the best performing version (rank 1)
                if sk_rankings:
                    best_version = min(sk_rankings.items(), key=lambda x: x[1])[0]
            
            # Add box plots for each version
            for version_idx, version in enumerate(versions):
                version_data = construct_versions.get(version, [])
                
                # Get statistical analysis rank for this version
                sk_rank = sk_rankings.get(version, None)
                
                # Create version label for hover (includes rank info)
                version_label_with_rank = version.title()
                if sk_rank is not None:
                    version_label_with_rank += f" (Rank {sk_rank})"
                
                # Create clean version label for legend (no rank info)
                legend_label = version.title()
                
                if version_data:  # Has data
                    fig.add_trace(go.Box(
                        x=[version] * len(version_data),  # Explicitly set x values
                        y=version_data,
                        name=legend_label,  # Use clean version name
                        boxpoints='all',  # Show all data points
                        jitter=0.3,      # Add jitter to points
                        pointpos=-1.8,   # Position points to the left of the box
                        marker=dict(
                            color=version_colors[version],
                            size=4,
                            opacity=0.7
                        ),
                        line=dict(color=version_colors[version]),
                        fillcolor=version_colors[version],
                        opacity=0.7,
                        showlegend=False,  # No legend needed - x-axis labels are clear
                        hovertemplate=f"<b>Construct {construct_rank}</b><br>Version: {version_label_with_rank}<br>Runtime: %{{y:.3f}}s<extra></extra>"
                    ), row=row, col=col)
                    
                    # Add mean marker
                    mean_value = np.mean(version_data)
                    fig.add_trace(go.Scatter(
                        x=[version],
                        y=[mean_value],
                        mode='markers',
                        marker=dict(
                            symbol='diamond',
                            size=8,
                            color='white',
                            line=dict(color=version_colors[version], width=2)
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>Construct {construct_rank}</b><br>Version: {version_label_with_rank}<br>Mean: {mean_value:.3f}s<extra></extra>",
                        name=f"Mean {legend_label}"
                    ), row=row, col=col)
                else:  # No data - show empty box plot
                    fig.add_trace(go.Box(
                        x=[version],  # Explicitly set x value even for empty box
                        y=[],  # Empty data
                        name=legend_label,  # Use clean version name
                        marker=dict(
                            color=version_colors[version],
                            size=4,
                            opacity=0.3
                        ),
                        line=dict(color=version_colors[version], width=1),
                        fillcolor=version_colors[version],
                        opacity=0.2,
                        showlegend=False,  # No legend needed - x-axis labels are clear
                        hovertemplate=f"<b>Construct {construct_rank}</b><br>Version: {version_label_with_rank}<br>No data<extra></extra>"
                    ), row=row, col=col)
        
        # Update layout
        fig.update_layout(
            height=300 * rows,  # Adjust height based on number of rows
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False  # No legend needed - x-axis labels are clear
        )
        
        # Update all x-axes and y-axes with custom styling for best versions
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if (i - 1) * cols + j <= num_constructs:
                    # Get the construct for this subplot
                    construct_idx = (i - 1) * cols + (j - 1)
                    construct_items = list(constructs_with_data.items())
                    if construct_idx < len(construct_items):
                        construct_id, details = construct_items[construct_idx]
                        
                        # Get the best version for this construct
                        best_version = None
                        if construct_id in construct_rankings and construct_rankings[construct_id].get("success", False):
                            sk_rankings = construct_rankings[construct_id]["rankings"]
                            if sk_rankings:
                                best_version = min(sk_rankings.items(), key=lambda x: x[1])[0]
                        
                        # Create custom tick labels with rank information and bold formatting for best version
                        tick_labels = []
                        tick_vals = []
                        for version in versions:
                            # Get the statistical analysis rank for this version in this construct
                            sk_rank = sk_rankings.get(version, None)
                            
                            # Create label with rank information
                            if sk_rank is not None:
                                version_label = f"{version.title()}\n(Rank {sk_rank})"
                            else:
                                version_label = version.title()
                            
                            # Make the best version bold - use HTML formatting
                            if sk_rank == 1:  # Changed from 'version in best_versions' to directly check rank
                                tick_labels.append(f"<b>{version_label}</b>")  # Bold the best version
                            else:
                                tick_labels.append(version_label)
                            
                            # Use version names to match box plot x-positions
                            tick_vals.append(version)
                        
                        fig.update_xaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                            categoryorder='array',
                            categoryarray=versions,  # Use the original lowercase versions to match box plots
                            ticktext=tick_labels,  # Custom tick labels with rank info and bold formatting
                            tickvals=tick_vals,  # Use version names to match box plot x-positions
                            title_text="Version" if i == rows else "",  # Only show x-axis title on bottom row
                            row=i, col=j
                        )
                        fig.update_yaxes(
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray',
                            title_text="Runtime (s)" if j == 1 else "",  # Only show y-axis title on left column
                            row=i, col=j
                        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating individual construct box plots: {str(e)}")
        return None


def create_version_level_box_plot(version_level_data: Dict[str, Any], project_name: str) -> Optional[go.Figure]:
    """Create box plot for version-level solutions with statistical analysis"""
    try:
        # Prepare data for box plot - ALWAYS include all 5 versions
        version_runtime_data = {
            "original": [],
            "baseline": [],
            "simplified": [],
            "standard": [],
            "enhanced": []
        }
        
        # Collect runtime data from each solution
        if version_level_data and version_level_data.get("solutions"):
            for solution in version_level_data["solutions"]:
                version_type = solution["version_type"]
                runtime_data = solution.get("runtime_data", [])
                if runtime_data and version_type in version_runtime_data:
                    version_runtime_data[version_type].extend(runtime_data)
        
        # Perform statistical analysis on version-level data
        sk_result = perform_statistical_analysis(version_runtime_data, alpha=0.05)
        sk_rankings = sk_result.get("rankings", {}) if sk_result.get("success", False) else {}
        
        # Store rankings in version_level_data for table display
        version_level_data["statistical_rankings"] = sk_rankings
        
        # Find best versions (all with rank 1)
        best_versions = [v for v, r in sk_rankings.items() if r == 1] if sk_rankings else []
        
        # Create the box plot
        fig = go.Figure()
        
        # Use SAME colors as construct-level box plots for consistency
        version_colors = {
            "original": "#ff7f7f",     # Light red
            "baseline": "#ffb347",     # Light orange  
            "simplified": "#87ceeb",   # Light blue
            "standard": "#98fb98",     # Light green
            "enhanced": "#90ee90"      # Green
        }
        
        versions = ["original", "baseline", "simplified", "standard", "enhanced"]
        
        # Add box plots for ALL versions (including empty ones)
        for version in versions:
            version_data = version_runtime_data[version]
            
            # Create version label for hover (includes rank info)
            version_label_with_rank = version.title()
            if version in sk_rankings:
                rank = sk_rankings[version]
                version_label_with_rank += f" (Rank {rank})"
            
            # Create clean version label for legend (no rank info)
            legend_label = version.title()
            
            if version_data:  # Has data
                fig.add_trace(go.Box(
                    x=[version] * len(version_data),  # Explicitly set x values
                    y=version_data,
                    name=legend_label,  # Use clean version name
                    boxpoints='all',  # Show all data points
                    jitter=0.3,      # Add jitter to points
                    pointpos=-1.8,   # Position points to the left of the box
                    marker=dict(
                        color=version_colors[version],
                        size=6,
                        opacity=0.7
                    ),
                    line=dict(color=version_colors[version]),
                    fillcolor=version_colors[version],
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate=f"<b>Version-Level</b><br>Version: {version_label_with_rank}<br>Runtime: %{{y:.3f}}s<extra></extra>"
                ))
                
                # Add mean marker
                mean_value = np.mean(version_data)
                fig.add_trace(go.Scatter(
                    x=[version],
                    y=[mean_value],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color='white',
                        line=dict(color=version_colors[version], width=2)
                    ),
                    showlegend=False,
                    hovertemplate=f"<b>Version-Level</b><br>Version: {version_label_with_rank}<br>Mean: {mean_value:.3f}s<extra></extra>",
                    name=f"Mean {legend_label}"
                ))
            else:  # No data - show empty box plot
                fig.add_trace(go.Box(
                    x=[version],  # Explicitly set x value even for empty box
                    y=[],  # Empty data
                    name=legend_label,  # Use clean version name
                    marker=dict(
                        color=version_colors[version],
                        size=6,
                        opacity=0.3
                    ),
                    line=dict(color=version_colors[version], width=1),
                    fillcolor=version_colors[version],
                    opacity=0.2,
                    showlegend=False,
                    hovertemplate=f"<b>Version-Level</b><br>Version: {version_label_with_rank}<br>No data<extra></extra>"
                ))
        
        # Create custom tick labels with rank information and bold formatting for best versions
        tick_labels = []
        tick_vals = []
        for version in versions:
            # Get the statistical analysis rank for this version
            sk_rank = sk_rankings.get(version, None)
            
            # Create label with rank information
            if sk_rank is not None:
                version_label = f"{version.title()}\n(Rank {sk_rank})"
            else:
                version_label = version.title()
            
            # Make the best versions bold - use HTML formatting
            if sk_rank == 1:  # Changed from 'version in best_versions' to directly check rank
                tick_labels.append(f"<b>{version_label}</b>")  # Bold the best version
            else:
                tick_labels.append(version_label)
            
            # Use version names to match box plot x-positions
            tick_vals.append(version)
        
        # Update layout - consistent with construct-level plots
        fig.update_layout(
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showline=False,
                tickvals=tick_vals,
                ticktext=tick_labels,
                title="Prompt Version"
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showline=False,
                title="Runtime (seconds)"
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating version-level box plot: {str(e)}")
        return None

def create_overall_project_box_plot(valid_projects: Dict[str, Dict[str, Any]]) -> Optional[go.Figure]:
    """Create overall project box plot comparing 5 versions across all constructs"""
    try:
        # Get project name (we know there's only one project)
        project_name = next(iter(valid_projects.values()))["project_name"]
        
        # Prepare data for box plot - ALWAYS include all 5 versions
        version_runtime_data = {
            "original": [],
            "baseline": [],
            "simplified": [],
            "standard": [],
            "enhanced": []
        }
        
        # Collect ALL runtime data from ALL constructs
        for project_id, result in valid_projects.items():
            # Get construct-level data
            construct_details = result.get("construct_details", {})
            for construct_id, construct_data in construct_details.items():
                # Get runtime data for each version in this construct
                construct_versions = construct_data.get("versions", {})  # Changed from versions_data to versions
                for version in version_runtime_data.keys():
                    version_data = construct_versions.get(version, [])  # Access data directly from versions dict
                    if version_data:
                        version_runtime_data[version].extend(version_data)
        
        # Check if we have any data to plot
        total_data_points = sum(len(data) for data in version_runtime_data.values())
        logger.info(f"Overall box plot: {total_data_points} total data points across all versions")
        for version, data in version_runtime_data.items():
            logger.info(f"  - {version}: {len(data)} data points")
        
        if total_data_points == 0:
            logger.warning("No data available for overall project box plot")
            return None
            
        # Perform statistical analysis on version-level data
        sk_result = perform_statistical_analysis(version_runtime_data, alpha=0.05)
        sk_rankings = sk_result.get("rankings", {}) if sk_result.get("success", False) else {}
        
        # Find best versions (all with rank 1)
        best_versions = [v for v, r in sk_rankings.items() if r == 1] if sk_rankings else []
        
        # Create the box plot
        fig = go.Figure()
        
        # Use SAME colors as version-level box plots for consistency
        version_colors = {
            "original": "#ff7f7f",     # Light red
            "baseline": "#ffb347",     # Light orange  
            "simplified": "#87ceeb",   # Light blue
            "standard": "#98fb98",     # Light green
            "enhanced": "#90ee90"      # Green
        }
        
        versions = ["original", "baseline", "simplified", "standard", "enhanced"]
        
        # Add box plots for ALL versions (including empty ones)
        for version in versions:
            version_data = version_runtime_data[version]
            
            # Create version label for hover (includes rank info)
            version_label_with_rank = version.title()
            if version in sk_rankings:
                rank = sk_rankings[version]
                version_label_with_rank += f" (Rank {rank})"
            
            # Create clean version label for legend (no rank info)
            legend_label = version.title()
            
            if version_data:  # Has data
                fig.add_trace(go.Box(
                    x=[version] * len(version_data),  # Explicitly set x values
                    y=version_data,
                    name=legend_label,  # Use clean version name
                    boxpoints='all',  # Show all data points
                    jitter=0.3,      # Add jitter to points
                    pointpos=-1.8,   # Position points to the left of the box
                    marker=dict(
                        color=version_colors[version],
                        size=6,
                        opacity=0.7
                    ),
                    line=dict(color=version_colors[version]),
                    fillcolor=version_colors[version],
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate=f"<b>Overall</b><br>Version: {version_label_with_rank}<br>Runtime: %{{y:.3f}}s<extra></extra>"
                ))
                
                # Add mean marker
                mean_value = np.mean(version_data)
                fig.add_trace(go.Scatter(
                    x=[version],
                    y=[mean_value],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color='white',
                        line=dict(color=version_colors[version], width=2)
                    ),
                    showlegend=False,
                    hovertemplate=f"<b>Overall</b><br>Version: {version_label_with_rank}<br>Mean: {mean_value:.3f}s<extra></extra>",
                    name=f"Mean {legend_label}"
                ))
            else:  # No data - show empty box plot
                fig.add_trace(go.Box(
                    x=[version],  # Explicitly set x value even for empty box
                    y=[],  # Empty data
                    name=legend_label,  # Use clean version name
                    marker=dict(
                        color=version_colors[version],
                        size=6,
                        opacity=0.3
                    ),
                    line=dict(color=version_colors[version], width=1),
                    fillcolor=version_colors[version],
                    opacity=0.2,
                    showlegend=False,
                    hovertemplate=f"<b>Overall</b><br>Version: {version_label_with_rank}<br>No data<extra></extra>"
                ))
        
        # Create custom tick labels with rank information and bold formatting for best versions
        tick_labels = []
        tick_vals = []
        for version in versions:
            # Get the statistical analysis rank for this version
            sk_rank = sk_rankings.get(version, None)
            
            # Create label with rank information
            if sk_rank is not None:
                version_label = f"{version.title()}\n(Rank {sk_rank})"
            else:
                version_label = version.title()
            
            # Make the best versions bold
            if version in best_versions:
                tick_labels.append(f"<b>{version_label}</b>")  # Bold the best version
            else:
                tick_labels.append(version_label)
            
            # Use version names to match box plot x-positions
            tick_vals.append(version)
        
        # Update layout - consistent with version-level plots
        fig.update_layout(
            title=f"Overall Runtime Performance Comparison - {project_name}",  # Updated title
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showline=False,
                tickvals=tick_vals,
                ticktext=tick_labels,
                title="Prompt Version"
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showline=False,
                title="Runtime (seconds)"
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating overall project box plot: {str(e)}")
        return None


def display_box_plot_analysis_results(analysis_results: Dict[str, Any]):
    """Display comprehensive box plot analysis results with warnings for missing data"""
    if not analysis_results:
        st.warning("⚠️ No analysis results to display")
        return
    
    st.markdown("### 📊 Runtime Impact Analysis Results")
    
    # Overall summary
    project_results = analysis_results.get("project_results", {})
    summary = analysis_results.get("summary", {})
    
    if not project_results:
        st.error("❌ No project results found")
        return
    
    # Display overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", len(project_results))
    with col2:
        total_constructs = sum(result.get("num_constructs", 0) for result in project_results.values())
        st.metric("Total Constructs", total_constructs)
    with col3:
        constructs_processed = sum(result.get("constructs_processed", 0) for result in project_results.values())
        st.metric("Constructs Processed", constructs_processed)
    with col4:
        avg_improvement = summary.get("average_improvement_percentage", 0)
        if not np.isnan(avg_improvement):
            st.metric("Avg Improvement", f"{avg_improvement:.1f}%")
        else:
            st.metric("Avg Improvement", "N/A")
    
    # Filter out projects with errors for visualization
    valid_projects = {pid: result for pid, result in project_results.items() 
                     if "error" not in result and result.get("constructs_processed", 0) > 0}
    
    if not valid_projects:
        st.error("❌ No valid project data available for visualization")
        return
    
    # Overall project comparison box plots
    st.markdown("### 📈 Overall Project Performance Comparison")
    
    try:
        overall_box_plot = create_overall_project_box_plot(valid_projects)
        if overall_box_plot:
            st.plotly_chart(overall_box_plot, use_container_width=True)
        else:
            st.warning("⚠️ Unable to create overall project box plot - insufficient data")
    except Exception as e:
        st.error(f"❌ Error creating overall project box plot: {str(e)}")
    

    for project_id, result in valid_projects.items():
        project_name = result["project_name"]
        
        # Display content directly without expander since we're analyzing one project at a time
        st.markdown(f"#### 📊 {project_name} - Construct-Level Box Plots")
        
        # Individual construct box plots - this is the main focus
        construct_details = result.get("construct_details", {})
        
        if construct_details and any(
            details.get("total_evaluations", 0) > 0 for details in construct_details.values()
        ):

            try:
                construct_plots = create_individual_construct_box_plots(construct_details, project_name)
                if construct_plots:
                    st.plotly_chart(construct_plots, use_container_width=True)
                else:
                    st.warning("⚠️ Unable to create construct-level box plots - insufficient data")
            except Exception as e:
                st.error(f"❌ Error creating construct box plots: {str(e)}")
            
            # Combined construct analysis table with rankings and detailed statistics
            st.markdown("#### 📊 Construct-Level Evaluation Details")
            st.markdown("*Rankings, statistics, and evaluation details for each construct (🥇 = Best Performing)*")
            
            # Create comprehensive combined table
            combined_data = []
            available_versions = set()
            
            for construct_id, details in construct_details.items():
                if details.get("total_evaluations", 0) > 0:
                    construct_rank = details.get("rank", "Unknown")
                    construct_versions = details.get("versions", {})
                    total_evals = details.get("total_evaluations", 0)
                    missing_versions = details.get("missing_versions", [])
                    
                    # Prepare data for statistical analysis
                    versions_data = {}
                    for version, data in construct_versions.items():
                        if data:  # Only include versions with data
                            versions_data[version] = data
                            available_versions.add(version)
                    
                    # Initialize row data for this construct
                    row_data = {
                        "Construct": f"Construct {construct_rank}",
                        "Construct ID": construct_id[:8] + "...",
                        "Total Evals": total_evals,
                        "Missing": 50 - total_evals
                    }
                    
                    # Perform statistical analysis for rankings
                    sk_rankings = {}
                    best_versions = []
                    if len(versions_data) >= 2:  # Need at least 2 versions for statistical testing
                        try:
                            sk_result = perform_statistical_analysis(versions_data, alpha=0.05)
                            if sk_result.get("success", False):
                                sk_rankings = sk_result.get("rankings", {})
                                # Find all versions with rank 1
                                if sk_rankings:
                                    best_versions = [v for v, r in sk_rankings.items() if r == 1]
                        except Exception as e:
                            logger.warning(f"Statistical analysis failed for Construct {construct_rank}: {str(e)}")
                    
                    # Calculate performance stats for each version
                    version_counts = {}
                    version_stats = {}
                    for version, data in construct_versions.items():
                        version_counts[version] = len(data) if data else 0
                        if data:
                            version_stats[version] = {
                                "avg": np.mean(data),
                                "min": np.min(data),
                                "max": np.max(data)
                            }
                            available_versions.add(version)
                        else:
                            version_stats[version] = {"avg": np.nan, "min": np.nan, "max": np.nan}
                    
                    # Format performance data for display
                    def format_perf(stats, metric):
                        val = stats.get(metric, np.nan)
                        return f"{val:.3f}s" if not np.isnan(val) else "N/A"
                    
                    # Add data for each available version
                    for version in available_versions:
                        # Ranking info
                        if version in sk_rankings:
                            rank = sk_rankings[version]
                            if version in best_versions:
                                rank_display = f"🥇 {rank}"
                            else:
                                rank_display = str(rank)
                        else:
                            rank_display = "N/A"
                        
                        # Performance stats
                        count = version_counts.get(version, 0)
                        stats = version_stats.get(version, {})
                        
                        # Add columns for this version
                        row_data[f"{version.title()} Rank"] = rank_display
                        row_data[f"{version.title()} Count"] = count
                        row_data[f"{version.title()} Avg"] = format_perf(stats, "avg")
                        row_data[f"{version.title()} Min"] = format_perf(stats, "min")
                        row_data[f"{version.title()} Max"] = format_perf(stats, "max")
                    
                    combined_data.append(row_data)
            
            if combined_data and available_versions:
                # Sort by construct rank
                combined_data.sort(key=lambda x: (
                    int(x["Construct"].split()[1]) if x["Construct"].split()[1].isdigit() else float('inf')
                ))
                
                # Create DataFrame with proper column order
                basic_columns = ["Construct", "Construct ID", "Total Evals", "Missing"]
                version_columns = []
                
                # Add columns for each version in a logical order
                for version in sorted(available_versions):
                    version_title = version.title()
                    version_columns.extend([
                        f"{version_title} Rank",
                        f"{version_title} Count", 
                        f"{version_title} Avg",
                        f"{version_title} Min",
                        f"{version_title} Max"
                    ])
                
                column_order = basic_columns + version_columns
                combined_df = pd.DataFrame(combined_data)
                
                # Reorder columns to ensure consistent display
                combined_df = combined_df.reindex(columns=column_order, fill_value="N/A")
                
                st.dataframe(combined_df, use_container_width=True)
            else:
                st.warning("⚠️ No construct performance data available")
        else:
            st.warning("⚠️ No individual construct data available for visualization")
        
        # Version-level evaluation details table
        version_level_data = result.get("version_level_data", {})
        if version_level_data and version_level_data.get("solutions"):
            st.markdown("#### 📊 Version-Level Performance Analysis")
            
            # Prepare data for statistical analysis
            version_runtime_data = {
                "original": [],
                "baseline": [],
                "simplified": [],
                "standard": [],
                "enhanced": []
            }
            
            # Collect runtime data from each solution
            for solution in version_level_data["solutions"]:
                version_type = solution["version_type"]
                runtime_data = solution.get("runtime_data", [])
                if runtime_data and version_type in version_runtime_data:
                    version_runtime_data[version_type].extend(runtime_data)
            
            # Perform statistical analysis on version-level data
            sk_result = perform_statistical_analysis(version_runtime_data, alpha=0.05)
            sk_rankings = sk_result.get("rankings", {}) if sk_result.get("success", False) else {}
            
            # Store rankings in version_level_data for table display
            version_level_data["statistical_rankings"] = sk_rankings
            
            # Version-level box plot
            try:
                version_box_plot = create_version_level_box_plot(version_level_data, project_name)
                if version_box_plot:
                    st.plotly_chart(version_box_plot, use_container_width=True)
                else:
                    st.warning("⚠️ Unable to create version-level box plot - insufficient data")
            except Exception as e:
                st.error(f"❌ Error creating version-level box plot: {str(e)}")

            # Version-level summary statistics (show all 5 versions)
            version_stats = version_level_data.get("version_stats", {})
            
            if version_stats:
                st.markdown("##### 📊 Version-Level Evaluation Details")
                st.markdown("*Summary across all version-level solutions (each solution evaluated ~10 times)*")
                
                # Ensure all versions are included
                all_versions = ["original", "baseline", "simplified", "standard", "enhanced"]
                stats_data = []
                
                for version in all_versions:
                    stats = version_stats.get(version, {
                        "solution_count": 0,
                        "total_measurements": 0,
                        "mean_runtime": np.nan,
                        "median_runtime": np.nan,
                        "std_runtime": np.nan,
                        "min_runtime": np.nan,
                        "max_runtime": np.nan
                    })
                    
                    # Add ranking information as a separate column
                    rank_display = str(sk_rankings.get(version, "N/A"))
                    if version in sk_rankings and sk_rankings[version] == 1:
                        rank_display = f"🥇 {sk_rankings[version]}"
                    
                    stats_data.append({
                        "Version": version.title(),
                        "Rank": rank_display,
                        "Solutions": stats["solution_count"],
                        "Total Measurements": stats["total_measurements"],
                        "Mean Runtime (s)": f"{stats['mean_runtime']:.3f}" if not np.isnan(stats['mean_runtime']) else "N/A",
                        "Median Runtime (s)": f"{stats['median_runtime']:.3f}" if not np.isnan(stats['median_runtime']) else "N/A",
                        "Std Dev (s)": f"{stats['std_runtime']:.3f}" if not np.isnan(stats['std_runtime']) else "N/A",
                        "Min Runtime (s)": f"{stats['min_runtime']:.3f}" if not np.isnan(stats['min_runtime']) else "N/A",
                        "Max Runtime (s)": f"{stats['max_runtime']:.3f}" if not np.isnan(stats['max_runtime']) else "N/A"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
            else:
                st.warning("⚠️ No version-level solution data available")
        else:
            st.info("ℹ️ No version-level solutions found for this project")
    
    # Export functionality removed - now handled in batch analysis configuration 