"""
Performance Improvement Analysis Module

This module provides analysis of performance improvements relative to the original version,
using the same patterns as the current runtime analysis but computing percentage improvements.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import statistics
from scipy import stats

# Import statistical testing from visualization module
from .visualization import perform_statistical_analysis


def compute_performance_improvements(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute performance improvements relative to the original version for each project.
    
    Args:
        analysis_results: Raw analysis results from the current runtime analysis
        
    Returns:
        Dictionary with performance improvement data structured for visualization
    """
    logger.info("🔄 Computing performance improvements relative to original version")
    
    project_results = analysis_results.get("project_results", {})
    improvement_results = {
        "project_results": {},
        "summary": {},
        "statistical_tests": {},
        "performance_improvements": {}
    }
    
    valid_projects = 0
    all_improvements = []
    
    for project_id, project_data in project_results.items():
        project_name = project_data.get("project_name", "Unknown")
        logger.info(f"📊 Computing improvements for project: {project_name}")
        
        # Get version-level data
        versions_data = project_data.get("versions_data", {})
        if not versions_data:
            logger.warning(f"⚠️ No version data found for project {project_name}")
            continue
            
        # Find original version data
        original_data = None
        for version_type, version_measurements in versions_data.items():
            if version_type.lower() == "original":
                original_data = version_measurements if isinstance(version_measurements, list) else []
                break
        
        if not original_data or len(original_data) == 0:
            logger.warning(f"⚠️ No original version data found for project {project_name}")
            continue
            
        original_mean = np.mean(original_data)
        if original_mean <= 0:
            logger.warning(f"⚠️ Invalid original mean ({original_mean}) for project {project_name}")
            continue
            
        # Compute improvements for each version
        project_improvements = {}
        construct_improvements = {}
        
        # Process version-level improvements
        for version_type, version_measurements in versions_data.items():
            if version_type.lower() == "original":
                continue  # Skip original as it's the baseline
                
            if not isinstance(version_measurements, list) or not version_measurements:
                continue
                
            # Compute percentage improvements for each measurement
            improvements = []
            for measurement in version_measurements:
                if measurement > 0:
                    improvement = ((original_mean - measurement) / original_mean) * 100
                    improvements.append(improvement)
            
            if improvements:
                project_improvements[version_type] = {
                    "improvements": improvements,
                    "mean_improvement": np.mean(improvements),
                    "std_improvement": np.std(improvements),
                    "count": len(improvements),
                    "median_improvement": np.median(improvements)
                }
                all_improvements.extend(improvements)
        
        # Process construct-level improvements
        construct_details = project_data.get("construct_details", {})
        if construct_details:
            logger.info(f"Processing construct-level improvements for {len(construct_details)} constructs")
            
            for construct_id, construct_data in construct_details.items():
                construct_versions = construct_data.get("versions", {})
                
                # Find original version for this construct
                construct_original_data = construct_versions.get("original", [])
                if not construct_original_data:
                    continue
                    
                construct_original_mean = np.mean(construct_original_data)
                if construct_original_mean <= 0:
                    continue
                    
                construct_improvements[construct_id] = {
                    "original_mean": construct_original_mean,
                    "versions": {}
                }
                
                # Compute improvements for each version of this construct
                for version_type, version_measurements in construct_versions.items():
                    if version_type.lower() == "original" or not version_measurements:
                        continue
                        
                    improvements = []
                    for measurement in version_measurements:
                        if measurement > 0:
                            improvement = ((construct_original_mean - measurement) / construct_original_mean) * 100
                            improvements.append(improvement)
                    
                    if improvements:
                        construct_improvements[construct_id]["versions"][version_type] = {
                            "improvements": improvements,
                            "mean_improvement": np.mean(improvements),
                            "std_improvement": np.std(improvements),
                            "count": len(improvements)
                        }
        
        # Process version-level improvements
        version_level_data = project_data.get("version_level_data", {})
        version_level_improvements = {}
        
        if version_level_data and version_level_data.get("solutions"):
            logger.info(f"Processing version-level improvements for {len(version_level_data['solutions'])} solutions")
            
            # Dynamically discover all available versions from the data
            all_versions = set()
            for solution in version_level_data["solutions"]:
                all_versions.add(solution["version_type"])
            
            # Initialize runtime data dictionary with discovered versions
            version_runtime_data = {version: [] for version in all_versions}
            
            for solution in version_level_data["solutions"]:
                version_type = solution["version_type"]
                runtime_data = solution.get("runtime_data", [])
                if runtime_data and version_type in version_runtime_data:
                    version_runtime_data[version_type].extend(runtime_data)
            
            # Find original version data for version-level
            original_version_data = version_runtime_data.get("original", [])
            if original_version_data:
                original_version_mean = np.mean(original_version_data)
                
                if original_version_mean > 0:
                    # Compute improvements for each version type
                    for version_type, version_measurements in version_runtime_data.items():
                        if version_type.lower() == "original" or not version_measurements:
                            continue
                            
                        improvements = []
                        for measurement in version_measurements:
                            if measurement > 0:
                                improvement = ((original_version_mean - measurement) / original_version_mean) * 100
                                improvements.append(improvement)
                        
                        if improvements:
                            version_level_improvements[version_type] = {
                                "improvements": improvements,
                                "mean_improvement": np.mean(improvements),
                                "std_improvement": np.std(improvements),
                                "count": len(improvements)
                            }
                            all_improvements.extend(improvements)
        
        # Store project results
        improvement_results["project_results"][project_id] = {
            "project_name": project_name,
            "version_improvements": project_improvements,
            "construct_improvements": construct_improvements,
            "version_level_improvements": version_level_improvements,
            "original_mean": original_mean,
            "num_constructs": len(construct_improvements),
            "constructs_processed": len([c for c in construct_improvements.values() if c.get("versions")])
        }
        
        valid_projects += 1
    
    # Compute overall summary
    if all_improvements:
        improvement_results["summary"] = {
            "total_projects": valid_projects,
            "total_improvements": len(all_improvements),
            "mean_improvement": np.mean(all_improvements),
            "std_improvement": np.std(all_improvements),
            "median_improvement": np.median(all_improvements),
            "min_improvement": np.min(all_improvements),
            "max_improvement": np.max(all_improvements)
        }
    
    logger.info(f"✅ Computed improvements for {valid_projects} projects with {len(all_improvements)} total measurements")
    return improvement_results


def perform_improvement_statistical_analysis(improvement_data: Dict[str, List[float]], alpha: float = 0.05, effect_size_threshold: float = 0.2) -> Dict[str, Any]:
    """
    Perform statistical analysis on performance improvement data.
    
    Args:
        improvement_data: Dictionary with version names as keys and lists of improvement percentages as values
        alpha: Significance level for statistical tests
        effect_size_threshold: Threshold for considering effect size non-negligible
        
    Returns:
        Dictionary containing statistical test results and rankings
    """
    # Filter out versions with no data and exclude original (it's the baseline)
    filtered_data = {k: v for k, v in improvement_data.items() 
                    if k.lower() != "original" and v and len(v) > 0}
    
    if len(filtered_data) < 2:
        logger.warning("⚠️ Need at least 2 versions with improvement data for statistical testing")
        return {
            "success": False,
            "error": "Insufficient data for statistical testing",
            "groups": {},
            "rankings": {},
            "p_values": {}
        }
    
    try:
        from scipy import stats
        
        version_names = list(filtered_data.keys())
        
        # Calculate means and variances
        means = {version: np.mean(data) for version, data in filtered_data.items()}
        
        # Sort versions by mean performance (higher is better for improvements)
        sorted_versions = sorted(means.keys(), key=lambda x: means[x], reverse=True)
        
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
        
        return result
        
    except Exception as e:
        logger.error(f"Statistical analysis failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "groups": {},
            "rankings": {},
            "p_values": {}
        }


def create_overall_improvement_summary_table(improvement_results: Dict[str, Any]) -> pd.DataFrame:
    """Create overall summary table of performance improvements across all projects"""
    
    project_results = improvement_results.get("project_results", {})
    summary_data = []
    
    # Collect all version improvement data across projects
    all_version_improvements = {
        "baseline": [],
        "simplified": [],
        "standard": [],
        "enhanced": []
    }
    
    for project_id, project_data in project_results.items():
        project_name = project_data.get("project_name", "Unknown")
        version_improvements = project_data.get("version_improvements", {})
        
        for version_type, version_data in version_improvements.items():
            if version_type.lower() != "original" and version_type in all_version_improvements:
                improvements = version_data.get("improvements", [])
                all_version_improvements[version_type].extend(improvements)
    
    # Perform statistical analysis
    statistical_results = perform_improvement_statistical_analysis(all_version_improvements)
    rankings = statistical_results.get("rankings", {})
    
    # Create summary table
    for version_type, improvements in all_version_improvements.items():
        if improvements:
            rank = rankings.get(version_type, "N/A")
            rank_display = f"🥇 {rank}" if rank == 1 else str(rank)
            
            summary_data.append({
                "Version": version_type.title(),
                "Rank": rank_display,
                "Sample Count": len(improvements),
                "Mean Improvement (%)": f"{np.mean(improvements):.1f}",
                "Median Improvement (%)": f"{np.median(improvements):.1f}",
                "Std Dev (%)": f"{np.std(improvements):.1f}",
                "Min Improvement (%)": f"{np.min(improvements):.1f}",
                "Max Improvement (%)": f"{np.max(improvements):.1f}",
                "Performance": "✅ Improved" if np.mean(improvements) > 0 else "❌ Degraded" if np.mean(improvements) < 0 else "➖ No Change"
            })
    
    if summary_data:
        # Sort by rank (extract numeric part from rank display)
        def get_rank_value(rank_str):
            if isinstance(rank_str, str) and rank_str != "N/A":
                # Extract number from strings like "🥇 1" or "2"
                import re
                match = re.search(r'\d+', rank_str)
                return int(match.group()) if match else float('inf')
            return float('inf')
        
        summary_data.sort(key=lambda x: get_rank_value(x["Rank"]))
        return pd.DataFrame(summary_data)
    else:
        return pd.DataFrame()


def create_project_improvement_summary_table(improvement_results: Dict[str, Any]) -> pd.DataFrame:
    """Create project-level summary table of performance improvements"""
    
    project_results = improvement_results.get("project_results", {})
    summary_data = []
    
    for project_id, project_data in project_results.items():
        project_name = project_data.get("project_name", "Unknown")
        version_improvements = project_data.get("version_improvements", {})
        
        # Perform statistical analysis for this project
        project_improvement_data = {}
        for version_type, version_data in version_improvements.items():
            if version_type.lower() != "original":
                project_improvement_data[version_type] = version_data.get("improvements", [])
        
        statistical_results = perform_improvement_statistical_analysis(project_improvement_data)
        rankings = statistical_results.get("rankings", {})
        
        for version_type, version_data in version_improvements.items():
            if version_type.lower() != "original":
                improvements = version_data.get("improvements", [])
                if improvements:
                    rank = rankings.get(version_type, "N/A")
                    rank_display = f"🥇 {rank}" if rank == 1 else str(rank)
                    
                    summary_data.append({
                        "Project": project_name,
                        "Version": version_type.title(),
                        "Rank": rank_display,
                        "Sample Count": len(improvements),
                        "Mean Improvement (%)": f"{np.mean(improvements):.1f}",
                        "Median Improvement (%)": f"{np.median(improvements):.1f}",
                        "Std Dev (%)": f"{np.std(improvements):.1f}",
                        "Performance": "✅ Improved" if np.mean(improvements) > 0 else "❌ Degraded" if np.mean(improvements) < 0 else "➖ No Change"
                    })
    
    if summary_data:
        return pd.DataFrame(summary_data)
    else:
        return pd.DataFrame()


def create_construct_improvement_summary_table(improvement_results: Dict[str, Any]) -> pd.DataFrame:
    """Create construct-level summary table of performance improvements"""
    
    project_results = improvement_results.get("project_results", {})
    summary_data = []
    
    for project_id, project_data in project_results.items():
        project_name = project_data.get("project_name", "Unknown")
        construct_improvements = project_data.get("construct_improvements", {})
        
        for construct_id, construct_data in construct_improvements.items():
            construct_versions = construct_data.get("versions", {})
            
            # Perform statistical analysis for this construct
            construct_improvement_data = {}
            for version_type, version_data in construct_versions.items():
                if version_type.lower() != "original":
                    construct_improvement_data[version_type] = version_data.get("improvements", [])
            
            statistical_results = perform_improvement_statistical_analysis(construct_improvement_data)
            rankings = statistical_results.get("rankings", {})
            
            for version_type, version_data in construct_versions.items():
                if version_type.lower() != "original":
                    improvements = version_data.get("improvements", [])
                    if improvements:
                        rank = rankings.get(version_type, "N/A")
                        rank_display = f"🥇 {rank}" if rank == 1 else str(rank)
                        
                        summary_data.append({
                            "Project": project_name,
                            "Construct": construct_id,
                            "Version": version_type.title(),
                            "Rank": rank_display,
                            "Sample Count": len(improvements),
                            "Mean Improvement (%)": f"{np.mean(improvements):.1f}",
                            "Median Improvement (%)": f"{np.median(improvements):.1f}",
                            "Std Dev (%)": f"{np.std(improvements):.1f}",
                            "Performance": "✅ Improved" if np.mean(improvements) > 0 else "❌ Degraded" if np.mean(improvements) < 0 else "➖ No Change"
                        })
    
    if summary_data:
        return pd.DataFrame(summary_data)
    else:
        return pd.DataFrame()


def create_version_level_improvement_summary_table(improvement_results: Dict[str, Any]) -> pd.DataFrame:
    """Create version-level summary table of performance improvements"""
    
    project_results = improvement_results.get("project_results", {})
    summary_data = []
    
    # Dynamically discover all available versions (excluding original as it's the baseline)
    all_versions = set()
    for project_data in project_results.values():
        version_level_improvements = project_data.get("version_level_improvements", {})
        all_versions.update(version_level_improvements.keys())
    
    # Convert to sorted list for consistent ordering
    all_versions = sorted(list(all_versions))
    
    for project_id, project_data in project_results.items():
        project_name = project_data.get("project_name", "Unknown")
        version_level_improvements = project_data.get("version_level_improvements", {})
        
        # Perform statistical analysis for this project's version-level data
        statistical_results = perform_improvement_statistical_analysis(
            {k: v.get("improvements", []) for k, v in version_level_improvements.items()}
        )
        rankings = statistical_results.get("rankings", {})
        
        # Create entries for all versions, even if they don't have data
        for version_type in all_versions:
            version_data = version_level_improvements.get(version_type, {})
            improvements = version_data.get("improvements", [])
            
            if improvements:
                # Has data
                rank = rankings.get(version_type, "N/A")
                rank_display = f"🥇 {rank}" if rank == 1 else str(rank)
                
                from meta_artemis_modules.visualization import get_template_display_name
                
                summary_data.append({
                    "Project": project_name,
                    "Version": get_template_display_name(version_type),
                    "Rank": rank_display,
                    "Sample Count": len(improvements),
                    "Mean Improvement (%)": f"{np.mean(improvements):.1f}",
                    "Median Improvement (%)": f"{np.median(improvements):.1f}",
                    "Std Dev (%)": f"{np.std(improvements):.1f}",
                    "Performance": "✅ Improved" if np.mean(improvements) > 0 else "❌ Degraded" if np.mean(improvements) < 0 else "➖ No Change"
                })
            else:
                # No data - show N/A
                from meta_artemis_modules.visualization import get_template_display_name
                
                summary_data.append({
                    "Project": project_name,
                    "Version": get_template_display_name(version_type),
                    "Rank": "N/A",
                    "Sample Count": 0,
                    "Mean Improvement (%)": "N/A",
                    "Median Improvement (%)": "N/A",
                    "Std Dev (%)": "N/A",
                    "Performance": "N/A"
                })
    
    if summary_data:
        return pd.DataFrame(summary_data)
    else:
        return pd.DataFrame()


def create_overall_improvement_box_plot(improvement_results: Dict[str, Any]) -> Optional[go.Figure]:
    """Create overall box plot showing performance improvements across all projects"""
    
    project_results = improvement_results.get("project_results", {})
    if not project_results:
        return None
    
    # Collect all improvement data
    plot_data = []
    
    for project_id, project_data in project_results.items():
        project_name = project_data.get("project_name", "Unknown")
        version_improvements = project_data.get("version_improvements", {})
        
        for version_type, version_data in version_improvements.items():
            improvements = version_data.get("improvements", [])
            for improvement in improvements:
                plot_data.append({
                    "Project": project_name,
                    "Version": version_type.title(),
                    "Improvement (%)": improvement
                })
    
    if not plot_data:
        return None
    
    df = pd.DataFrame(plot_data)
    
    # Create box plot
    fig = go.Figure()
    
    # Define colors for versions
    version_colors = {
        "Baseline": "#FF6B6B",
        "Simplified": "#4ECDC4", 
        "Standard": "#45B7D1",
        "Enhanced": "#96CEB4"
    }
    
    versions = df["Version"].unique()
    for version in versions:
        version_data = df[df["Version"] == version]["Improvement (%)"]
        
        fig.add_trace(go.Box(
            y=version_data,
            name=version,
            marker_color=version_colors.get(version, "#999999"),
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(size=4, opacity=0.6)
        ))
    
    fig.update_layout(
        title="Performance Improvement Across All Projects<br><sub>Percentage improvement relative to original version</sub>",
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title="Version Type",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=False,
            showline=False
        ),
        yaxis=dict(
            title="Performance Improvement (%)",
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1,
            showline=False
        )
    )
    
    return fig


def create_project_improvement_box_plots(improvement_results: Dict[str, Any]) -> List[go.Figure]:
    """Create individual box plots for each project showing performance improvements"""
    
    project_results = improvement_results.get("project_results", {})
    figures = []
    
    for project_id, project_data in project_results.items():
        project_name = project_data.get("project_name", "Unknown")
        version_improvements = project_data.get("version_improvements", {})
        
        if not version_improvements:
            continue
        
        # Perform statistical analysis for this project
        project_improvement_data = {}
        for version_type, version_data in version_improvements.items():
            if version_type.lower() != "original":
                project_improvement_data[version_type] = version_data.get("improvements", [])
        
        statistical_results = perform_improvement_statistical_analysis(project_improvement_data)
        rankings = statistical_results.get("rankings", {})
        
        # Prepare data for this project
        plot_data = []
        for version_type, version_data in version_improvements.items():
            improvements = version_data.get("improvements", [])
            for improvement in improvements:
                plot_data.append({
                    "Version": version_type.title(),
                    "Improvement (%)": improvement
                })
        
        if not plot_data:
            continue
        
        df = pd.DataFrame(plot_data)
        
        # Create box plot
        fig = go.Figure()
        
        version_colors = {
            "Baseline": "#FF6B6B",
            "Simplified": "#4ECDC4", 
            "Standard": "#45B7D1",
            "Enhanced": "#96CEB4"
        }
        
        versions = df["Version"].unique()
        for version in versions:
            version_data = df[df["Version"] == version]["Improvement (%)"]
            
            fig.add_trace(go.Box(
                y=version_data,
                name=version,
                marker_color=version_colors.get(version, "#999999"),
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(size=4, opacity=0.6)
            ))
        
        # Create custom x-axis labels with rankings
        x_labels = []
        for version in versions:
            rank = rankings.get(version.lower(), "N/A")
            if rank == 1:
                x_labels.append(f"<b>{version}<br>Rank {rank}</b>")
            else:
                x_labels.append(f"{version}<br>Rank {rank}")
        
        fig.update_layout(
            title=f"Performance Improvement - {project_name}<br><sub>Percentage improvement relative to original version</sub>",
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title="Version Type",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showline=False,
                tickmode='array',
                tickvals=list(range(len(versions))),
                ticktext=x_labels
            ),
            yaxis=dict(
                title="Performance Improvement (%)",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                zerolinewidth=1,
                showline=False
            )
        )
        
        figures.append(fig)
    
    return figures


def create_version_level_improvement_box_plots(improvement_results: Dict[str, Any]) -> List[go.Figure]:
    """Create version-level box plots showing performance improvements (solutions with 10 specs)"""
    
    project_results = improvement_results.get("project_results", {})
    figures = []
    
    # Dynamically discover all available versions (excluding original as it's the baseline)
    all_versions = set()
    for project_data in project_results.values():
        version_level_improvements = project_data.get("version_level_improvements", {})
        all_versions.update(version_level_improvements.keys())
    
    # Convert to sorted list for consistent ordering
    all_versions = sorted(list(all_versions))
    
    for project_id, project_data in project_results.items():
        project_name = project_data.get("project_name", "Unknown")
        version_level_improvements = project_data.get("version_level_improvements", {})
        
        # Perform statistical analysis for this project's version-level data
        statistical_results = perform_improvement_statistical_analysis(
            {k: v.get("improvements", []) for k, v in version_level_improvements.items() if v.get("improvements", [])}
        )
        rankings = statistical_results.get("rankings", {})
        
        # Create box plot for version-level improvements
        fig = go.Figure()
        
        # Generate dynamic colors for all versions
        from meta_artemis_modules.utils import generate_colors
        color_list = generate_colors(len(all_versions))
        version_colors = dict(zip(all_versions, color_list))
        
        # Import display name function
        from meta_artemis_modules.visualization import get_template_display_name
        
        # Process all versions, even those without data
        for version_type in all_versions:
            version_data = version_level_improvements.get(version_type, {})
            improvements = version_data.get("improvements", [])
            display_name = get_template_display_name(version_type)
            
            if improvements:
                # Has data - create box plot
                fig.add_trace(go.Box(
                    y=improvements,
                    name=display_name,
                    marker_color=version_colors.get(version_type, "#999999"),
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(size=4, opacity=0.6)
                ))
            else:
                # No data - create empty placeholder
                fig.add_trace(go.Box(
                    y=[],  # Empty data
                    name=display_name,
                    marker_color=version_colors.get(version_type, "#999999"),
                    boxpoints=False,
                    line=dict(color=version_colors.get(version_type, "#999999"), width=1),
                    fillcolor=version_colors.get(version_type, "#999999"),
                    opacity=0.3
                ))
        
        # Create custom x-axis labels with rankings for all versions
        x_labels = []
        for version in all_versions:
            display_name = get_template_display_name(version)
            rank = rankings.get(version, "N/A")
            if rank == 1:
                x_labels.append(f"<b>{display_name}<br>Rank {rank}</b>")
            elif rank != "N/A":
                x_labels.append(f"{display_name}<br>Rank {rank}")
            else:
                x_labels.append(f"{display_name}")  # No rank for versions without data
        
        fig.update_layout(
            title=f"Version-Level Performance Improvements - {project_name}<br><sub>Percentage improvement relative to original version (Solutions with 10 specs)</sub>",
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                title="Version Type",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False,
                showline=False,
                tickmode='array',
                tickvals=list(range(len(all_versions))),
                ticktext=x_labels
            ),
            yaxis=dict(
                title="Performance Improvement (%)",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                zerolinecolor='gray',
                zerolinewidth=1,
                showline=False
            )
        )
        
        figures.append(fig)
    
    return figures


def create_construct_improvement_box_plots(improvement_results: Dict[str, Any]) -> List[go.Figure]:
    """Create construct-level box plots showing performance improvements"""
    
    project_results = improvement_results.get("project_results", {})
    figures = []
    
    for project_id, project_data in project_results.items():
        project_name = project_data.get("project_name", "Unknown")
        construct_improvements = project_data.get("construct_improvements", {})
        
        if not construct_improvements:
            continue
        
        # Prepare data for all constructs in this project
        plot_data = []
        construct_rankings = {}
        construct_name_to_id = {}
        
        for construct_id, construct_data in construct_improvements.items():
            construct_versions = construct_data.get("versions", {})
            construct_name = f"Construct {construct_id[:8]}..."
            construct_name_to_id[construct_name] = construct_id
            
            # Perform statistical analysis for this construct
            construct_improvement_data = {}
            for version_type, version_data in construct_versions.items():
                if version_type.lower() != "original":
                    construct_improvement_data[version_type] = version_data.get("improvements", [])
            
            statistical_results = perform_improvement_statistical_analysis(construct_improvement_data)
            construct_rankings[construct_id] = statistical_results.get("rankings", {})
            
            for version_type, version_data in construct_versions.items():
                improvements = version_data.get("improvements", [])
                for improvement in improvements:
                    plot_data.append({
                        "Construct": construct_name,
                        "Version": version_type.title(),
                        "Improvement (%)": improvement
                    })
        
        if not plot_data:
            continue
        
        df = pd.DataFrame(plot_data)
        
        # Create subplot for each construct
        constructs = df["Construct"].unique()
        if len(constructs) == 0:
            continue
            
        # Create subplots
        cols = min(2, len(constructs))
        rows = (len(constructs) + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=constructs,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        version_colors = {
            "Baseline": "#FF6B6B",
            "Simplified": "#4ECDC4", 
            "Standard": "#45B7D1",
            "Enhanced": "#96CEB4"
        }
        
        for i, construct in enumerate(constructs):
            row = i // cols + 1
            col = i % cols + 1
            
            construct_data = df[df["Construct"] == construct]
            construct_id = construct_name_to_id[construct]
            versions = list(set(construct_data["Version"].tolist()))
            rankings = construct_rankings.get(construct_id, {})
            
            for version in versions:
                version_data = construct_data[construct_data["Version"] == version]["Improvement (%)"]
                
                fig.add_trace(
                    go.Box(
                        y=version_data,
                        name=version,
                        marker_color=version_colors.get(version, "#999999"),
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8,
                        marker=dict(size=3, opacity=0.6),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
            
            # Create custom x-axis labels with rankings for this construct
            x_labels = []
            for version in versions:
                rank = rankings.get(version.lower(), "N/A")
                if rank == 1:
                    x_labels.append(f"<b>{version}<br>Rank {rank}</b>")
                else:
                    x_labels.append(f"{version}<br>Rank {rank}")
            
            # Update x-axis for this subplot
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(len(versions))),
                ticktext=x_labels,
                row=row, col=col
            )
        
        fig.update_layout(
            title=f"Construct-Level Performance Improvements - {project_name}<br><sub>Percentage improvement relative to original version</sub>",
            height=300 * rows,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Update layout for all subplots
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update all axes
        fig.update_yaxes(title_text="Improvement (%)", showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_xaxes(title_text="Version", showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        figures.append(fig)
    
    return figures


def create_improvement_summary_table(improvement_results: Dict[str, Any]) -> pd.DataFrame:
    """Create summary table of performance improvements"""
    
    project_results = improvement_results.get("project_results", {})
    summary_data = []
    
    for project_id, project_data in project_results.items():
        project_name = project_data.get("project_name", "Unknown")
        version_improvements = project_data.get("version_improvements", {})
        
        for version_type, version_data in version_improvements.items():
            mean_improvement = version_data.get("mean_improvement", 0)
            std_improvement = version_data.get("std_improvement", 0)
            count = version_data.get("count", 0)
            median_improvement = version_data.get("median_improvement", 0)
            
            summary_data.append({
                "Project": project_name,
                "Version": version_type.title(),
                "Mean Improvement (%)": f"{mean_improvement:.1f}",
                "Median Improvement (%)": f"{median_improvement:.1f}",
                "Std Dev (%)": f"{std_improvement:.1f}",
                "Sample Count": count,
                "Performance": "✅ Improved" if mean_improvement > 0 else "❌ Degraded" if mean_improvement < 0 else "➖ No Change"
            })
    
    if summary_data:
        return pd.DataFrame(summary_data)
    else:
        return pd.DataFrame()


def display_performance_improvement_analysis(analysis_results: Dict[str, Any]):
    """Display comprehensive performance improvement analysis"""
    
    st.markdown("---")
    st.markdown("### 📈 Performance Improvement Analysis")
    st.markdown("*Analysis of percentage improvements relative to the original version*")
    
    # Compute performance improvements
    with st.spinner("Computing performance improvements..."):
        improvement_results = compute_performance_improvements(analysis_results)
    
    project_results = improvement_results.get("project_results", {})
    summary = improvement_results.get("summary", {})
    
    if not project_results:
        st.warning("⚠️ No performance improvement data available")
        return
    
    # Calculate positive improvements count for success rate
    all_improvements = []
    for project_data in project_results.values():
        # Collect from version improvements
        version_improvements = project_data.get("version_improvements", {})
        for version_data in version_improvements.values():
            improvements = version_data.get("improvements", [])
            all_improvements.extend(improvements)
        
        # Collect from version-level improvements
        version_level_improvements = project_data.get("version_level_improvements", {})
        for version_data in version_level_improvements.values():
            improvements = version_data.get("improvements", [])
            all_improvements.extend(improvements)
    
    positive_improvements = len([x for x in all_improvements if x > 0])
    
    # Display overall metrics
    st.markdown("#### 📊 Overall Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Projects Analyzed", summary.get("total_projects", 0))
    with col2:
        avg_improvement = summary.get("mean_improvement", 0)
        st.metric("Average Improvement", f"{avg_improvement:.1f}%")
    with col3:
        total_count = summary.get("total_improvements", 1)
        success_rate = (positive_improvements / total_count) * 100 if total_count > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        median_improvement = summary.get("median_improvement", 0)
        st.metric("Median Improvement", f"{median_improvement:.1f}%")
    
    # Project-level box plots
    st.markdown("#### 📊 Project-Level Performance Improvements")
    project_figures = create_project_improvement_box_plots(improvement_results)
    if project_figures:
        for fig in project_figures:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No project-level improvement data available")
 
    # Project-level Summary Table
    st.markdown("#### 📊 Project-Level Summary Table")
    project_summary_table = create_project_improvement_summary_table(improvement_results)
    if not project_summary_table.empty:
        st.dataframe(project_summary_table, use_container_width=True)
    else:
        st.info("No project-level improvement data available for statistical analysis.")
       
    # Construct-level box plots
    st.markdown("#### 🏗️ Construct-Level Performance Improvements")
    construct_figures = create_construct_improvement_box_plots(improvement_results)
    if construct_figures:
        for fig in construct_figures:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No construct-level improvement data available")
    
    # Construct-level Summary Table
    st.markdown("#### 🏗️ Construct-Level Summary Table")
    construct_summary_table = create_construct_improvement_summary_table(improvement_results)
    if not construct_summary_table.empty:
        st.dataframe(construct_summary_table, use_container_width=True)
    else:
        st.info("No construct-level improvement data available for statistical analysis.")
    
    # Version-level box plots (solutions with 10 specs)
    st.markdown("#### 📋 Version-Level Performance Improvements")
    st.markdown("*Solutions with 10 specifications each*")
    version_level_figures = create_version_level_improvement_box_plots(improvement_results)
    if version_level_figures:
        for fig in version_level_figures:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No version-level improvement data available")
    
    # Version-level Summary Table
    st.markdown("#### 📋 Version-Level Summary Table")
    version_level_summary_table = create_version_level_improvement_summary_table(improvement_results)
    if not version_level_summary_table.empty:
        st.dataframe(version_level_summary_table, use_container_width=True)
    else:
        st.info("No version-level improvement data available for statistical analysis.")
    
    st.markdown("---")
    st.markdown("*Note: Positive percentages indicate performance improvements (faster execution), negative percentages indicate performance degradation (slower execution)*") 