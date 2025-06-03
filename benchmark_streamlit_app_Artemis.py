import streamlit as st
import asyncio
import json
from datetime import datetime
import os
from uuid import UUID
from artemis_client.vision.client import VisionAsyncClient, VisionSettings
from artemis_client.falcon.client import ThanosSettings, FalconSettings, FalconClient
from loguru import logger
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Import required models for solution creation
try:
    from falcon_models.rest_api.code_models import (
        FullSolutionInfoRequest, 
        SolutionSpecResponseBase,
        SolutionResultsRequest
    )
    from falcon_models.service.code import SolutionResultsBase, SolutionStatusEnum
except ImportError as e:
    logger.warning(f"Could not import falcon models: {e}")
    # Define minimal fallback classes if imports fail
    class FullSolutionInfoRequest:
        def __init__(self, specs, **kwargs):
            self.specs = specs
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self):
            return self.__dict__
    
    class SolutionSpecResponseBase:
        def __init__(self, spec_id):
            self.spec_id = spec_id
    
    class SolutionResultsRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self, mode=None):
            return self.__dict__

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

def initialize_session_state():
    """Initialize all session state variables"""
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = {
            "step": 1,  # Current workflow step
            "project_id": None,
            "project_info": None,
            "available_specs": None,
            "selected_specs": {},  # {construct_id: spec_id}
            "solutions": [],  # List of created solutions
            "solutions_created": False,
            "evaluation_started": False,
            "evaluation_complete": False,
            "solution_details": {},  # {solution_id: solution_detail}
            "falcon_client": None
        }

async def setup_falcon_client():
    """Setup and authenticate Falcon client"""
    if st.session_state.workflow_state["falcon_client"] is None:
        try:
            falcon_settings = FalconSettings.with_env_prefix("falcon", _env_file=".env")
            thanos_settings = ThanosSettings.with_env_prefix("thanos", _env_file=".env")
            falcon_client = FalconClient(falcon_settings, thanos_settings)
            falcon_client.authenticate()
            st.session_state.workflow_state["falcon_client"] = falcon_client
            return falcon_client
        except Exception as e:
            st.error(f"Failed to setup Falcon client: {e}")
            return None
    return st.session_state.workflow_state["falcon_client"]

async def get_project_info_and_specs(falcon_client, project_id: str) -> dict:
    """Get project information and available specs"""
    try:
        # Get project information
        project_info = falcon_client.get_project(project_id)
        
        # Get project constructs to find available specs
        constructs_info = falcon_client.get_constructs_info(project_id)
        
        specs_by_construct = {}
        for construct_id, construct in constructs_info.items():
            construct_specs = {
                'construct_id': str(construct_id),
                'original_spec': {
                    'spec_id': str(construct.original_spec_id),
                    'name': 'Original Spec',
                    'type': 'original'
                },
                'recommendation_specs': []
            }
            
            # Add custom specs (recommendations)
            if hasattr(construct, 'custom_specs') and construct.custom_specs:
                for spec in construct.custom_specs:
                    if spec.enabled:  # Only include enabled specs
                        spec_info = {
                            'spec_id': str(spec.id),
                            'name': spec.name,
                            'type': 'recommendation',
                            'content_preview': spec.content[:100] + '...' if len(spec.content) > 100 else spec.content
                        }
                        construct_specs['recommendation_specs'].append(spec_info)
            
            specs_by_construct[str(construct_id)] = construct_specs
        
        return {
            'project_info': {
                'id': str(project_info.id),
                'name': project_info.name,
                'description': project_info.description,
                'num_constructs': len(constructs_info)
            },
            'specs_by_construct': specs_by_construct
        }
        
    except Exception as e:
        logger.error(f"Error getting project info and specs: {str(e)}")
        return None

async def create_optimization_run(falcon_client, project_id: str, selected_specs: dict, optimization_id: str = None) -> dict:
    """Create a single solution with all selected specs from all constructs"""
    logger.info(f"🚀 Creating solution with {len(selected_specs)} selected specs")
    
    try:
        # Convert all selected specs to solution specs for ONE solution
        solution_specs = []
        
        for i, (construct_id, spec_info) in enumerate(selected_specs.items(), 1):
            try:
                # Extract spec_id from the spec_info dictionary
                spec_id = spec_info['spec_id'] if isinstance(spec_info, dict) else spec_info
                
                # Convert to UUID and create SolutionSpecResponseBase
                spec_uuid = UUID(spec_id)
                solution_spec = SolutionSpecResponseBase(spec_id=spec_uuid)
                solution_specs.append(solution_spec)
                logger.info(f"   ✅ Added spec {i}: {spec_id[:8]}...")
                
            except Exception as spec_error:
                logger.error(f"   ❌ Error processing spec {i}: {str(spec_error)}")
                raise spec_error
        
        # Create the single solution request with ALL specs
        status = SolutionStatusEnum.created if hasattr(SolutionStatusEnum, 'created') else 'created'
        solution_request = FullSolutionInfoRequest(
            specs=solution_specs,
            status=status
        )
        
        # Use provided optimization ID or get default
        if optimization_id:
            optimisation_id = optimization_id
            logger.info(f"✅ Using provided optimization ID: {optimisation_id}")
            
            # Verify the optimization exists
            try:
                optimization_info = falcon_client.get_optimisation(optimisation_id)
                logger.info(f"✅ Optimization verified: {getattr(optimization_info, 'name', 'Unknown')}")
            except Exception as verify_error:
                logger.error(f"❌ Could not verify optimization ID {optimisation_id}: {verify_error}")
                raise Exception(f"Optimization ID {optimisation_id} does not exist or is not accessible")
        else:
            optimisation_id = await get_or_create_optimisation_id(falcon_client, project_id)
            if optimisation_id is None:
                logger.error(f"❌ No optimisation_id available - evaluation may fail!")
        
        # Add the solution
        response = falcon_client.add_solution(
            project_id=project_id,
            optimisation_id=optimisation_id,
            solution=solution_request
        )
        
        # Extract solution ID
        if isinstance(response, dict):
            solution_id = response.get('solution_id') or response.get('id') or response.get('solutionId')
        else:
            solution_id = str(response)
        
        if not solution_id:
            logger.warning(f"⚠️ Could not extract solution_id from response!")
            solution_id = "UNKNOWN"
        
        logger.info(f"✅ Solution created: {solution_id} ({len(solution_specs)} specs)")
        
        # Create result
        result = {
            'success': True,
            'message': 'Single solution created successfully with all selected specs',
            'solutions': [{
                'solution_id': solution_id,
                'specs_count': len(solution_specs),
                'constructs_count': len(selected_specs),
                'selected_specs': selected_specs,
                'spec_metadata': selected_specs,  # Store the spec names and types for Step 4
                'response': response
            }],
            'total_solutions': 1,
            'debug_info': {
                'project_id': project_id,
                'selected_specs_count': len(selected_specs),
                'solution_specs_count': len(solution_specs),
                'response_type': str(type(response))
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Solution creation failed: {str(e)}")
        
        result = {
            'success': False,
            'error': str(e),
            'message': f'Failed to create optimization context: {str(e)}',
            'debug_info': {
                'project_id': project_id,
                'selected_specs': selected_specs,
                'exception_type': str(type(e))
            }
        }
        
        return result



async def evaluate_solution_async(falcon_client, solution_id: str, custom_worker_name: str = None, custom_command: str = None, unit_test: bool = False) -> dict:
    """Evaluate a solution with proper default configurations"""
    logger.info(f"🚀 Starting evaluation for solution: {solution_id[:8]}...")
    
    try:
        # Convert solution_id to UUID
        solution_uuid = UUID(solution_id)
        
        # Check if solution exists and get its details
        try:
            solution_check = falcon_client.get_solution(solution_id)
            project_id = getattr(solution_check, 'project_id', None)
            opt_id = getattr(solution_check, 'optimisation_id', None)
            
            if opt_id is None:
                logger.warning(f"⚠️ Solution has no optimisation_id - evaluation may fail!")
                
        except Exception as check_error:
            logger.error(f"❌ Failed to get solution details: {check_error}")
            project_id = None
            opt_id = None
        
        # Get default configurations if not provided by user
        if custom_worker_name is None or custom_command is None:
            # Get project configuration for default command
            if project_id and custom_command is None:
                try:
                    project_info = falcon_client.get_project(str(project_id))
                    default_command = getattr(project_info, 'perf_command', None)
                    if default_command:
                        custom_command = default_command
                        logger.info(f"✅ Using project command: {custom_command}")
                except Exception as project_error:
                    logger.warning(f"⚠️ Failed to get project configuration: {project_error}")
            
            # Use the known working worker name
            if custom_worker_name is None:
                custom_worker_name = "jing_runner"
                logger.info(f"✅ Using worker: {custom_worker_name}")
        
        # Call falcon client evaluate_solution
        response = falcon_client.evaluate_solution(
            solution_id=solution_uuid,
            custom_worker_name=custom_worker_name,
            custom_command=custom_command,
            unit_test=unit_test
        )
        
        # Check for issues in response
        if isinstance(response, dict):
            if 'error' in response:
                logger.error(f"❌ Error in response: {response['error']}")
            if 'message' in response and response['message'] == '':
                logger.warning(f"⚠️ Empty response message")
        
        logger.info(f"✅ Evaluation started successfully")
        
        result = {
            'success': True,
            'response': response,
            'message': 'Solution evaluation started successfully',
            'debug_info': {
                'solution_id': solution_id,
                'solution_uuid': str(solution_uuid),
                'response_type': str(type(response))
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Solution evaluation failed: {str(e)}")
        
        result = {
            'success': False,
            'error': str(e),
            'message': f'Failed to evaluate solution: {str(e)}',
            'debug_info': {
                'solution_id': solution_id,
                'exception_type': str(type(e))
            }
        }
        
        return result

async def get_solution_details_async(falcon_client, solution_id: str) -> dict:
    """Get detailed information about a specific solution"""
    logger.info(f"🔍 Getting details for solution: {solution_id[:8]}...")
    
    try:
        # Call falcon client get_solution
        solution_detail = falcon_client.get_solution(solution_id)
        
        # Get status
        if hasattr(solution_detail, 'status'):
            status = str(solution_detail.status)
            logger.info(f"   Status: {status}")
        else:
            status = "UNKNOWN"
            logger.warning(f"   ⚠️ No status found")
        
        result = {
            'solution_id': solution_id,
            'status': status,
            'results': None,
            'specs': [],
            'raw_solution_detail': str(solution_detail)
        }
        
        # Extract results (metrics data)
        if hasattr(solution_detail, 'results') and solution_detail.results:
            results = solution_detail.results
            
            if hasattr(results, 'values'):
                values = results.values
                
                # Log metrics summary
                if isinstance(values, dict):
                    metrics = list(values.keys())
                    logger.info(f"   Metrics found: {metrics}")
                    for metric_name, metric_data in values.items():
                        if isinstance(metric_data, list):
                            logger.info(f"     {metric_name}: {len(metric_data)} values")
                
                result['results'] = {
                    'values': values
                }
                
                # Extract stats if available
                if hasattr(results, 'stats'):
                    result['results']['stats'] = results.stats
        else:
            logger.info(f"   No results found")
        
        # Extract specs information
        if hasattr(solution_detail, 'specs') and solution_detail.specs:
            specs_list = solution_detail.specs
            logger.info(f"   Specs found: {len(specs_list)}")
            
            for i, spec in enumerate(specs_list):
                spec_info = {
                    'index': i,
                    'raw_spec': str(spec),
                    'spec_type': str(type(spec))
                }
                
                # Extract spec_id and construct_id
                if hasattr(spec, 'spec_id'):
                    spec_info['spec_id'] = str(spec.spec_id)
                if hasattr(spec, 'construct_id'):
                    spec_info['construct_id'] = str(spec.construct_id)
                
                # Extract other common attributes
                for attr in ['id', 'name', 'content', 'enabled', 'status']:
                    if hasattr(spec, attr):
                        spec_info[attr] = str(getattr(spec, attr))
                
                result['specs'].append(spec_info)
        else:
            logger.info(f"   No specs found")
        
        logger.info(f"✅ Solution details retrieved successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get solution details: {str(e)}")
        return None

def display_solution_details(solution_detail: dict):
    """Display detailed information about a solution with interactive graphs"""
    st.subheader(f"Solution Details: {solution_detail['solution_id']}")
    
    # Display status with color coding
    status = solution_detail['status']
    if 'fail' in status.lower():
        st.error(f"**Status:** {status}")
    elif 'success' in status.lower() or 'complete' in status.lower():
        st.success(f"**Status:** {status}")
    else:
        st.info(f"**Status:** {status}")
    
    # Display results (metrics data) with interactive graphs
    if solution_detail['results']:
        st.subheader("📊 Performance Metrics")
        
        if 'values' in solution_detail['results']:
            values = solution_detail['results']['values']
            
            # Create tabs for different metrics
            metric_tabs = []
            if 'runtime' in values:
                metric_tabs.append("⏱️ Runtime")
            if 'memory' in values:
                metric_tabs.append("💾 Memory")
            if 'cpu' in values:
                metric_tabs.append("🖥️ CPU")
            
            if metric_tabs:
                tabs = st.tabs(metric_tabs)
                
                # Runtime visualization
                if 'runtime' in values and len(metric_tabs) > 0:
                    with tabs[0]:
                        runtime_data = values['runtime']
                        st.write(f"**{len(runtime_data)} runtime measurements**")
                        
                        # Create subplot with line chart and histogram
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Runtime Over Measurements', 'Runtime Distribution', 
                                          'Runtime Statistics', 'Runtime Box Plot'),
                            specs=[[{"colspan": 2}, None],
                                   [{"type": "bar"}, {"type": "box"}]]
                        )
                        
                        # Line chart
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(1, len(runtime_data) + 1)),
                                y=runtime_data,
                                mode='lines+markers',
                                name='Runtime',
                                line=dict(color='blue', width=2),
                                marker=dict(size=6)
                            ),
                            row=1, col=1
                        )
                        
                        # Statistics bar chart
                        stats = {
                            'Min': min(runtime_data),
                            'Max': max(runtime_data),
                            'Mean': sum(runtime_data) / len(runtime_data),
                            'Median': sorted(runtime_data)[len(runtime_data)//2]
                        }
                        
                        fig.add_trace(
                            go.Bar(
                                x=list(stats.keys()),
                                y=list(stats.values()),
                                name='Statistics',
                                marker_color=['green', 'red', 'blue', 'orange']
                            ),
                            row=2, col=1
                        )
                        
                        # Box plot
                        fig.add_trace(
                            go.Box(
                                y=runtime_data,
                                name='Runtime Distribution',
                                marker_color='lightblue'
                            ),
                            row=2, col=2
                        )
                        
                        fig.update_layout(
                            height=600,
                            title_text="Runtime Analysis",
                            showlegend=False
                        )
                        fig.update_xaxes(title_text="Measurement #", row=1, col=1)
                        fig.update_yaxes(title_text="Runtime (seconds)", row=1, col=1)
                        fig.update_yaxes(title_text="Runtime (seconds)", row=2, col=1)
                        fig.update_yaxes(title_text="Runtime (seconds)", row=2, col=2)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Min Runtime", f"{stats['Min']:.6f}s")
                        with col2:
                            st.metric("Max Runtime", f"{stats['Max']:.6f}s")
                        with col3:
                            st.metric("Mean Runtime", f"{stats['Mean']:.6f}s")
                        with col4:
                            st.metric("Median Runtime", f"{stats['Median']:.6f}s")
                
                # Memory visualization
                if 'memory' in values and len(metric_tabs) > 1:
                    tab_idx = 1 if 'runtime' in values else 0
                    with tabs[tab_idx]:
                        memory_data = values['memory']
                        st.write(f"**{len(memory_data)} memory measurements**")
                        
                        # Convert bytes to MB for better readability
                        memory_mb = [m / (1024 * 1024) for m in memory_data]
                        
                        # Create memory visualization
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Memory Usage Over Time', 'Memory Distribution',
                                          'Memory Statistics (MB)', 'Memory Usage Pattern'),
                            specs=[[{"colspan": 2}, None],
                                   [{"type": "bar"}, {"type": "scatter"}]]
                        )
                        
                        # Memory usage line chart
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(1, len(memory_mb) + 1)),
                                y=memory_mb,
                                mode='lines+markers',
                                name='Memory Usage',
                                line=dict(color='purple', width=2),
                                marker=dict(size=6)
                            ),
                            row=1, col=1
                        )
                        
                        # Memory statistics
                        mem_stats = {
                            'Min': min(memory_mb),
                            'Max': max(memory_mb),
                            'Mean': sum(memory_mb) / len(memory_mb),
                            'Median': sorted(memory_mb)[len(memory_mb)//2]
                        }
                        
                        fig.add_trace(
                            go.Bar(
                                x=list(mem_stats.keys()),
                                y=list(mem_stats.values()),
                                name='Memory Stats',
                                marker_color=['lightgreen', 'lightcoral', 'lightblue', 'lightyellow']
                            ),
                            row=2, col=1
                        )
                        
                        # Memory pattern scatter
                        fig.add_trace(
                            go.Scatter(
                                x=list(range(1, len(memory_mb) + 1)),
                                y=memory_mb,
                                mode='markers',
                                name='Memory Points',
                                marker=dict(
                                    size=8,
                                    color=memory_mb,
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Memory (MB)")
                                )
                            ),
                            row=2, col=2
                        )
                        
                        fig.update_layout(
                            height=600,
                            title_text="Memory Usage Analysis",
                            showlegend=False
                        )
                        fig.update_xaxes(title_text="Measurement #", row=1, col=1)
                        fig.update_yaxes(title_text="Memory (MB)", row=1, col=1)
                        fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
                        fig.update_xaxes(title_text="Measurement #", row=2, col=2)
                        fig.update_yaxes(title_text="Memory (MB)", row=2, col=2)
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Memory summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Min Memory", f"{mem_stats['Min']:.2f} MB")
                        with col2:
                            st.metric("Max Memory", f"{mem_stats['Max']:.2f} MB")
                        with col3:
                            st.metric("Mean Memory", f"{mem_stats['Mean']:.2f} MB")
                        with col4:
                            st.metric("Median Memory", f"{mem_stats['Median']:.2f} MB")
                
                # CPU visualization
                if 'cpu' in values:
                    tab_idx = len([m for m in ['runtime', 'memory'] if m in values])
                    with tabs[tab_idx]:
                        cpu_data = values['cpu']
                        st.write(f"**{len(cpu_data)} CPU measurements**")
                        
                        # CPU usage visualization
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(cpu_data) + 1)),
                            y=cpu_data,
                            mode='lines+markers',
                            name='CPU Usage',
                            line=dict(color='orange', width=3),
                            marker=dict(size=8),
                            fill='tonexty'
                        ))
                        
                        fig.update_layout(
                            title="CPU Usage Over Time",
                            xaxis_title="Measurement #",
                            yaxis_title="CPU Usage (%)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # CPU summary
                        cpu_stats = {
                            'Min': min(cpu_data),
                            'Max': max(cpu_data),
                            'Mean': sum(cpu_data) / len(cpu_data)
                        }
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Min CPU", f"{cpu_stats['Min']:.2f}%")
                        with col2:
                            st.metric("Max CPU", f"{cpu_stats['Max']:.2f}%")
                        with col3:
                            st.metric("Mean CPU", f"{cpu_stats['Mean']:.2f}%")
        
        # Display stats if available
        if 'stats' in solution_detail['results']:
            st.subheader("📈 Statistical Summary")
            stats_data = solution_detail['results']['stats']
            
            # Convert stats to a more readable format
            if isinstance(stats_data, dict):
                # Convert ResultsStatsModel objects to dictionaries
                formatted_stats = {}
                for metric_name, stats_obj in stats_data.items():
                    if hasattr(stats_obj, '__dict__'):
                        # Convert the stats object to a dictionary
                        stats_dict = {}
                        for attr in ['mean', 'mean_adjusted', 'std', 'min', 'max']:
                            if hasattr(stats_obj, attr):
                                value = getattr(stats_obj, attr)
                                stats_dict[attr] = float(value) if value is not None else 0.0
                        formatted_stats[metric_name] = stats_dict
                    else:
                        formatted_stats[metric_name] = stats_obj
                
                # Create a clean dataframe
                stats_df = pd.DataFrame.from_dict(formatted_stats, orient='index')
                
                # Format the dataframe for better display
                if not stats_df.empty:
                    # Round numeric values for better display
                    numeric_columns = stats_df.select_dtypes(include=[float, int]).columns
                    stats_df[numeric_columns] = stats_df[numeric_columns].round(6)
                    
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Also create a visual representation
                    if len(formatted_stats) > 0:
                        # Create a metrics comparison chart
                        metrics_for_chart = []
                        for metric, stats in formatted_stats.items():
                            if isinstance(stats, dict) and 'mean' in stats:
                                metrics_for_chart.append({
                                    'Metric': metric.title(),
                                    'Mean': stats['mean'],
                                    'Min': stats['min'],
                                    'Max': stats['max'],
                                    'Std Dev': stats['std']
                                })
                        
                        if metrics_for_chart:
                            chart_df = pd.DataFrame(metrics_for_chart)
                            
                            # Create a comparison chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                name='Mean',
                                x=chart_df['Metric'],
                                y=chart_df['Mean'],
                                marker_color='lightblue'
                            ))
                            
                            fig.add_trace(go.Bar(
                                name='Min',
                                x=chart_df['Metric'],
                                y=chart_df['Min'],
                                marker_color='lightgreen'
                            ))
                            
                            fig.add_trace(go.Bar(
                                name='Max',
                                x=chart_df['Metric'],
                                y=chart_df['Max'],
                                marker_color='lightcoral'
                            ))
                            
                            fig.update_layout(
                                title="Performance Metrics Comparison",
                                xaxis_title="Metrics",
                                yaxis_title="Values",
                                barmode='group',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No statistical data available")
            else:
                # Fallback to JSON display if it's not a dictionary
                st.json(stats_data)
    else:
        st.warning("**Results:** No metrics data available for this solution")
    
    # Display specs with enhanced visualization
    st.subheader("🔧 Specification Details")
    if solution_detail['specs']:
        st.write(f"**Total specs found:** {len(solution_detail['specs'])}")
        
        # Create a specs overview chart
        specs_data = []
        for i, spec in enumerate(solution_detail['specs']):
            specs_data.append({
                'Spec Index': i + 1,
                'Spec ID': spec.get('spec_id', 'Unknown')[:8] + '...',
                'Construct ID': spec.get('construct_id', 'Unknown')[:8] + '...',
                'Full Spec ID': spec.get('spec_id', 'Unknown'),
                'Full Construct ID': spec.get('construct_id', 'Unknown')
            })
        
        specs_df = pd.DataFrame(specs_data)
        
        # Create an interactive specs chart
        fig = px.bar(
            specs_df, 
            x='Spec Index', 
            y=[1] * len(specs_df),  # All bars same height
            hover_data=['Full Spec ID', 'Full Construct ID'],
            title="Specs Overview (Hover for details)",
            labels={'y': 'Spec Count', 'Spec Index': 'Specification Index'}
        )
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display specs table
        display_df = specs_df[['Spec Index', 'Spec ID', 'Construct ID']].copy()
        st.dataframe(display_df, use_container_width=True)
        
        # Get spec metadata from the solution if available
        spec_metadata = {}
        solution_data = st.session_state.workflow_state.get('solutions', [{}])[0]
        
        # Handle both manual selection metadata (selected_specs) and auto-test metadata (spec_metadata)
        if 'spec_metadata' in solution_data:
            metadata = solution_data['spec_metadata']
            for key, spec_info in metadata.items():
                if isinstance(spec_info, dict):
                    spec_metadata[spec_info['spec_id']] = spec_info
        elif 'selected_specs' in solution_data:
            # Fallback to selected_specs for manual selection
            metadata = solution_data['selected_specs']
            for construct_id, spec_info in metadata.items():
                if isinstance(spec_info, dict):
                    spec_metadata[spec_info['spec_id']] = spec_info
        
        # Show detailed spec information in expandable sections
        for i, spec in enumerate(solution_detail['specs']):
            spec_id = spec.get('spec_id', f'Spec {i+1}')
            construct_id = spec.get('construct_id', 'Unknown')
            
            # Try to get spec name from stored metadata first
            if spec_id in spec_metadata:
                metadata = spec_metadata[spec_id]
                spec_name = metadata['spec_name']
                spec_type = metadata['spec_type']
                
                if spec_type == 'original':
                    spec_type_display = "Original Spec"
                    type_icon = "📄"
                elif spec_type == 'optimization':
                    spec_type_display = f"Optimization Spec: {spec_name}"
                    type_icon = "⚙️"
                else:
                    spec_type_display = f"LLM Recommendation: {spec_name}"
                    type_icon = "🔧"
            else:
                # Fallback to trying to extract from spec data
                spec_name = spec.get('name', 'Unknown Spec')
                spec_type_display = "Original Spec"
                type_icon = "📄"
                
                # Check if this looks like a recommendation (has AI-generated characteristics)
                if any(keyword in spec_name.lower() for keyword in ['recommendation', 'optimized', 'improved', 'generated']):
                    spec_type_display = f"LLM Recommendation: {spec_name}"
                    type_icon = "🔧"
                elif spec_name != 'Unknown Spec' and spec_name != 'Original Spec':
                    spec_type_display = f"Custom Spec: {spec_name}"
                    type_icon = "🔧"
            
            with st.expander(f"{type_icon} Spec {i+1}: {spec_type_display} ({spec_id[:8]}...)"):
                # Show spec metadata
                st.write("**Specification Information:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Name:** {spec_name}")
                    st.info(f"**Type:** {spec_type_display}")
                
                with col2:
                    st.code(f"Spec ID: {spec_id}", language='text')
                    st.code(f"Construct ID: {construct_id}", language='text')
                
                # Show source code if available
                st.write("**Source Code:**")
                
                # Try to get the actual code content
                code_content = None
                if 'content' in spec:
                    code_content = spec['content']
                elif 'raw_spec' in spec:
                    # Try to extract code from raw spec data
                    raw_spec = spec['raw_spec']
                    if 'content' in str(raw_spec):
                        code_content = raw_spec
                
                if code_content and code_content != 'Not available':
                    # Determine language for syntax highlighting
                    language = 'python'  # Default to python
                    if 'language' in spec:
                        language = spec['language'].lower()
                    
                    st.code(code_content, language=language)
                else:
                    st.warning("Source code not available for this specification")
                    
                    # Show available spec data for debugging
                    if show_debug:
                        st.write("**Debug - Available Spec Data:**")
                        available_keys = list(spec.keys())
                        st.write(f"Available keys: {available_keys}")
                        for key in available_keys[:5]:  # Show first 5 keys
                            st.write(f"**{key}:** `{str(spec[key])[:100]}...`")
    else:
        st.warning("No specs found for this solution")

async def get_existing_solutions(falcon_client, project_id: str, optimization_id: str = None) -> dict:
    """Get existing solutions from the specified optimization"""
    try:
        # Use provided optimization ID or default
        optimisation_id = optimization_id or "49b08c56-620f-4ae8-96d3-1675e6a17b2a"
        
        # Get optimization info
        optimization_info = falcon_client.get_optimisation(optimisation_id)
        
        # Get all solutions for this optimization
        solutions_response = falcon_client.get_solutions(optimisation_id, per_page=-1)
        
        existing_solutions = []
        for solution in solutions_response.docs:
            # Get detailed solution info
            try:
                solution_detail = falcon_client.get_solution(str(solution.id))
                
                # Extract basic info
                solution_info = {
                    'solution_id': str(solution.id),
                    'name': getattr(solution, 'name', f'Solution {str(solution.id)[:8]}...'),
                    'status': str(getattr(solution, 'status', 'unknown')),
                    'created_at': getattr(solution, 'created_at', None),
                    'specs_count': len(getattr(solution_detail, 'specs', [])),
                    'has_results': hasattr(solution_detail, 'results') and solution_detail.results is not None
                }
                
                # Try to get performance metrics if available
                if solution_info['has_results'] and hasattr(solution_detail.results, 'values'):
                    values = solution_detail.results.values
                    if 'runtime' in values and values['runtime']:
                        solution_info['avg_runtime'] = sum(values['runtime']) / len(values['runtime'])
                    if 'memory' in values and values['memory']:
                        solution_info['avg_memory'] = sum(values['memory']) / len(values['memory']) / (1024 * 1024)  # Convert to MB
                
                existing_solutions.append(solution_info)
                
            except Exception as detail_error:
                logger.warning(f"Could not get details for solution {solution.id}: {detail_error}")
                # Add basic info even if details fail
                existing_solutions.append({
                    'solution_id': str(solution.id),
                    'name': getattr(solution, 'name', f'Solution {str(solution.id)[:8]}...'),
                    'status': str(getattr(solution, 'status', 'unknown')),
                    'created_at': getattr(solution, 'created_at', None),
                    'specs_count': 'Unknown',
                    'has_results': False
                })
        
        return {
            'success': True,
            'optimization_id': optimisation_id,
            'optimization_name': getattr(optimization_info, 'name', 'Unknown Optimization'),
            'solutions': existing_solutions,
            'total_solutions': len(existing_solutions)
        }
        
    except Exception as e:
        logger.error(f"Error getting existing solutions: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'solutions': [],
            'total_solutions': 0
        }

async def get_or_create_optimisation_id(falcon_client, project_id: str) -> str:
    """Get an existing optimisation ID or use the default provided by Artemis team"""
    try:
        # Use the optimization ID provided by Artemis team
        default_optimisation_id = "49b08c56-620f-4ae8-96d3-1675e6a17b2a"
        
        # Verify the optimization exists
        try:
            optimization_info = falcon_client.get_optimisation(default_optimisation_id)
            logger.info(f"✅ Using optimization: {getattr(optimization_info, 'name', 'Unknown')}")
            return default_optimisation_id
        except Exception as verify_error:
            logger.warning(f"⚠️ Could not verify optimization: {verify_error}")
            return default_optimisation_id
        
    except Exception as e:
        logger.error(f"❌ Error getting optimisation ID: {str(e)}")
        return None

async def find_existing_optimisations(falcon_client, project_id: str) -> list:
    """Try to find existing optimisation runs for the project"""
    logger.info("🔍 SEARCHING FOR EXISTING OPTIMISATION RUNS:")
    
    try:
        # Unfortunately, there's no get_optimisations_by_project method either
        # We would need to know specific optimisation IDs to retrieve them
        
        logger.warning(f"⚠️ LIMITATION: No method to list optimisations by project")
        logger.warning(f"   - Falcon client only has get_optimisation(id) method")
        logger.warning(f"   - Cannot discover existing optimisation runs")
        
        return []
        
    except Exception as e:
        logger.error(f"❌ Error searching for optimisations: {str(e)}")
        return []

def main():
    st.title("🚀 Artemis Integrated Solution Workflow")
    
    # Initialize session state
    initialize_session_state()
    
    # Get current workflow state
    workflow_state = st.session_state.workflow_state
    
    # Add debug toggle in sidebar
    with st.sidebar:
        st.header("Debug Options")
        show_debug = st.checkbox("Show Debug Information", value=False, help="Display detailed logs and debug information")
        
        if show_debug:
            st.subheader("Debug Information")
            if "debug_logs" in st.session_state:
                with st.expander("Recent Debug Logs", expanded=False):
                    for log_entry in st.session_state.get("debug_logs", [])[-10:]:  # Show last 10 entries
                        st.text(log_entry)
    
    # Progress indicator - Updated to 4 steps instead of 5
    progress_steps = ["Project Setup", "Solution Creation", "Evaluation", "Results"]
    current_step = workflow_state["step"]
    
    # Display progress bar
    progress_cols = st.columns(len(progress_steps))
    for i, step_name in enumerate(progress_steps):
        with progress_cols[i]:
            if i + 1 < current_step:
                st.success(f"✅ {step_name}")
            elif i + 1 == current_step:
                st.info(f"🔄 {step_name}")
            else:
                st.write(f"⏳ {step_name}")
    
    st.divider()
    
    # Step 1: Project Setup
    if current_step == 1:
        st.header("Step 1: Project Setup")
        st.write("Enter a project ID to load available specifications.")
        
        project_id = st.text_input(
            "Project ID",
            value=workflow_state.get("project_id") or "6c47d53e-7384-44d8-be9d-c186a7af480a",
            help="Enter the project ID to load available specs"
        )
        
        if st.button("Load Project", key="load_project_btn"):
            if not project_id:
                st.error("Please enter a project ID")
            else:
                with st.spinner("Loading project information and specifications..."):
                    falcon_client = asyncio.run(setup_falcon_client())
                    if falcon_client:
                        project_data = asyncio.run(get_project_info_and_specs(falcon_client, project_id))
                        
                        if project_data:
                            workflow_state["project_id"] = project_id
                            workflow_state["project_info"] = project_data["project_info"]
                            workflow_state["available_specs"] = project_data["specs_by_construct"]
                            workflow_state["step"] = 2
                            st.success(f"Loaded project: {project_data['project_info']['name']}")
                            st.rerun()
                        else:
                            st.error("Failed to load project information")
    
    # Step 2: Solution Creation (Spec Selection + Solution Creation → Auto-proceed to Evaluation)
    elif current_step == 2:
        st.header("Step 2: Solution Creation")
        
        project_info = workflow_state["project_info"]
        available_specs = workflow_state["available_specs"]
        
        st.write(f"**Project:** {project_info['name']}")
        st.write(f"**Description:** {project_info.get('description', 'No description')}")
        st.write(f"**Number of Constructs:** {project_info['num_constructs']}")
        
        if not workflow_state["solutions_created"]:
            # Show two options: Manual selection or Auto-test
            st.subheader("🎯 Choose Solution Creation Method")
            
            creation_method = st.radio(
                "Select how to proceed:",
                ["Manual Spec Selection", "Use Existing Solution"],
                help="Manual selection lets you choose specific specs for each construct to create a new solution. Use existing solution lets you select from previously created solutions."
            )
            
            if creation_method == "Manual Spec Selection":
                st.subheader("🔧 Manual Spec Selection")
                st.write("Select specific specs for each construct to create a custom solution.")
                
                # Important note about partial solutions
                st.info("ℹ️ **Note**: You can select specs from any subset of constructs. For constructs without explicit specs, the system will automatically use their original specs during execution.")
                
                # Optimization ID selection
                st.subheader("🎯 Target Optimization")
                
                optimization_choice = st.radio(
                    "Choose optimization option:",
                    ["Use Default Optimization", "Enter Custom Optimization ID"],
                    help="Select whether to use the default optimization or specify a custom one"
                )
                
                if optimization_choice == "Use Default Optimization":
                    optimization_id = "49b08c56-620f-4ae8-96d3-1675e6a17b2a"
                    st.info(f"📋 Using default optimization: `{optimization_id}`")
                    
                    # Try to verify the default optimization exists
                    try:
                        falcon_client = workflow_state["falcon_client"]
                        if falcon_client:
                            optimization_info = falcon_client.get_optimisation(optimization_id)
                            st.success(f"✅ Default optimization found: {getattr(optimization_info, 'name', 'Unknown')}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not verify default optimization: {str(e)}")
                        
                else:  # Enter Custom Optimization ID
                    optimization_id = st.text_input(
                        "Custom Optimization ID",
                        value="",
                        placeholder="Enter a new or existing optimization ID",
                        help="Enter a custom optimization ID. If it doesn't exist, a new optimization may be created."
                    )
                    
                    if optimization_id:
                        # Try to verify the custom optimization exists
                        try:
                            falcon_client = workflow_state["falcon_client"]
                            if falcon_client:
                                optimization_info = falcon_client.get_optimisation(optimization_id)
                                st.success(f"✅ Existing optimization found: {getattr(optimization_info, 'name', 'Unknown')}")
                        except Exception as e:
                            st.info(f"ℹ️ Optimization ID not found - a new optimization may be created")
                            st.code(f"Details: {str(e)}", language="text")
                    else:
                        st.warning("⚠️ Please enter a custom optimization ID")
                
                # Display constructs and their available specs for selection
                selected_specs = {}
                
                if available_specs:
                    st.write("**Available Constructs and Specs:**")
                    
                    for construct_id, construct_data in available_specs.items():
                        with st.expander(f"🏗️ Construct: {construct_id[:8]}...", expanded=True):
                            st.write(f"**Full Construct ID:** `{construct_id}`")
                            
                            # Prepare options for radio button selection
                            spec_options = []
                            spec_ids = []
                            
                            # Add original spec
                            original_spec = construct_data['original_spec']
                            spec_options.append(f"📄 Original: {original_spec['name']}")
                            spec_ids.append(original_spec['spec_id'])
                            
                            # Add recommendation specs
                            recommendation_specs = construct_data['recommendation_specs']
                            for i, rec_spec in enumerate(recommendation_specs):
                                spec_options.append(f"🔧 Recommendation {i+1}: {rec_spec['name']}")
                                spec_ids.append(rec_spec['spec_id'])
                            
                            # Add "None" option
                            spec_options.append("❌ Don't use this construct")
                            spec_ids.append(None)
                            
                            # Radio button for spec selection
                            selected_option = st.radio(
                                f"Select spec for construct {construct_id[:8]}...",
                                range(len(spec_options)),
                                format_func=lambda x: spec_options[x],
                                key=f"spec_selection_{construct_id}",
                                index=len(spec_options) - 1  # Default to "Don't use"
                            )
                            
                            # Store the selected spec ID and metadata
                            if spec_ids[selected_option] is not None:
                                selected_spec_id = spec_ids[selected_option]
                                
                                # Determine spec name and type
                                if selected_option == 0:  # Original spec
                                    spec_name = original_spec['name']
                                    spec_type = 'original'
                                else:  # Recommendation spec
                                    rec_index = selected_option - 1
                                    spec_name = recommendation_specs[rec_index]['name']
                                    spec_type = 'recommendation'
                                
                                selected_specs[construct_id] = {
                                    'spec_id': selected_spec_id,
                                    'spec_name': spec_name,
                                    'spec_type': spec_type,
                                    'construct_id': construct_id
                                }
                            
                            # Show details of selected spec
                            if selected_option < len(spec_ids) - 1:  # Not "Don't use"
                                selected_spec_id = spec_ids[selected_option]
                                st.success(f"✅ Selected: `{selected_spec_id}`")
                                
                                # Show content preview for recommendations
                                if selected_option > 0:  # It's a recommendation
                                    rec_index = selected_option - 1
                                    if rec_index < len(recommendation_specs):
                                        rec_spec = recommendation_specs[rec_index]
                                        if 'content_preview' in rec_spec:
                                            st.write("**📖 Spec Content Preview:**")
                                            st.code(rec_spec['content_preview'], language='python')
                            else:
                                st.info("This construct will not be included in the solution.")
                            
                            st.divider()
                
                # Show selection summary and validation
                total_constructs = len(available_specs)
                selected_constructs = len(selected_specs)
                
                if selected_specs:
                    st.subheader("📋 Selection Summary")
                    
                    # Show selection status
                    if selected_constructs == total_constructs:
                        st.success(f"✅ **Full Custom Solution**: Selected custom specs for all {selected_constructs}/{total_constructs} constructs")
                    else:
                        missing_constructs = total_constructs - selected_constructs
                        st.info(f"📋 **Partial Custom Solution**: Selected custom specs for {selected_constructs}/{total_constructs} constructs ({missing_constructs} will use original specs)")
                    
                    st.write(f"**Selected specs:**")
                    for i, (construct_id, spec_info) in enumerate(selected_specs.items(), 1):
                        if isinstance(spec_info, dict):
                            spec_name = spec_info['spec_name']
                            spec_type = spec_info['spec_type']
                            spec_id = spec_info['spec_id']
                            type_icon = "📄" if spec_type == 'original' else "🔧"
                            st.write(f"{i}. Construct `{construct_id[:8]}...` → {type_icon} {spec_name} (`{spec_id[:8]}...`)")
                        else:
                            # Fallback for old format
                            st.write(f"{i}. Construct `{construct_id[:8]}...` → Spec `{spec_info[:8]}...`")
                    
                    # Show constructs using original specs if any
                    if selected_constructs < total_constructs:
                        missing_construct_ids = set(available_specs.keys()) - set(selected_specs.keys())
                        st.write(f"**📄 Constructs using original specs ({len(missing_construct_ids)}):**")
                        for construct_id in missing_construct_ids:
                            st.write(f"  - Construct `{construct_id[:8]}...` (will use original spec)")
                    
                    # Create solution button
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("← Back to Project Setup", key="back_to_step1_manual"):
                            workflow_state["step"] = 1
                            st.rerun()
                    
                    with col2:
                        # Show different button text based on selection
                        if selected_constructs == total_constructs:
                            button_text = "🚀 Create Full Custom Solution"
                            button_help = "Create a solution with custom specs for all constructs"
                        else:
                            button_text = "📋 Create Partial Custom Solution"
                            button_help = "Create a solution with custom specs for selected constructs (others use original specs)"
                        
                        if st.button(button_text, key="create_custom_solution_btn", help=button_help):
                            # Validate optimization ID based on choice
                            optimization_id_valid = False
                            if optimization_choice == "Use Default Optimization":
                                optimization_id_valid = True  # Default is always valid
                            else:  # Custom optimization ID
                                if optimization_id and optimization_id.strip():
                                    optimization_id_valid = True
                                else:
                                    st.error("❌ Please enter a custom optimization ID")
                            
                            if optimization_id_valid:
                                with st.spinner("Creating custom solution with selected specs..."):
                                    falcon_client = workflow_state["falcon_client"]
                                    result = asyncio.run(create_optimization_run(
                                        falcon_client,
                                        workflow_state["project_id"],
                                        selected_specs,
                                        optimization_id.strip()
                                    ))
                                    
                                    if result["success"]:
                                        workflow_state["solutions"] = result["solutions"]
                                        workflow_state["solutions_created"] = True
                                        workflow_state["step"] = 3  # Jump directly to evaluation step
                                        st.success(result["message"])
                                        
                                        # Show the created solution details
                                        solution = result["solutions"][0]
                                        st.write(f"**✅ Custom Solution Created Successfully**")
                                        st.write(f"**Solution ID:** `{solution['solution_id']}`")
                                        st.write(f"**Optimization ID:** `{optimization_id}`")
                                        specs_count = solution.get('specs_count', 'Unknown')
                                        constructs_count = solution.get('constructs_count', 'Unknown')
                                        st.write(f"  - Specs in solution: {specs_count}")
                                        st.write(f"  - Constructs represented: {constructs_count}")
                                        
                                        if selected_constructs == total_constructs:
                                            st.success("✅ Full custom solution created - all constructs use custom specs!")
                                        else:
                                            st.success("📋 Partial custom solution created - selected constructs use custom specs, others use original specs!")
                                        
                                        st.info("🚀 Proceeding to evaluation step...")
                                        st.rerun()
                                    else:
                                        st.error(result["message"])
                                        if "error" in result:
                                            st.code(result["error"])
                else:
                    st.warning("⚠️ Please select at least one spec from the available constructs.")
                    
                    # Back button when no specs selected
                    if st.button("← Back to Project Setup", key="back_to_step1_no_specs"):
                        workflow_state["step"] = 1
                        st.rerun()
            
            else:  # Use Existing Solution
                st.subheader("📋 Use Existing Solution")
                st.info("Select from previously created solutions in the optimization. This allows you to evaluate existing solutions without creating new ones.")
                
                # Optimization ID selection for existing solutions
                st.subheader("🎯 Source Optimization")
                
                existing_optimization_choice = st.radio(
                    "Choose source optimization:",
                    ["Use Default Optimization", "Enter Custom Optimization ID"],
                    help="Select which optimization to load existing solutions from",
                    key="existing_optimization_choice"
                )
                
                if existing_optimization_choice == "Use Default Optimization":
                    existing_optimization_id = "49b08c56-620f-4ae8-96d3-1675e6a17b2a"
                    st.info(f"📋 Loading from default optimization: `{existing_optimization_id}`")
                else:  # Enter Custom Optimization ID
                    existing_optimization_id = st.text_input(
                        "Custom Optimization ID",
                        value="",
                        placeholder="Enter optimization ID to load solutions from",
                        help="Enter the optimization ID to load existing solutions from.",
                        key="existing_optimization_id"
                    )
                
                # Load existing solutions
                if st.button("🔍 Load Existing Solutions", key="load_existing_solutions_btn"):
                    # Validate optimization ID based on choice
                    existing_optimization_id_valid = False
                    if existing_optimization_choice == "Use Default Optimization":
                        existing_optimization_id_valid = True  # Default is always valid
                    else:  # Custom optimization ID
                        if existing_optimization_id and existing_optimization_id.strip():
                            existing_optimization_id_valid = True
                        else:
                            st.error("❌ Please enter a custom optimization ID")
                    
                    if existing_optimization_id_valid:
                        with st.spinner("Loading existing solutions..."):
                            falcon_client = workflow_state["falcon_client"]
                            existing_data = asyncio.run(get_existing_solutions(
                                falcon_client,
                                workflow_state["project_id"],
                                existing_optimization_id.strip()
                            ))
                        
                        if existing_data["success"]:
                            st.session_state.existing_solutions_data = existing_data
                            st.success(f"Found {existing_data['total_solutions']} existing solutions")
                            st.rerun()
                        else:
                            st.error(f"Failed to load existing solutions: {existing_data.get('error', 'Unknown error')}")
                
                # Display existing solutions if loaded
                if hasattr(st.session_state, 'existing_solutions_data') and st.session_state.existing_solutions_data["success"]:
                    existing_data = st.session_state.existing_solutions_data
                    
                    st.write(f"**Optimization:** {existing_data['optimization_name']}")
                    st.write(f"**Optimization ID:** `{existing_data['optimization_id']}`")
                    st.write(f"**Total Solutions:** {existing_data['total_solutions']}")
                    
                    if existing_data['solutions']:
                        st.subheader("📊 Available Solutions")
                        
                        # Create a table of solutions
                        solutions_data = []
                        for i, solution in enumerate(existing_data['solutions']):
                            # Format creation date
                            created_str = "Unknown"
                            if solution.get('created_at'):
                                try:
                                    from datetime import datetime
                                    if isinstance(solution['created_at'], str):
                                        created_dt = datetime.fromisoformat(solution['created_at'].replace('Z', '+00:00'))
                                    else:
                                        created_dt = solution['created_at']
                                    created_str = created_dt.strftime("%Y-%m-%d %H:%M")
                                except:
                                    created_str = str(solution['created_at'])[:16]
                            
                            # Format performance metrics
                            runtime_str = "No data"
                            memory_str = "No data"
                            if solution.get('avg_runtime'):
                                runtime_str = f"{solution['avg_runtime']:.6f}s"
                            if solution.get('avg_memory'):
                                memory_str = f"{solution['avg_memory']:.2f} MB"
                            
                            # Status with color coding
                            status = solution['status']
                            status_icon = "✅" if "success" in status.lower() or "complete" in status.lower() else "❌" if "fail" in status.lower() else "⏳"
                            
                            solutions_data.append({
                                'Select': i,
                                'Solution ID': solution['solution_id'][:8] + '...',
                                'Name': solution['name'],
                                'Status': f"{status_icon} {status}",
                                'Specs': solution['specs_count'],
                                'Has Results': "✅" if solution['has_results'] else "❌",
                                'Avg Runtime': runtime_str,
                                'Avg Memory': memory_str,
                                'Created': created_str,
                                'Full ID': solution['solution_id']  # Hidden column for reference
                            })
                        
                        # Display solutions table
                        df = pd.DataFrame(solutions_data)
                        display_df = df.drop('Full ID', axis=1)  # Hide the full ID column
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Solution selection
                        if solutions_data:
                            selected_solution_idx = st.selectbox(
                                "Select a solution to use:",
                                range(len(solutions_data)),
                                format_func=lambda x: f"Solution {x+1}: {solutions_data[x]['Name']} ({solutions_data[x]['Status']})",
                                key="existing_solution_selection"
                            )
                            
                            selected_solution = existing_data['solutions'][selected_solution_idx]
                            
                            # Show selected solution details
                            st.subheader("📋 Selected Solution Details")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.info(f"**ID:** `{selected_solution['solution_id'][:8]}...`")
                                st.info(f"**Name:** {selected_solution['name']}")
                                st.info(f"**Status:** {selected_solution['status']}")
                            
                            with col2:
                                st.info(f"**Specs Count:** {selected_solution['specs_count']}")
                                st.info(f"**Has Results:** {'Yes' if selected_solution['has_results'] else 'No'}")
                                st.info(f"**Created:** {created_str}")
                            
                            with col3:
                                if selected_solution.get('avg_runtime'):
                                    st.success(f"**Avg Runtime:** {selected_solution['avg_runtime']:.6f}s")
                                else:
                                    st.warning("**Runtime:** No data")
                                
                                if selected_solution.get('avg_memory'):
                                    st.success(f"**Avg Memory:** {selected_solution['avg_memory']:.2f} MB")
                                else:
                                    st.warning("**Memory:** No data")
                            
                            # Navigation and use buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("← Back to Project Setup", key="back_to_step1_existing"):
                                    workflow_state["step"] = 1
                                    st.rerun()
                            
                            with col2:
                                if st.button("📋 Use This Solution", key="use_existing_solution_btn"):
                                    # Create a solution entry in the same format as newly created solutions
                                    solution_entry = {
                                        'solution_id': selected_solution['solution_id'],
                                        'specs_count': selected_solution['specs_count'],
                                        'constructs_count': 'Unknown',  # We don't have this info for existing solutions
                                        'existing_solution': True,  # Mark as existing
                                        'solution_name': selected_solution['name'],
                                        'solution_status': selected_solution['status'],
                                        'has_results': selected_solution['has_results']
                                    }
                                    
                                    workflow_state["solutions"] = [solution_entry]
                                    workflow_state["solutions_created"] = True
                                    workflow_state["step"] = 3  # Jump directly to evaluation step
                                    
                                    st.success(f"Selected existing solution: {selected_solution['name']}")
                                    st.info("🚀 Proceeding to evaluation step...")
                                    st.rerun()
                    else:
                        st.warning("No solutions found in the optimization.")
                        
                        # Back button when no solutions
                        if st.button("← Back to Project Setup", key="back_to_step1_no_existing"):
                            workflow_state["step"] = 1
                            st.rerun()
                else:
                    # Show back button when solutions not loaded yet
                    if st.button("← Back to Project Setup", key="back_to_step1_not_loaded"):
                        workflow_state["step"] = 1
                        st.rerun()
        
        else:
            # Solution already created - automatically redirect to evaluation
            st.success("✅ Solutions created successfully!")
            st.info("🚀 Redirecting to evaluation step...")
            
            # Automatically proceed to Step 3 (Evaluation)
            workflow_state["step"] = 3
            st.rerun()
    
    # Step 3: Evaluation (Auto-reached after Solution Creation)
    elif current_step == 3:
        st.header("Step 3: Solution Evaluation")
        
        solutions = workflow_state["solutions"]
        if not solutions:
            st.error("No solutions available for evaluation")
            return
        
        # Get the solution to evaluate
        if len(solutions) > 1:
            solution_options = []
            for i, solution in enumerate(solutions):
                solution_options.append(f"Solution {i+1}: {solution['solution_id'][:8]}...")
            
            selected_solution_idx = st.selectbox(
                "Select solution to evaluate:",
                range(len(solutions)),
                format_func=lambda x: solution_options[x]
            )
            selected_solution = solutions[selected_solution_idx]
        else:
            selected_solution = solutions[0]
            st.write(f"**Solution ID:** `{selected_solution['solution_id']}`")
        
        solution_id = selected_solution['solution_id']
        
        if not workflow_state["evaluation_started"]:
            # Get default configurations
            falcon_client = workflow_state["falcon_client"]
            project_id = workflow_state["project_id"]
            
            default_worker = "jing_runner"
            default_command = None
            
            try:
                project_info = falcon_client.get_project(project_id)
                default_command = getattr(project_info, 'perf_command', None)
            except Exception as config_error:
                st.warning(f"Could not load project configurations: {config_error}")
            
            # Configuration options (expanded by default)
            st.subheader("🛠️ Evaluation Configuration")
            
            custom_worker = st.text_input(
                "Worker Name", 
                value=default_worker,
                help="The worker that will execute the evaluation"
            )
            
            custom_command = st.text_area(
                "Performance Command", 
                value=default_command or "",
                help="Command to run for performance evaluation"
            )
            
            unit_test = st.checkbox("Run Unit Tests", help="Include unit tests in evaluation")
            
            # Final configuration that will be used
            final_worker = custom_worker.strip() if custom_worker.strip() else default_worker
            final_command = custom_command.strip() if custom_command.strip() else default_command
            
            # Show summary in a compact format
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"👷 **Worker:** `{final_worker}`")
            with col2:
                if final_command:
                    st.success(f"⚡ **Command:** `{final_command}`")
                else:
                    st.error("❌ **Command:** Not configured")
            with col3:
                st.info(f"🧪 **Unit Tests:** {'Yes' if unit_test else 'No'}")
            
            if st.button("Start Evaluation", key="start_evaluation_btn"):
                with st.spinner("Starting solution evaluation..."):
                    falcon_client = workflow_state["falcon_client"]
                    result = asyncio.run(evaluate_solution_async(
                        falcon_client, 
                        solution_id,
                        custom_worker_name=custom_worker if custom_worker.strip() else None,
                        custom_command=custom_command if custom_command.strip() else None,
                        unit_test=unit_test
                    ))
                    
                    # Store debug information
                    if "debug_info" in result:
                        if "debug_logs" not in st.session_state:
                            st.session_state.debug_logs = []
                        st.session_state.debug_logs.append(f"Evaluation Start Debug: {result['debug_info']}")
                    
                    if result["success"]:
                        workflow_state["evaluation_started"] = True
                        st.success(result["message"])
                        st.info("Evaluation started. This may take several minutes...")
                        
                        # Show debug information if enabled
                        if show_debug:
                            with st.expander("🔍 Evaluation Start Debug Information"):
                                st.json(result.get("debug_info", {}))
                                if "response" in result:
                                    st.subheader("Raw Evaluation Response")
                                    st.code(str(result["response"]), language="json")
                        
                        st.rerun()
                    else:
                        st.error(result["message"])
                        if "error" in result:
                            st.code(result["error"])
                        
                        # Show debug information for errors
                        with st.expander("🔍 Error Debug Information"):
                            st.json(result.get("debug_info", {}))
        
        elif not workflow_state["evaluation_complete"]:
            st.info("🔄 Evaluation in progress...")
            
            if st.button("Check Status", key="check_status_btn"):
                with st.spinner("Checking evaluation status..."):
                    falcon_client = workflow_state["falcon_client"]
                    solution_detail = asyncio.run(get_solution_details_async(falcon_client, solution_id))
                    
                    if solution_detail:
                        status = solution_detail["status"].lower()
                        st.write(f"**Current Status:** {solution_detail['status']}")
                        
                        if "complete" in status or "success" in status or solution_detail["results"]:
                            workflow_state["evaluation_complete"] = True
                            workflow_state["solution_details"][solution_id] = solution_detail
                            st.success("✅ Evaluation completed!")
                            st.rerun()
                        elif "fail" in status or "error" in status:
                            st.error("❌ Evaluation failed!")
                            workflow_state["evaluation_complete"] = True
                            workflow_state["solution_details"][solution_id] = solution_detail
                            st.rerun()
                        else:
                            st.info("Evaluation still in progress...")
                    else:
                        st.error("Failed to get solution status")
        
        else:
            st.success("✅ Evaluation completed!")
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Solution Creation", key="back_to_step2"):
                    workflow_state["step"] = 2
                    workflow_state["evaluation_started"] = False
                    workflow_state["evaluation_complete"] = False
                    st.rerun()
            
            with col2:
                if st.button("View Results →", key="proceed_to_step4"):
                    workflow_state["step"] = 4
                    st.rerun()
    
    # Step 4: Results Analysis
    elif current_step == 4:
        st.header("Step 4: Results Analysis")
        
        solutions = workflow_state["solutions"]
        solution_details = workflow_state["solution_details"]
        
        if not solutions:
            st.error("No solutions available")
            return
        
        # Let user select which solution results to view
        if len(solutions) > 1:
            solution_options = []
            for i, solution in enumerate(solutions):
                solution_id = solution['solution_id']
                has_results = solution_id in solution_details
                status = "✅ Evaluated" if has_results else "⏳ Not evaluated"
                specs_count = solution.get('specs_count', 'Unknown')
                constructs_count = solution.get('constructs_count', 'Unknown')
                solution_options.append(f"Solution {i+1}: {solution_id[:8]}... ({specs_count} specs, {constructs_count} constructs) {status}")
            
            selected_solution_idx = st.selectbox(
                "Select solution to view results:",
                range(len(solutions)),
                format_func=lambda x: solution_options[x]
            )
            selected_solution = solutions[selected_solution_idx]
        else:
            selected_solution = solutions[0]
        
        solution_id = selected_solution['solution_id']
        solution_detail = solution_details.get(solution_id)
        
        if solution_detail:
            display_solution_details(solution_detail)
            
            # Show debug information if enabled
            if show_debug:
                with st.expander("🔍 Solution Details Debug Information"):
                    st.subheader("Raw Solution Detail")
                    st.code(solution_detail.get("raw_solution_detail", "Not available"), language="text")
                    
                    st.subheader("Processed Solution Data")
                    debug_data = {
                        "solution_id": solution_id,
                        "status": solution_detail["status"],
                        "has_results": solution_detail["results"] is not None,
                        "num_specs": len(solution_detail["specs"]),
                        "specs_details": solution_detail["specs"]
                    }
                    st.json(debug_data)
            
            # Download button
            st.subheader("Download Solution Data")
            solution_json = json.dumps(solution_detail, indent=2, default=str)
            st.download_button(
                label="Download Solution Details",
                data=solution_json,
                file_name=f"solution_{solution_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download solution details as JSON"
            )
            
            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Evaluation", key="back_to_step3"):
                    workflow_state["step"] = 3
                    st.rerun()
            
            with col2:
                if st.button("Start New Workflow", key="restart_workflow"):
                    # Reset workflow state
                    st.session_state.workflow_state = {
                        "step": 1,
                        "project_id": None,
                        "project_info": None,
                        "available_specs": None,
                        "selected_specs": {},
                        "solutions": [],
                        "solutions_created": False,
                        "evaluation_started": False,
                        "evaluation_complete": False,
                        "solution_details": {},
                        "falcon_client": workflow_state["falcon_client"]  # Keep the client
                    }
                    st.rerun()
        else:
            st.error("No solution details available")
    
    # Sidebar with workflow information
    with st.sidebar:
        st.header("Workflow Status")
        
        if workflow_state["project_info"]:
            st.write(f"**Project:** {workflow_state['project_info']['name']}")
        
        if workflow_state["solutions"]:
            st.write(f"**Solutions:** {len(workflow_state['solutions'])} selected")
            for i, solution in enumerate(workflow_state["solutions"][:3], 1):  # Show first 3
                specs_count = solution.get('specs_count', 'Unknown')
                constructs_count = solution.get('constructs_count', 'Unknown')
                
                # Check if it's an existing solution
                if solution.get('existing_solution', False):
                    solution_name = solution.get('solution_name', 'Unknown')
                    solution_status = solution.get('solution_status', 'Unknown')
                    st.write(f"  {i}. 📋 **Existing:** {solution_name}")
                    st.write(f"     `{solution['solution_id'][:8]}...` ({solution_status})")
                    st.write(f"     ({specs_count} specs)")
                else:
                    st.write(f"  {i}. 🆕 **New:** `{solution['solution_id'][:8]}...`")
                    st.write(f"     ({specs_count} specs, {constructs_count} constructs)")
                
                # Show selected specs if available
                if 'selected_specs' in solution:
                    with st.expander(f"View Selected Specs for Solution {i}"):
                        for construct_id, spec_info in solution['selected_specs'].items():
                            if isinstance(spec_info, dict):
                                spec_name = spec_info['spec_name']
                                spec_type = spec_info['spec_type']
                                spec_id = spec_info['spec_id']
                                type_icon = "📄" if spec_type == 'original' else "🔧"
                                st.write(f"🏗️ `{construct_id[:8]}...` → {type_icon} {spec_name} (`{spec_id[:8]}...`)")
                            else:
                                # Fallback for old format
                                st.write(f"🏗️ `{construct_id[:8]}...` → `{spec_info[:8]}...`")
                elif 'optimization_specs' in solution:
                    with st.expander(f"View Optimization Specs for Solution {i}"):
                        for j, spec_id in enumerate(solution['optimization_specs'], 1):
                            st.write(f"{j}. `{spec_id[:8]}...`")
                elif solution.get('existing_solution', False):
                    st.write(f"     ℹ️ Existing solution - specs loaded from optimization")
            if len(workflow_state["solutions"]) > 3:
                st.write(f"  ... and {len(workflow_state['solutions']) - 3} more")
        
        st.write(f"**Current Step:** {current_step}/4")
        
        # Quick navigation (only to completed steps)
        st.subheader("Quick Navigation")
        if current_step > 1:
            if st.button("Go to Project Setup", key="nav_step1"):
                workflow_state["step"] = 1
                st.rerun()
        
        if current_step > 2:
            if st.button("Go to Solution Creation", key="nav_step2"):
                workflow_state["step"] = 2
                st.rerun()
        
        if current_step > 3:
            if st.button("Go to Evaluation", key="nav_step3"):
                workflow_state["step"] = 3
                st.rerun()
        
        if current_step == 4:
            if st.button("Go to Results", key="nav_step4"):
                workflow_state["step"] = 4
                st.rerun()

if __name__ == "__main__":
    main() 