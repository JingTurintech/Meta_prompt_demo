import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, Any, List, Tuple
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

def load_data(llm_scores_file: str, runtime_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LLM scores and runtime data from CSV files."""
    print(f"\nLoading data from:")
    print(f"- LLM scores: {llm_scores_file}")
    print(f"- Runtime data: {runtime_file}")
    
    llm_df = pd.read_csv(llm_scores_file)
    runtime_df = pd.read_csv(runtime_file)
    
    # Convert metric_measurements from string to float list
    runtime_df['metric_measurements'] = runtime_df['metric_measurements'].apply(
        lambda x: [float(v) for v in str(x).split(',')]
    )
    
    # Get original version runtime (where num_specs_in_solution = 0)
    original_runtime = runtime_df[runtime_df['num_specs_in_solution'] == 0]['metric_measurements'].iloc[0]
    original_mean_runtime = np.mean(original_runtime)
    
    # Calculate mean runtime for each solution
    runtime_df['mean_runtime'] = runtime_df['metric_measurements'].apply(np.mean)
    
    # Calculate runtime change (negative means faster = better)
    runtime_df['runtime_change'] = runtime_df['mean_runtime'] - original_mean_runtime
    
    # Calculate percentage change
    runtime_df['runtime_change_percent'] = (runtime_df['runtime_change'] / original_mean_runtime) * 100
    
    return llm_df, runtime_df, original_mean_runtime

def compute_correlations(llm_df: pd.DataFrame, runtime_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute correlations between LLM scores and runtime changes, grouped by construct."""
    print("\nComputing correlations...")
    
    # Filter for only Performance scores
    llm_df = llm_df[llm_df['task'] == 'Performance']
    print(f"\nFiltered for Performance scores only. Number of scores: {len(llm_df)}")
    
    # Filter runtime data for solutions with only 1 spec
    runtime_df = runtime_df[runtime_df['num_specs_in_solution'] == 1]
    print(f"\nFiltered for solutions with 1 spec only. Number of solutions: {len(runtime_df)}")
    
    # Prepare data for correlation analysis
    correlation_data = []
    
    # First group by construct_id, then by spec_id
    for construct_id in runtime_df['construct_id'].unique():
        if construct_id == 'All':  # Skip aggregate measurements
            continue
            
        construct_runtime = runtime_df[runtime_df['construct_id'] == construct_id]
        
        # For each spec in this construct
        for spec_id in construct_runtime['spec_id'].unique():
            llm_scores = llm_df[llm_df['spec_id'] == spec_id]
            runtime_data = construct_runtime[construct_runtime['spec_id'] == spec_id]
            
            if len(runtime_data) > 0:
                # Add correlation data for each model
                for _, row in llm_scores.iterrows():
                    correlation_data.append({
                        'construct_id': construct_id,
                        'spec_id': spec_id,
                        'model': row['model'],
                        'llm_score': row['score'],
                        'runtime': runtime_data['mean_runtime'].iloc[0],
                        'runtime_change': runtime_data['runtime_change'].iloc[0],
                        'runtime_change_percent': runtime_data['runtime_change_percent'].iloc[0]
                    })
    
    # Convert to DataFrame
    corr_df = pd.DataFrame(correlation_data)
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(f"Number of unique constructs: {len(corr_df['construct_id'].unique())}")
    print(f"Number of unique specs: {len(corr_df['spec_id'].unique())}")
    print(f"Number of total correlation points: {len(corr_df)}")
    print("\nRuntime change statistics by construct:")
    for construct in corr_df['construct_id'].unique():
        construct_data = corr_df[corr_df['construct_id'] == construct]
        print(f"\nConstruct {construct}:")
        print(construct_data['runtime_change'].describe())
    
    # Compute correlations for each model, grouped by construct
    results = {}
    for model in corr_df['model'].unique():
        model_data = corr_df[corr_df['model'] == model]
        construct_results = {}
        
        # Overall correlation for the model
        if len(model_data) >= 2:
            pearson_r, pearson_p = stats.pearsonr(model_data['llm_score'], model_data['runtime_change'])
            spearman_r, spearman_p = stats.spearmanr(model_data['llm_score'], model_data['runtime_change'])
            
            construct_results['overall'] = {
                'pearson_correlation': pearson_r,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_r,
                'spearman_p_value': spearman_p,
                'n_samples': len(model_data)
            }
        
        # Per-construct correlations
        for construct in model_data['construct_id'].unique():
            construct_data = model_data[model_data['construct_id'] == construct]
            if len(construct_data) >= 2:
                pearson_r, pearson_p = stats.pearsonr(construct_data['llm_score'], construct_data['runtime_change'])
                spearman_r, spearman_p = stats.spearmanr(construct_data['llm_score'], construct_data['runtime_change'])
                
                construct_results[construct] = {
                    'pearson_correlation': pearson_r,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_r,
                    'spearman_p_value': spearman_p,
                    'n_samples': len(construct_data)
                }
        
        results[model] = construct_results
    
    return {
        'correlations': results,
        'raw_data': corr_df.to_dict('records')
    }

def plot_correlation_scatter(data: pd.DataFrame, model: str, original_runtime: float) -> go.Figure:
    """Create an interactive scatter plot for a specific model."""
    model_data = data[data['model'] == model]
    
    # Create scatter plot
    fig = px.scatter(
        model_data,
        x='llm_score',
        y='runtime_change',
        title=f'🎯 LLM Score vs Runtime Change for {model}<br>(Negative change = faster = better)',
        labels={
            'llm_score': 'LLM Score (higher = better)',
            'runtime_change': 'Runtime Change (ms, negative = faster = better)'
        },
        hover_data=['spec_id', 'runtime_change_percent']
    )
    
    # Add a reference line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="No Change")
    
    # Update hover template to show percentage change
    fig.update_traces(
        hovertemplate="<br>".join([
            "LLM Score: %{x}",
            "Runtime Change: %{y:.3f} ms",
            "Change: %{customdata[1]:.2f}%",
            "Spec ID: %{customdata[0]}",
            "<extra></extra>"
        ])
    )
    
    return fig

def plot_spec_comparison(data: pd.DataFrame, spec_id: str) -> go.Figure:
    """Create a comparison plot for a specific spec showing scores from all models and runtime."""
    spec_data = data[data['spec_id'] == spec_id]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add runtime as a horizontal line
    runtime = spec_data['runtime'].iloc[0]
    fig.add_hline(
        y=runtime,
        line_dash="dash",
        line_color="red",
        secondary_y=True,
        name=f"Runtime: {runtime:.2f}ms"
    )
    
    # Add bar chart for LLM scores
    fig.add_trace(
        go.Bar(
            x=spec_data['model'],
            y=spec_data['llm_score'],
            name="LLM Score",
            text=spec_data['llm_score'].round(2),
            textposition='auto',
        ),
        secondary_y=False
    )
    
    # Update layout
    fig.update_layout(
        title=f"Spec {spec_id}: LLM Scores vs Runtime",
        showlegend=True,
        height=400
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="LLM Score", secondary_y=False)
    fig.update_yaxes(title_text="Runtime (ms)", secondary_y=True)
    
    return fig

def perform_bayesian_regression(model_data: pd.DataFrame) -> dict:
    """
    Perform hierarchical Bayesian regression to predict runtime change from LLM scores,
    accounting for construct-specific effects.
    
    Args:
        model_data: DataFrame containing 'construct_id', 'llm_score', and 'runtime_change' columns
    
    Returns:
        dict containing model results and predictions
    """
    # Standardize the data
    X = (model_data['llm_score'] - model_data['llm_score'].mean()) / model_data['llm_score'].std()
    y = model_data['runtime_change']
    
    # Create construct index mapping
    constructs = model_data['construct_id'].unique()
    construct_to_idx = {construct: idx for idx, construct in enumerate(constructs)}
    construct_idx = np.array([construct_to_idx[c] for c in model_data['construct_id']])
    n_constructs = len(constructs)
    
    # Build and run the hierarchical model
    with pm.Model() as model:
        # Global (population-level) priors
        global_intercept = pm.Normal('global_intercept', mu=0, sigma=10)
        global_slope = pm.Normal('global_slope', mu=0, sigma=10)
        
        # Construct-level variance parameters
        intercept_sigma = pm.HalfNormal('intercept_sigma', sigma=10)
        slope_sigma = pm.HalfNormal('slope_sigma', sigma=10)
        
        # Construct-specific effects
        construct_intercepts = pm.Normal('construct_intercepts', 
                                       mu=global_intercept,
                                       sigma=intercept_sigma,
                                       shape=n_constructs)
        construct_slopes = pm.Normal('construct_slopes',
                                   mu=global_slope,
                                   sigma=slope_sigma,
                                   shape=n_constructs)
        
        # Observation noise
        sigma = pm.HalfNormal('sigma', sigma=10)
        
        # Linear model with construct-specific effects
        mu = construct_intercepts[construct_idx] + construct_slopes[construct_idx] * X
        
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y)
        
        # Sample from the posterior
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    
    # Extract posterior samples
    global_slope_samples = trace.posterior['global_slope'].values.flatten()
    global_intercept_samples = trace.posterior['global_intercept'].values.flatten()
    construct_slopes_samples = trace.posterior['construct_slopes'].values.reshape(-1, n_constructs)
    construct_intercepts_samples = trace.posterior['construct_intercepts'].values.reshape(-1, n_constructs)
    
    # Generate predictions for each construct
    X_new = np.linspace(X.min(), X.max(), 100)
    X_orig = X_new * model_data['llm_score'].std() + model_data['llm_score'].mean()
    
    predictions = {}
    
    # Global predictions
    y_pred_global = np.zeros((len(global_slope_samples), len(X_new)))
    for i in range(len(global_slope_samples)):
        y_pred_global[i] = global_intercept_samples[i] + global_slope_samples[i] * X_new
    
    predictions['global'] = {
        'X_pred': X_orig,
        'y_mean': y_pred_global.mean(axis=0),
        'y_lower': np.percentile(y_pred_global, 2.5, axis=0),
        'y_upper': np.percentile(y_pred_global, 97.5, axis=0),
        'slope_mean': global_slope_samples.mean(),
        'slope_std': global_slope_samples.std(),
        'prob_negative_slope': (global_slope_samples < 0).mean()
    }
    
    # Construct-specific predictions
    for construct_idx, construct in enumerate(constructs):
        y_pred = np.zeros((len(construct_slopes_samples), len(X_new)))
        for i in range(len(construct_slopes_samples)):
            y_pred[i] = (construct_intercepts_samples[i, construct_idx] + 
                        construct_slopes_samples[i, construct_idx] * X_new)
        
        predictions[construct] = {
            'X_pred': X_orig,
            'y_mean': y_pred.mean(axis=0),
            'y_lower': np.percentile(y_pred, 2.5, axis=0),
            'y_upper': np.percentile(y_pred, 97.5, axis=0),
            'slope_mean': construct_slopes_samples[:, construct_idx].mean(),
            'slope_std': construct_slopes_samples[:, construct_idx].std(),
            'prob_negative_slope': (construct_slopes_samples[:, construct_idx] < 0).mean()
        }
    
    return predictions

def plot_bayesian_regression(data: pd.DataFrame, model: str, bayes_results: dict) -> go.Figure:
    """Create an interactive plot showing Bayesian regression results."""
    model_data = data[data['model'] == model]
    
    # Create scatter plot with points colored by construct
    fig = px.scatter(
        model_data,
        x='llm_score',
        y='runtime_change',
        color='construct_id',
        title=f'🎯 LLM Score vs Runtime Change for {model}<br>Global and Construct-Specific Effects',
        labels={
            'llm_score': 'LLM Score (higher = better)',
            'runtime_change': 'Runtime Change (ms, negative = faster = better)',
            'construct_id': 'Code Construct'
        },
        hover_data=['spec_id', 'runtime_change_percent']
    )
    
    # Add global regression line
    global_results = bayes_results['global']
    fig.add_trace(
        go.Scatter(
            x=global_results['X_pred'],
            y=global_results['y_mean'],
            mode='lines',
            name='Global Trend',
            line=dict(color='black', width=2),
            showlegend=True
        )
    )
    
    # Add global credible interval
    fig.add_trace(
        go.Scatter(
            x=global_results['X_pred'].tolist() + global_results['X_pred'].tolist()[::-1],
            y=global_results['y_upper'].tolist() + global_results['y_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,0,0,0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Global 95% CI',
            showlegend=True
        )
    )
    
    # Add construct-specific regression lines
    for construct in model_data['construct_id'].unique():
        if construct in bayes_results:
            construct_results = bayes_results[construct]
            
            # Add construct regression line
            fig.add_trace(
                go.Scatter(
                    x=construct_results['X_pred'],
                    y=construct_results['y_mean'],
                    mode='lines',
                    name=f'Trend: {construct}',
                    line=dict(dash='dash'),
                    showlegend=True
                )
            )
            
            # Add construct credible interval
            fig.add_trace(
                go.Scatter(
                    x=construct_results['X_pred'].tolist() + construct_results['X_pred'].tolist()[::-1],
                    y=construct_results['y_upper'].tolist() + construct_results['y_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(128,128,128,0.1)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'95% CI: {construct}',
                    showlegend=True
                )
            )
    
    # Add a reference line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="No Change")
    
    # Update hover template
    fig.update_traces(
        hovertemplate="<br>".join([
            "LLM Score: %{x}",
            "Runtime Change: %{y:.3f} ms",
            "Change: %{customdata[1]:.2f}%",
            "Spec ID: %{customdata[0]}",
            "Construct: %{marker.color}",
            "<extra></extra>"
        ]),
        selector=dict(mode='markers')
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def main():
    st.set_page_config(page_title="LLM Score vs Runtime Analysis", layout="wide")
    st.title("🔍 LLM Score vs Runtime Analysis")
    
    # File upload
    llm_scores_file = st.file_uploader("Upload LLM scores CSV", type="csv")
    runtime_file = st.file_uploader("Upload runtime data CSV", type="csv")
    
    if llm_scores_file and runtime_file:
        # Load and process data
        llm_df, runtime_df, original_runtime = load_data(llm_scores_file, runtime_file)
        correlation_results = compute_correlations(llm_df, runtime_df)
        
        # Display correlation results
        st.header("📊 Correlation Analysis")
        
        # Model selection
        models = list(correlation_results['correlations'].keys())
        selected_model = st.selectbox("Select Model", models)
        
        if selected_model:
            model_results = correlation_results['correlations'][selected_model]
            
            # Create tabs for different views
            global_tab, construct_tab = st.tabs(["Global Analysis", "Construct-Level Analysis"])
            
            with global_tab:
                st.subheader("Global Correlations")
                if 'overall' in model_results:
                    overall = model_results['overall']
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Pearson Correlation", f"{overall['pearson_correlation']:.3f}")
                        st.metric("p-value", f"{overall['pearson_p_value']:.3f}")
                    
                    with col2:
                        st.metric("Spearman Correlation", f"{overall['spearman_correlation']:.3f}")
                        st.metric("p-value", f"{overall['spearman_p_value']:.3f}")
            
            with construct_tab:
                st.subheader("Construct-Level Correlations")
                for construct, results in model_results.items():
                    if construct != 'overall':
                        st.write(f"**Construct: {construct}**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Pearson Correlation", f"{results['pearson_correlation']:.3f}")
                            st.metric("p-value", f"{results['pearson_p_value']:.3f}")
                        
                        with col2:
                            st.metric("Spearman Correlation", f"{results['spearman_correlation']:.3f}")
                            st.metric("p-value", f"{results['spearman_p_value']:.3f}")
            
            # Perform Bayesian regression
            model_data = pd.DataFrame(correlation_results['raw_data'])
            model_data = model_data[model_data['model'] == selected_model]
            
            st.header("📈 Bayesian Regression Analysis")
            
            with st.spinner("Running Bayesian regression..."):
                bayes_results = perform_bayesian_regression(model_data)
            
            # Display global results
            st.subheader("Global Effects")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Global Slope",
                    f"{bayes_results['global']['slope_mean']:.3f} ± {bayes_results['global']['slope_std']:.3f}"
                )
            
            with col2:
                st.metric(
                    "P(Improvement)",
                    f"{bayes_results['global']['prob_negative_slope']:.1%}"
                )
            
            # Display construct-specific results
            st.subheader("Construct-Specific Effects")
            for construct in model_data['construct_id'].unique():
                if construct in bayes_results:
                    results = bayes_results[construct]
                    st.write(f"**Construct: {construct}**")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Slope",
                            f"{results['slope_mean']:.3f} ± {results['slope_std']:.3f}"
                        )
                    
                    with col2:
                        st.metric(
                            "P(Improvement)",
                            f"{results['prob_negative_slope']:.1%}"
                        )
            
            # Plot regression results
            st.plotly_chart(
                plot_bayesian_regression(model_data, selected_model, bayes_results),
                use_container_width=True
            )
            
            # Display raw data table
            st.header("📋 Raw Data")
            st.dataframe(model_data)

if __name__ == "__main__":
    main() 