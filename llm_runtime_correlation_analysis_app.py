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
    """Compute correlations between LLM scores and runtime changes."""
    print("\nComputing correlations...")
    
    # Filter for only Performance scores
    llm_df = llm_df[llm_df['task'] == 'Performance']
    print(f"\nFiltered for Performance scores only. Number of scores: {len(llm_df)}")
    
    # Filter runtime data for solutions with only 1 spec
    runtime_df = runtime_df[runtime_df['num_specs_in_solution'] == 1]
    print(f"\nFiltered for solutions with 1 spec only. Number of solutions: {len(runtime_df)}")
    
    # Prepare data for correlation analysis
    correlation_data = []
    
    # Group by spec_id to match LLM scores with runtime measurements
    for spec_id in llm_df['spec_id'].unique():
        llm_scores = llm_df[llm_df['spec_id'] == spec_id]
        runtime_data = runtime_df[runtime_df['spec_id'] == spec_id]
        
        if len(runtime_data) > 0:
            # Add correlation data for each model
            for _, row in llm_scores.iterrows():
                correlation_data.append({
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
    print(f"Number of unique specs: {len(corr_df['spec_id'].unique())}")
    print(f"Number of total correlation points: {len(corr_df)}")
    print("\nRuntime change statistics:")
    print(corr_df['runtime_change'].describe())
    print("\nLLM score statistics:")
    print(corr_df['llm_score'].describe())
    
    # Compute correlations for each model
    results = {}
    for model in corr_df['model'].unique():
        model_data = corr_df[corr_df['model'] == model]
        if len(model_data) >= 2:  # Need at least 2 points for correlation
            # Correlation with absolute change
            pearson_r, pearson_p = stats.pearsonr(model_data['llm_score'], model_data['runtime_change'])
            spearman_r, spearman_p = stats.spearmanr(model_data['llm_score'], model_data['runtime_change'])
            
            # Correlation with percentage change
            pearson_r_pct, pearson_p_pct = stats.pearsonr(model_data['llm_score'], model_data['runtime_change_percent'])
            spearman_r_pct, spearman_p_pct = stats.spearmanr(model_data['llm_score'], model_data['runtime_change_percent'])
            
            results[model] = {
                'pearson_correlation': pearson_r,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_r,
                'spearman_p_value': spearman_p,
                'pearson_correlation_pct': pearson_r_pct,
                'pearson_p_value_pct': pearson_p_pct,
                'spearman_correlation_pct': spearman_r_pct,
                'spearman_p_value_pct': spearman_p_pct,
                'n_samples': len(model_data)
            }
    
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
    Perform Bayesian linear regression to predict runtime change from LLM scores.
    
    Args:
        model_data: DataFrame containing 'llm_score' and 'runtime_change' columns
    
    Returns:
        dict containing model results and predictions
    """
    # Standardize the data
    X = (model_data['llm_score'] - model_data['llm_score'].mean()) / model_data['llm_score'].std()
    y = model_data['runtime_change']
    
    # Build and run the model
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        slope = pm.Normal('slope', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=10)
        
        # Linear model
        mu = intercept + slope * X
        
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y)
        
        # Sample from the posterior
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)
    
    # Extract posterior samples
    slope_samples = trace.posterior['slope'].values.flatten()
    intercept_samples = trace.posterior['intercept'].values.flatten()
    
    # Generate predictions across the range of X
    X_new = np.linspace(X.min(), X.max(), 100)
    y_pred = np.zeros((len(slope_samples), len(X_new)))
    
    for i in range(len(slope_samples)):
        y_pred[i] = intercept_samples[i] + slope_samples[i] * X_new
    
    # Calculate mean and credible intervals
    y_mean = y_pred.mean(axis=0)
    y_lower = np.percentile(y_pred, 2.5, axis=0)
    y_upper = np.percentile(y_pred, 97.5, axis=0)
    
    # Transform X back to original scale
    X_orig = X_new * model_data['llm_score'].std() + model_data['llm_score'].mean()
    
    # Calculate probability of negative slope (improvement)
    prob_negative_slope = (slope_samples < 0).mean()
    
    return {
        'X_pred': X_orig,
        'y_mean': y_mean,
        'y_lower': y_lower,
        'y_upper': y_upper,
        'slope_mean': slope_samples.mean(),
        'slope_std': slope_samples.std(),
        'prob_negative_slope': prob_negative_slope,
        'trace': trace
    }

def plot_bayesian_regression(data: pd.DataFrame, model: str, bayes_results: dict) -> go.Figure:
    """Create a plot showing Bayesian regression results."""
    model_data = data[data['model'] == model]
    
    fig = go.Figure()
    
    # Add original data points
    fig.add_trace(go.Scatter(
        x=model_data['llm_score'],
        y=model_data['runtime_change'],
        mode='markers',
        name='Observed Data',
        marker=dict(
            size=10,
            color=model_data['runtime_change'],
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title='Runtime Change (ms)')
        )
    ))
    
    # Add regression line and credible intervals
    fig.add_trace(go.Scatter(
        x=bayes_results['X_pred'],
        y=bayes_results['y_mean'],
        mode='lines',
        name='Regression Line',
        line=dict(color='black')
    ))
    
    # Add credible intervals
    fig.add_trace(go.Scatter(
        x=bayes_results['X_pred'],
        y=bayes_results['y_upper'],
        mode='lines',
        name='95% Credible Interval',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=bayes_results['X_pred'],
        y=bayes_results['y_lower'],
        mode='lines',
        name='95% Credible Interval',
        fill='tonexty',
        line=dict(width=0),
        fillcolor='rgba(0,0,0,0.1)'
    ))
    
    # Add reference line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="No Change")
    
    # Update layout
    fig.update_layout(
        title=f"Bayesian Regression: LLM Score vs Runtime Change for {model}<br>" + 
              f"P(Negative Slope) = {bayes_results['prob_negative_slope']:.3f}",
        xaxis_title="LLM Score",
        yaxis_title="Runtime Change (ms)",
        hovermode='closest'
    )
    
    return fig

def main():
    st.set_page_config(page_title="LLM Score vs Runtime Change Analysis", layout="wide")
    
    st.title("🎯 LLM Score vs Runtime Change Analysis")
    st.caption("Note: Negative runtime change means faster execution (better performance)")
    
    # File paths
    llm_scores_file = 'results/llm_scores_BitNet_20250626_151400.csv'
    runtime_file = 'results/evaluation_data_BitNet_20250626_144039.csv'
    
    try:
        # Load data
        llm_df, runtime_df, original_runtime = load_data(llm_scores_file, runtime_file)
        
        # Compute correlations
        results = compute_correlations(llm_df, runtime_df)
        corr_df = pd.DataFrame(results['raw_data'])
        
        # Display original runtime information
        st.header("📊 Baseline Information")
        st.write(f"Original Version Runtime: {original_runtime:.2f} ms")
        
        # Display correlation results and Bayesian analysis
        st.header("🤖 Model-Level Analysis")
        for model in corr_df['model'].unique():
            with st.expander(f"📊 Model: {model}", expanded=True):
                model_data = corr_df[corr_df['model'] == model]
                
                # Regular correlation analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 Frequentist Correlation")
                    corr = results['correlations'][model]
                    st.write(f"Pearson r = {corr['pearson_correlation']:.3f}")
                    st.write(f"p-value = {corr['pearson_p_value']:.3f}")
                    if corr['pearson_p_value'] < 0.05:
                        st.write("✅ Statistically significant")
                    else:
                        st.write("❌ Not statistically significant")
                
                with col2:
                    st.subheader("🔄 Rank Correlation")
                    st.write(f"Spearman ρ = {corr['spearman_correlation']:.3f}")
                    st.write(f"p-value = {corr['spearman_p_value']:.3f}")
                    if corr['spearman_p_value'] < 0.05:
                        st.write("✅ Statistically significant")
                    else:
                        st.write("❌ Not statistically significant")
                
                # Bayesian regression analysis
                st.subheader("🎲 Bayesian Regression Analysis")
                with st.spinner("Running Bayesian regression..."):
                    bayes_results = perform_bayesian_regression(model_data)
                
                # Display Bayesian results
                st.write("Regression Results:")
                st.write(f"- Mean Slope: {bayes_results['slope_mean']:.3f} ± {bayes_results['slope_std']:.3f}")
                st.write(f"- Probability of Negative Slope: {bayes_results['prob_negative_slope']:.3%}")
                st.write(f"- Interpretation: {bayes_results['prob_negative_slope']:.1%} probability that higher LLM scores predict runtime improvements")
                
                # Plot Bayesian regression
                fig = plot_bayesian_regression(corr_df, model, bayes_results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot posterior distributions
                st.subheader("📊 Posterior Distributions")
                col1, col2 = st.columns([1, 1])  # Create two equal columns
                
                # Create figure with original proportions
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                az.plot_posterior(bayes_results['trace'], var_names=['slope'], ax=axes[0])
                az.plot_posterior(bayes_results['trace'], var_names=['intercept'], ax=axes[1])
                axes[0].set_title("Slope Posterior", fontsize=10)
                axes[1].set_title("Intercept Posterior", fontsize=10)
                plt.tight_layout()
                
                # Use columns to control width while maintaining aspect ratio
                with col1:
                    st.pyplot(fig, use_container_width=False)
                plt.close(fig)  # Clean up the figure
                
                # Add some space after the plots
                st.write("")
                
                # Continue with existing group analysis...
                st.subheader("⚡ Runtime Change by Score Group")
                
                # Create custom bins based on score distribution
                scores = model_data['llm_score']
                if len(scores.unique()) <= 1:
                    st.write("⚠️ All scores are identical - cannot create score groups")
                else:
                    # Create meaningful groups based on score ranges
                    bins = [-float('inf'), -0.5, 0, 0.5, float('inf')]
                    labels = ['Very Negative (≤ -0.5)', 'Negative (-0.5 to 0)', 'Positive (0 to 0.5)', 'Very Positive (> 0.5)']
                    
                    model_data['score_group'] = pd.cut(model_data['llm_score'], bins=bins, labels=labels)
                
                # Calculate statistics for each group
                stats = []
                for group in model_data['score_group'].unique():
                    group_data = model_data[model_data['score_group'] == group]
                    group_size = len(group_data)
                    group_pct = round((group_size / len(model_data) * 100), 1)
                    
                    stats.append({
                        'Score Group': group,
                        'Count': group_size,
                        'Group %': group_pct,
                        'Mean Change (ms)': round(group_data['runtime_change'].mean(), 3),
                        'Std Dev (ms)': round(group_data['runtime_change'].std(), 3),
                        'Mean Change (%)': round(group_data['runtime_change_percent'].mean(), 3),
                        'Std Dev (%)': round(group_data['runtime_change_percent'].std(), 3)
                    })
                
                # Convert to DataFrame and set index
                score_groups = pd.DataFrame(stats).set_index('Score Group')
                
                st.table(score_groups)
        
        # Display distribution plots
        st.header("📈 Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏱️ Runtime Change Distribution")
            fig = px.histogram(
                corr_df,
                x='runtime_change',
                title='Runtime Change Distribution<br>(Negative = faster = better)',
                labels={'runtime_change': 'Runtime Change (ms)'},
                marginal='box'
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Change")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 LLM Score Distribution")
            fig = px.histogram(
                corr_df,
                x='llm_score',
                title='LLM Score Distribution<br>(Higher = better)',
                labels={'llm_score': 'LLM Score'},
                marginal='box'
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="No Change")
            st.plotly_chart(fig, use_container_width=True)
        
        # Add score agreement analysis
        st.header("🤝 Model Agreement Analysis")
        
        # Calculate agreement matrix
        models = sorted(corr_df['model'].unique())
        agreement_matrix = pd.DataFrame(index=models, columns=models)
        
        for m1 in models:
            for m2 in models:
                scores1 = corr_df[corr_df['model'] == m1].set_index('spec_id')['llm_score']
                scores2 = corr_df[corr_df['model'] == m2].set_index('spec_id')['llm_score']
                correlation = scores1.corr(scores2)
                agreement_matrix.loc[m1, m2] = correlation
        
        # Plot agreement heatmap
        fig = px.imshow(
            agreement_matrix,
            title="🔄 Model Score Agreement (Correlation)",
            labels=dict(x="Model", y="Model", color="Correlation"),
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main() 