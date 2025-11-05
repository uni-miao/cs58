"""
Retraction Consistency Analysis Dashboard Page

This page provides interactive visualizations for retraction consistency analysis data,
comparing before and after retraction patterns across multiple dimensions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import numpy as np
import json

# Add parent directory to path to import cache_utils
current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import cache utilities
from cache_utils import (
    get_cache_path, get_cache_metadata_path, is_cache_valid,
    load_cache, save_cache, clear_cache, load_data_from_source
)

# Page config is set by parent interactive_dashboard.py
# No need to set it here to avoid conflicts

st.title("🔍 Retraction Consistency Analysis Dashboard")

# Sidebar configuration
st.sidebar.header("Analysis Settings")

# Get the directory of datasets - use DATASETS_DIR if provided by parent, otherwise calculate
try:
    # DATASETS_DIR is injected by interactive_dashboard.py when using exec()
    datasets_dir = DATASETS_DIR  # type: ignore
except NameError:
    # Fallback: calculate from __file__ location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # Go up one level from pages/
    datasets_dir = os.path.join(parent_dir, "datasets")

# Define paths to retraction consistency data files
retraction_consistency_dir = os.path.join(datasets_dir, "retraction_consistency")
full_results_path = os.path.join(retraction_consistency_dir, "advanced_consistency_analysis_results_full.csv")
comparative_summary_path = os.path.join(retraction_consistency_dir, "before_after_comparative_summary.csv")
before_analysis_path = os.path.join(retraction_consistency_dir, "before_after_consistency_analysis_BEFORE.csv")
after_analysis_path = os.path.join(retraction_consistency_dir, "before_after_consistency_analysis_AFTER.csv")

# Auto-load option
auto_load = st.sidebar.checkbox("Auto-load on startup", value=True, 
                               help="Automatically load data when page opens. Uncheck to manually load.")

# Load data function with caching
def load_retraction_consistency_data():
    """Load all retraction consistency datasets with caching support"""
    data = {}
    try:
        # Load full results with caching
        if os.path.exists(full_results_path):
            cache_path = get_cache_path(full_results_path)
            metadata_path = get_cache_metadata_path(full_results_path)
            
            if is_cache_valid(full_results_path, cache_path, metadata_path):
                # Load from cache
                cached_df = load_cache(full_results_path)
                if cached_df is not None:
                    data['full_results'] = cached_df
                else:
                    # Fallback to source
                    df = pd.read_csv(full_results_path)
                    data['full_results'] = df
                    save_cache(full_results_path, df)
            else:
                # Load from source and cache
                df = pd.read_csv(full_results_path)
                data['full_results'] = df
                save_cache(full_results_path, df)
        
        # Load comparative summary with caching
        if os.path.exists(comparative_summary_path):
            cache_path = get_cache_path(comparative_summary_path)
            metadata_path = get_cache_metadata_path(comparative_summary_path)
            
            if is_cache_valid(comparative_summary_path, cache_path, metadata_path):
                # Load from cache
                cached_df = load_cache(comparative_summary_path)
                if cached_df is not None:
                    data['comparative_summary'] = cached_df
                else:
                    # Fallback to source
                    df = pd.read_csv(comparative_summary_path)
                    data['comparative_summary'] = df
                    save_cache(comparative_summary_path, df)
            else:
                # Load from source and cache
                df = pd.read_csv(comparative_summary_path)
                data['comparative_summary'] = df
                save_cache(comparative_summary_path, df)
        
        # Load before analysis with caching
        if os.path.exists(before_analysis_path):
            cache_path = get_cache_path(before_analysis_path)
            metadata_path = get_cache_metadata_path(before_analysis_path)
            
            if is_cache_valid(before_analysis_path, cache_path, metadata_path):
                # Load from cache
                cached_df = load_cache(before_analysis_path)
                if cached_df is not None:
                    data['before_analysis'] = cached_df
                else:
                    # Fallback to source
                    df = pd.read_csv(before_analysis_path)
                    data['before_analysis'] = df
                    save_cache(before_analysis_path, df)
            else:
                # Load from source and cache
                df = pd.read_csv(before_analysis_path)
                data['before_analysis'] = df
                save_cache(before_analysis_path, df)
        
        # Load after analysis with caching
        if os.path.exists(after_analysis_path):
            cache_path = get_cache_path(after_analysis_path)
            metadata_path = get_cache_metadata_path(after_analysis_path)
            
            if is_cache_valid(after_analysis_path, cache_path, metadata_path):
                # Load from cache
                cached_df = load_cache(after_analysis_path)
                if cached_df is not None:
                    data['after_analysis'] = cached_df
                else:
                    # Fallback to source
                    df = pd.read_csv(after_analysis_path)
                    data['after_analysis'] = df
                    save_cache(after_analysis_path, df)
            else:
                # Load from source and cache
                df = pd.read_csv(after_analysis_path)
                data['after_analysis'] = df
                save_cache(after_analysis_path, df)
        
        return data
    except Exception as e:
        st.error(f"Error loading retraction consistency data: {str(e)}")
        return {}

# Cache management buttons
st.sidebar.markdown("---")
with st.sidebar.expander("Cache Management", expanded=False):
    if st.button("Clear Cache for Retraction Consistency Data"):
        for path in [full_results_path, comparative_summary_path, before_analysis_path, after_analysis_path]:
            if os.path.exists(path):
                clear_cache(path)
        st.success("Retraction consistency cache cleared!")
        st.session_state['retraction_data'] = None  # Force reload
    if st.button("Clear All Caches"):
        clear_cache()
        st.success("All caches cleared!")
        st.session_state['retraction_data'] = None  # Force reload

# Cache status indicator
st.sidebar.markdown("---")
cache_status = st.sidebar.empty()

# Load data automatically or manually
should_load = False
load_source = None  # 'cache', 'source', or None

if auto_load:
    # Auto-load mode
    should_load = True
    load_source = 'auto'
else:
    # Manual load mode
    if st.sidebar.button("Load Data"):
        should_load = True
        load_source = 'manual'

# Load data
if should_load:
    try:
        with st.spinner("Loading retraction consistency data..."):
            data = load_retraction_consistency_data()
            
        if data:
            st.session_state['retraction_data'] = data
            cache_status.success("✅ Data loaded successfully")
        else:
            st.error("No data files found in retraction_consistency directory")
            cache_status.error("❌ No data loaded")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        cache_status.error("❌ Error loading data")

# Display data if available
if 'retraction_data' in st.session_state and st.session_state['retraction_data']:
    data = st.session_state['retraction_data']
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Before vs After Comparison", 
        "🔍 Consistency Analysis", 
        "💭 Sentiment Analysis", 
        "📋 Data Explorer"
    ])
    
    with tab1:
        st.header("📊 Retraction Consistency Analysis Overview")
        
        # Display summary statistics if available
        if 'comparative_summary' in data:
            summary_df = data['comparative_summary']
            st.subheader("Before vs After Summary Statistics")
            st.dataframe(summary_df, use_container_width=True)
            
            # Create comparison chart
            if len(summary_df) >= 2:
                metrics = ['Semantic_Notice_vs_Text', 'Semantic_Notice_vs_Title', 'Semantic_Text_vs_Title',
                          'Entity_Overlap_Notice_News', 'Reason_Consistency_Notice_News', 
                          'Sentiment_Consistency_Notice_News']
                
                fig = go.Figure()
                
                # Add bars for BEFORE and AFTER
                before_data = summary_df[summary_df['Category'] == 'BEFORE'].iloc[0] if len(summary_df[summary_df['Category'] == 'BEFORE']) > 0 else None
                after_data = summary_df[summary_df['Category'] == 'AFTER'].iloc[0] if len(summary_df[summary_df['Category'] == 'AFTER']) > 0 else None
                
                if before_data is not None and after_data is not None:
                    before_values = [before_data.get(metric, 0) for metric in metrics]
                    after_values = [after_data.get(metric, 0) for metric in metrics]
                    
                    fig.add_trace(go.Bar(
                        name='Before Retraction',
                        x=metrics,
                        y=before_values,
                        marker_color='lightcoral'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='After Retraction',
                        x=metrics,
                        y=after_values,
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title="Before vs After Retraction: Key Metrics Comparison",
                        xaxis_title="Metrics",
                        yaxis_title="Values",
                        barmode='group',
                        height=500
                    )
                    
                    # Rotate x-axis labels for better readability
                    fig.update_xaxes(tickangle=45)
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Display dataset information
        st.subheader("📋 Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'full_results' in data:
                st.metric("Full Results Records", len(data['full_results']))
        
        with col2:
            if 'before_analysis' in data:
                st.metric("Before Analysis Records", len(data['before_analysis']))
        
        with col3:
            if 'after_analysis' in data:
                st.metric("After Analysis Records", len(data['after_analysis']))
        
        with col4:
            if 'comparative_summary' in data:
                st.metric("Summary Categories", len(data['comparative_summary']))
    
    with tab2:
        st.header("📈 Before vs After Comparison")
        
        if 'before_analysis' in data and 'after_analysis' in data:
            before_df = data['before_analysis']
            after_df = data['after_analysis']
            
            # Select metrics for comparison
            numeric_columns = [col for col in before_df.columns if before_df[col].dtype in ['float64', 'int64']]
            if numeric_columns:
                selected_metric = st.selectbox("Select metric to compare:", numeric_columns)
                
                # Create comparison visualization
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Before Retraction Distribution', 'After Retraction Distribution'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Before distribution
                fig.add_trace(
                    go.Histogram(x=before_df[selected_metric], name='Before', 
                               marker_color='lightcoral', opacity=0.7),
                    row=1, col=1
                )
                
                # After distribution
                fig.add_trace(
                    go.Histogram(x=after_df[selected_metric], name='After', 
                               marker_color='lightblue', opacity=0.7),
                    row=1, col=2
                )
                
                fig.update_layout(
                    title=f"Distribution Comparison: {selected_metric}",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Before Retraction Statistics")
                    st.write(before_df[selected_metric].describe())
                
                with col2:
                    st.subheader("After Retraction Statistics")
                    st.write(after_df[selected_metric].describe())
        else:
            st.info("Before and After analysis data not available for comparison.")
    
    with tab3:
        st.header("🔍 Consistency Analysis")
        
        if 'full_results' in data:
            full_df = data['full_results']
            
            # Consistency metrics
            consistency_metrics = [col for col in full_df.columns if 'consistency' in col.lower()]
            
            if consistency_metrics:
                st.subheader("Consistency Metrics Overview")
                
                # Create correlation heatmap
                consistency_data = full_df[consistency_metrics].corr()
                
                fig = px.imshow(
                    consistency_data,
                    title="Consistency Metrics Correlation Heatmap",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Consistency distribution
                selected_consistency = st.selectbox("Select consistency metric:", consistency_metrics)
                
                fig = px.histogram(
                    full_df, 
                    x=selected_consistency,
                    title=f"Distribution of {selected_consistency}",
                    nbins=30
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Full results data not available for consistency analysis.")
    
    with tab4:
        st.header("💭 Sentiment Analysis")
        
        if 'full_results' in data:
            full_df = data['full_results']
            
            # Sentiment columns
            sentiment_cols = [col for col in full_df.columns if 'sentiment' in col.lower()]
            
            if sentiment_cols:
                st.subheader("Sentiment Distribution Comparison")
                
                # Create sentiment comparison
                fig = make_subplots(
                    rows=len(sentiment_cols), cols=1,
                    subplot_titles=sentiment_cols,
                    vertical_spacing=0.1
                )
                
                for i, col in enumerate(sentiment_cols):
                    fig.add_trace(
                        go.Histogram(x=full_df[col], name=col, opacity=0.7),
                        row=i+1, col=1
                    )
                
                fig.update_layout(
                    title="Sentiment Metrics Distribution",
                    height=200 * len(sentiment_cols),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sentiment correlation
                if len(sentiment_cols) > 1:
                    st.subheader("Sentiment Correlation Analysis")
                    sentiment_corr = full_df[sentiment_cols].corr()
                    
                    fig = px.imshow(
                        sentiment_corr,
                        title="Sentiment Metrics Correlation",
                        color_continuous_scale='RdBu_r'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Full results data not available for sentiment analysis.")
    
    with tab5:
        st.header("📋 Data Explorer")
        
        # Dataset selector
        available_datasets = list(data.keys())
        selected_dataset = st.selectbox("Select dataset to explore:", available_datasets)
        
        if selected_dataset in data:
            df = data[selected_dataset]
            
            st.subheader(f"Dataset: {selected_dataset}")
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Display basic info
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Column Information:**")
                st.write(df.dtypes)
            
            with col2:
                st.write("**Missing Values:**")
                st.write(df.isnull().sum())
            
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(df.head(100), use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label=f"Download {selected_dataset} as CSV",
                data=csv,
                file_name=f"{selected_dataset}.csv",
                mime="text/csv"
            )

else:
    # No data loaded
    st.info("👆 Please load data using the sidebar controls to start analysis.")
    
    st.markdown("""
    ### 🔍 Retraction Consistency Analysis
    
    This dashboard analyzes the consistency patterns in retraction-related content across multiple dimensions:
    
    **📊 Available Analysis:**
    - **Before vs After Comparison**: Compare metrics before and after retraction
    - **Consistency Analysis**: Explore consistency patterns across different text sources
    - **Sentiment Analysis**: Analyze sentiment patterns in retraction notices and news
    - **Data Explorer**: Browse and download the raw datasets
    
    **📁 Expected Data Files:**
    - `advanced_consistency_analysis_results_full.csv`
    - `before_after_comparative_summary.csv`
    - `before_after_consistency_analysis_BEFORE.csv`
    - `before_after_consistency_analysis_AFTER.csv`
    
    **🚀 Getting Started:**
    1. Ensure your data files are in the `datasets/retraction_consistency/` directory
    2. Enable "Auto-load on startup" or click "Load Data" in the sidebar
    3. Explore the different analysis tabs
    """)

# Add footer information
st.sidebar.markdown("---")
st.sidebar.markdown("**Retraction Consistency Analysis**")
st.sidebar.markdown("Analyzing consistency patterns in retraction-related content")
