"""
Altmetric News Analysis Dashboard Page

This page provides interactive visualizations for style features comparison across news sources.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

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

st.title("📝 Altmetric News Analysis Dashboard")

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
default_csv = os.path.join(datasets_dir, "style_features_data.csv")

csv_path = st.sidebar.text_input("Dataset Path", value=default_csv)

# Auto-load option
auto_load = st.sidebar.checkbox("Auto-load on startup", value=True, 
                               help="Automatically load data when page opens. Uncheck to manually load.")

# Cache management buttons
st.sidebar.markdown("---")
with st.sidebar.expander("Cache Management", expanded=False):
    if st.button("Clear Cache for Current File"):
        clear_cache(csv_path)
        st.success("Cache cleared!")
        st.session_state['df_results'] = None  # Force reload
    if st.button("Clear All Caches"):
        clear_cache()
        st.success("All caches cleared!")
        st.session_state['df_results'] = None  # Force reload

# Cache status indicator
st.sidebar.markdown("---")
cache_status = st.sidebar.empty()

# Load data automatically or manually
should_load = False
load_source = None  # 'cache', 'source', or None

if auto_load:
    # Auto-load mode
    if os.path.exists(csv_path):
        cache_path = get_cache_path(csv_path)
        metadata_path = get_cache_metadata_path(csv_path)
        
        if is_cache_valid(csv_path, cache_path, metadata_path):
            # Valid cache exists, load from cache
            should_load = True
            load_source = 'cache'
        else:
            # No valid cache, load from source
            should_load = True
            load_source = 'source'
    else:
        # File doesn't exist - still try to load to show error message
        should_load = True
        load_source = 'source'
else:
    # Manual load mode
    if st.sidebar.button("Load Data"):
        should_load = True
        load_source = 'source'

# Load data
if should_load:
    try:
        if load_source == 'cache':
            # Try to load from cache silently (no spinner for instant load)
            df_results = load_cache(csv_path)
            if df_results is not None:
                st.session_state['df_results'] = df_results
                st.session_state['load_source'] = 'cache'
                cache_status.success("✓ Loaded cached results")
                # Only show success message if not auto-loading (to avoid cluttering UI)
                if not auto_load:
                    st.success(f"Loaded {len(df_results):,} records from cache")
            else:
                # Cache load failed, fallback to source
                load_source = 'source'
        
        if load_source == 'source':
            with st.spinner("Loading data from source file..."):
                df_results = load_data_from_source(csv_path)
                st.session_state['df_results'] = df_results
                st.session_state['load_source'] = 'source'
                cache_status.info("⚙ Loaded from source & cached")
                st.success(f"Data loaded successfully! Loaded {len(df_results):,} records and cached")
                
    except FileNotFoundError:
        st.error(f"File not found: {csv_path}")
        st.session_state['df_results'] = None
        cache_status.error("✗ File not found")
    except pd.errors.EmptyDataError:
        st.error(f"File is empty: {csv_path}")
        st.session_state['df_results'] = None
        cache_status.error("✗ File is empty")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.session_state['df_results'] = None
        cache_status.error(f"✗ Error: {str(e)[:30]}...")
else:
    # Check if data is already loaded in session state
    if 'df_results' not in st.session_state:
        st.info("💡 Enable 'Auto-load on startup' or click 'Load Data' to begin.")
        cache_status.info("⏸ Waiting for data...")

# Display visualizations if results are available
if 'df_results' in st.session_state and st.session_state['df_results'] is not None:
    df_results = st.session_state['df_results']
    
    # Ensure source column exists
    if 'source' not in df_results.columns:
        st.error("Data must contain a 'source' column")
        st.stop()
    
    # Create tabs for Style Features Analysis
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Sentiment Analysis", "Score Distributions", "Data Table"
    ])
    
    with tab1:
        st.header("Altmetric News Analysis Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df_results):,}")
        if 'source' in df_results.columns:
            col2.metric("Unique Sources", f"{df_results['source'].nunique()}")
        if 'sentiment_positive' in df_results.columns:
            col3.metric("Avg Positive Sentiment", f"{df_results['sentiment_positive'].mean():.3f}")
        if 'subjectivity_score' in df_results.columns:
            col4.metric("Avg Subjectivity", f"{df_results['subjectivity_score'].mean():.2f}")
        
        # Source distribution
        if 'source' in df_results.columns:
            st.subheader("Records by News Source")
            source_counts = df_results['source'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_source_pie = px.pie(
                    values=source_counts.values,
                    names=source_counts.index,
                    title="Distribution of Records by Source",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_source_pie, use_container_width=True)
            
            with col2:
                fig_source_bar = px.bar(
                    x=source_counts.index,
                    y=source_counts.values,
                    title="Record Count by Source",
                    labels={'x': 'Source', 'y': 'Count'},
                    color=source_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_source_bar, use_container_width=True)
            
            # Display summary statistics table
            st.subheader("Summary Statistics by Source")
            if all(col in df_results.columns for col in ['sentiment_positive', 'sentiment_neutral', 
                                                          'sentiment_negative', 'sensationality_score', 
                                                          'subjectivity_score']):
                summary_stats = df_results.groupby('source').agg({
                    'sentiment_positive': ['mean', 'std'],
                    'sentiment_neutral': ['mean', 'std'],
                    'sentiment_negative': ['mean', 'std'],
                    'sensationality_score': ['mean', 'std'],
                    'subjectivity_score': ['mean', 'std']
                }).round(3)
                st.dataframe(summary_stats, use_container_width=True)
    
    with tab2:
        st.header("Style Feature Comparison Across News Sources")
        
        # Create 2x2 grid layout matching the image
        col1, col2 = st.columns(2)
        
        # Top-left: Average Sentiment Scores
        with col1:
            if all(col in df_results.columns for col in ['sentiment_positive', 'sentiment_neutral', 
                                                          'sentiment_negative', 'source']):
                # Calculate average sentiment scores by source
                avg_sentiment = df_results.groupby('source').agg({
                    'sentiment_positive': 'mean',
                    'sentiment_neutral': 'mean',
                    'sentiment_negative': 'mean'
                }).reset_index()
                
                # Melt for grouped bar chart
                avg_sentiment_melted = avg_sentiment.melt(
                    id_vars='source',
                    value_vars=['sentiment_positive', 'sentiment_neutral', 'sentiment_negative'],
                    var_name='sentiment_type',
                    value_name='score'
                )
                
                # Map sentiment types to readable names
                avg_sentiment_melted['sentiment_type'] = avg_sentiment_melted['sentiment_type'].map({
                    'sentiment_positive': 'Positive',
                    'sentiment_neutral': 'Neutral',
                    'sentiment_negative': 'Negative'
                })
                
                # Create grouped bar chart
                fig_sentiment = px.bar(
                    avg_sentiment_melted,
                    x='source',
                    y='score',
                    color='sentiment_type',
                    barmode='group',
                    title="Average Sentiment Scores",
                    labels={'source': 'source', 'score': 'Score', 'sentiment_type': 'Sentiment Type'},
                    color_discrete_map={
                        'Positive': '#2ecc71',  # Green
                        'Neutral': '#95a5a6',   # Gray
                        'Negative': '#e74c3c'   # Red
                    }
                )
                fig_sentiment.update_layout(
                    xaxis={'categoryorder': 'array', 'categoryarray': ['JunkScience', 'Original', 'Retraction']},
                    legend=dict(title="Sentiment Type"),
                    height=400
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Top-right: Sensationality Score Distribution
        with col2:
            if 'sensationality_score' in df_results.columns and 'source' in df_results.columns:
                fig_sensationality = px.box(
                    df_results,
                    x='source',
                    y='sensationality_score',
                    title="Sensationality Score Distribution",
                    labels={'source': 'source', 'sensationality_score': 'sensationality_score'},
                    points="outliers"
                )
                fig_sensationality.update_layout(
                    xaxis={'categoryorder': 'array', 'categoryarray': ['Original', 'Retraction', 'JunkScience']},
                    height=400
                )
                st.plotly_chart(fig_sensationality, use_container_width=True)
        
        # Bottom-left: Subjectivity Score Distribution
        col3, col4 = st.columns(2)
        with col3:
            if 'subjectivity_score' in df_results.columns and 'source' in df_results.columns:
                fig_subjectivity = px.box(
                    df_results,
                    x='source',
                    y='subjectivity_score',
                    title="Subjectivity Score Distribution",
                    labels={'source': 'source', 'subjectivity_score': 'subjectivity_score'},
                    points="outliers"
                )
                fig_subjectivity.update_layout(
                    xaxis={'categoryorder': 'array', 'categoryarray': ['Original', 'Retraction', 'JunkScience']},
                    height=400
                )
                st.plotly_chart(fig_subjectivity, use_container_width=True)
        
        # Bottom-right: Retraction Language Frequency
        with col4:
            if 'has_retraction_language' in df_results.columns and 'source' in df_results.columns:
                retraction_counts = df_results.groupby('source')['has_retraction_language'].sum().reset_index()
                retraction_counts.columns = ['source', 'count']
                
                fig_retraction = px.bar(
                    retraction_counts,
                    x='source',
                    y='count',
                    title="Retraction Language Frequency",
                    labels={'source': 'source', 'count': 'Count'},
                    color='source',
                    color_discrete_sequence=['#e74c3c', '#3498db', '#2ecc71']
                )
                fig_retraction.update_layout(
                    xaxis={'categoryorder': 'array', 'categoryarray': ['JunkScience', 'Original', 'Retraction']},
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_retraction, use_container_width=True)
        
        # Additional detailed analysis below the main grid
        st.markdown("---")
        st.subheader("Detailed Sentiment Analysis")
        
        if all(col in df_results.columns for col in ['sentiment_positive', 'sentiment_neutral', 
                                                      'sentiment_negative', 'source']):
            # Additional sentiment distribution visualization
            col_detail1, col_detail2, col_detail3 = st.columns(3)
            
            with col_detail1:
                fig_pos = px.box(
                    df_results,
                    x='source',
                    y='sentiment_positive',
                    title="Positive Sentiment Distribution",
                    labels={'source': 'Source', 'sentiment_positive': 'Positive Sentiment Score'}
                )
                st.plotly_chart(fig_pos, use_container_width=True)
            
            with col_detail2:
                fig_neu = px.box(
                    df_results,
                    x='source',
                    y='sentiment_neutral',
                    title="Neutral Sentiment Distribution",
                    labels={'source': 'Source', 'sentiment_neutral': 'Neutral Sentiment Score'}
                )
                st.plotly_chart(fig_neu, use_container_width=True)
            
            with col_detail3:
                fig_neg = px.box(
                    df_results,
                    x='source',
                    y='sentiment_negative',
                    title="Negative Sentiment Distribution",
                    labels={'source': 'Source', 'sentiment_negative': 'Negative Sentiment Score'}
                )
                st.plotly_chart(fig_neg, use_container_width=True)
    
    with tab3:
        st.header("Score Distributions")
        
        if 'sensationality_score' in df_results.columns and 'source' in df_results.columns:
            # Sensationality Score Distribution
            st.subheader("Sensationality Score Distribution")
            fig_sensationality = px.box(
                df_results,
                x='source',
                y='sensationality_score',
                title="Sensationality Score Distribution by Source",
                labels={'source': 'Source', 'sensationality_score': 'Sensationality Score'},
                points="outliers"
            )
            fig_sensationality.update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': ['Original', 'Retraction', 'JunkScience']}
            )
            st.plotly_chart(fig_sensationality, use_container_width=True)
            
            # Sensationality statistics
            if st.expander("View Sensationality Statistics", expanded=False):
                sens_stats = df_results.groupby('source')['sensationality_score'].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median'
                ]).round(3)
                st.dataframe(sens_stats, use_container_width=True)
        
        if 'subjectivity_score' in df_results.columns and 'source' in df_results.columns:
            # Subjectivity Score Distribution
            st.subheader("Subjectivity Score Distribution")
            fig_subjectivity = px.box(
                df_results,
                x='source',
                y='subjectivity_score',
                title="Subjectivity Score Distribution by Source",
                labels={'source': 'Source', 'subjectivity_score': 'Subjectivity Score'},
                points="outliers"
            )
            fig_subjectivity.update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': ['Original', 'Retraction', 'JunkScience']}
            )
            st.plotly_chart(fig_subjectivity, use_container_width=True)
            
            # Subjectivity statistics
            if st.expander("View Subjectivity Statistics", expanded=False):
                subj_stats = df_results.groupby('source')['subjectivity_score'].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median'
                ]).round(3)
                st.dataframe(subj_stats, use_container_width=True)
        
        # Combined comparison
        if all(col in df_results.columns for col in ['sensationality_score', 'subjectivity_score', 'source']):
            st.subheader("Sensationality vs Subjectivity")
            fig_scatter = px.scatter(
                df_results.sample(min(10000, len(df_results))),  # Sample for performance
                x='sensationality_score',
                y='subjectivity_score',
                color='source',
                title="Sensationality vs Subjectivity Score",
                labels={
                    'sensationality_score': 'Sensationality Score',
                    'subjectivity_score': 'Subjectivity Score'
                },
                opacity=0.6,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Distribution histograms
        st.subheader("Score Distribution Histograms")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'sensationality_score' in df_results.columns and 'source' in df_results.columns:
                fig_hist_sens = px.histogram(
                    df_results,
                    x='sensationality_score',
                    color='source',
                    nbins=50,
                    title="Sensationality Score Histogram",
                    labels={'sensationality_score': 'Sensationality Score', 'count': 'Frequency'},
                    barmode='overlay',
                    opacity=0.7
                )
                st.plotly_chart(fig_hist_sens, use_container_width=True)
        
        with col2:
            if 'subjectivity_score' in df_results.columns and 'source' in df_results.columns:
                fig_hist_subj = px.histogram(
                    df_results,
                    x='subjectivity_score',
                    color='source',
                    nbins=50,
                    title="Subjectivity Score Histogram",
                    labels={'subjectivity_score': 'Subjectivity Score', 'count': 'Frequency'},
                    barmode='overlay',
                    opacity=0.7
                )
                st.plotly_chart(fig_hist_subj, use_container_width=True)
    
    with tab4:
        st.header("Data Table")
        
        # Show current dataset
        st.subheader("Current Dataset")
        st.dataframe(df_results, use_container_width=True, height=400)
        
        # Download option
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="style_features_analysis.csv",
            mime="text/csv"
        )
        
        # Filter options
        st.subheader("Filter Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'source' in df_results.columns:
                selected_sources = st.multiselect(
                    "Filter by Source",
                    options=df_results['source'].unique().tolist(),
                    default=df_results['source'].unique().tolist()
                )
            else:
                selected_sources = []
        
        with col2:
            if 'has_retraction_language' in df_results.columns:
                retraction_filter = st.selectbox(
                    "Filter by Retraction Language",
                    options=['All', 'Has Retraction Language', 'No Retraction Language'],
                    index=0
                )
            else:
                retraction_filter = 'All'
        
        # Apply filters
        filtered_df = df_results.copy()
        if selected_sources:
            filtered_df = filtered_df[filtered_df['source'].isin(selected_sources)]
        if retraction_filter == 'Has Retraction Language':
            filtered_df = filtered_df[filtered_df['has_retraction_language'] == True]
        elif retraction_filter == 'No Retraction Language':
            filtered_df = filtered_df[filtered_df['has_retraction_language'] == False]
        
        if len(filtered_df) < len(df_results):
            st.info(f"Showing {len(filtered_df):,} of {len(df_results):,} records")
            st.dataframe(filtered_df, use_container_width=True, height=400)

else:
    st.info("Click 'Load Data' in the sidebar to start analyzing your data.")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Altmetric News Analysis Dashboard")

