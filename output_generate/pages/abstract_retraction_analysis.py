"""
Abstract & Retraction Notice Analysis Dashboard Page

This page provides interactive visualizations for two-layer text analysis comparison 
between Abstract and Retraction Notice texts.
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

st.title("📊 Abstract & Retraction Notice Analysis Dashboard")

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

# Define paths to abstract_retractionNotice data files
abstract_retraction_dir = os.path.join(datasets_dir, "abstract_retractionNotice")
abstract_features_path = os.path.join(abstract_retraction_dir, "abstract_features.csv")
retraction_features_path = os.path.join(abstract_retraction_dir, "retraction_features.csv")
difference_stats_path = os.path.join(abstract_retraction_dir, "difference_statistics.json")

# Auto-load option
auto_load = st.sidebar.checkbox("Auto-load on startup", value=True, 
                               help="Automatically load data when page opens. Uncheck to manually load.")

# Load data function with caching
def load_abstract_retraction_data():
    """Load all abstract and retraction analysis datasets with caching support"""
    data = {}
    try:
        # Load abstract features with caching
        if os.path.exists(abstract_features_path):
            cache_path = get_cache_path(abstract_features_path)
            metadata_path = get_cache_metadata_path(abstract_features_path)
            
            if is_cache_valid(abstract_features_path, cache_path, metadata_path):
                # Load from cache
                cached_df = load_cache(abstract_features_path)
                if cached_df is not None:
                    data['abstract'] = cached_df
                else:
                    # Fallback to source
                    df = pd.read_csv(abstract_features_path)
                    data['abstract'] = df
                    save_cache(abstract_features_path, df)
            else:
                # Load from source and cache
                df = pd.read_csv(abstract_features_path)
                data['abstract'] = df
                save_cache(abstract_features_path, df)
        
        # Load retraction features with caching
        if os.path.exists(retraction_features_path):
            cache_path = get_cache_path(retraction_features_path)
            metadata_path = get_cache_metadata_path(retraction_features_path)
            
            if is_cache_valid(retraction_features_path, cache_path, metadata_path):
                # Load from cache
                cached_df = load_cache(retraction_features_path)
                if cached_df is not None:
                    data['retraction'] = cached_df
                else:
                    # Fallback to source
                    df = pd.read_csv(retraction_features_path)
                    data['retraction'] = df
                    save_cache(retraction_features_path, df)
            else:
                # Load from source and cache
                df = pd.read_csv(retraction_features_path)
                data['retraction'] = df
                save_cache(retraction_features_path, df)
        
        # Load difference statistics (JSON file - no caching needed, small file)
        if os.path.exists(difference_stats_path):
            with open(difference_stats_path, 'r', encoding='utf-8') as f:
                data['differences'] = json.load(f)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Cache management buttons
st.sidebar.markdown("---")
with st.sidebar.expander("Cache Management", expanded=False):
    if st.button("Clear Cache for Abstract & Retraction"):
        # Clear cache for abstract and retraction files
        if os.path.exists(abstract_features_path):
            clear_cache(abstract_features_path)
        if os.path.exists(retraction_features_path):
            clear_cache(retraction_features_path)
        st.success("Cache cleared!")
        st.session_state['abstract_retraction_data'] = None
        st.rerun()
    if st.button("Clear All Caches"):
        clear_cache()
        st.success("All caches cleared!")
        st.session_state['abstract_retraction_data'] = None
        st.rerun()

# Cache status indicator
st.sidebar.markdown("---")
cache_status = st.sidebar.empty()

# Load data automatically or manually
should_load = False
load_source = None  # 'cache', 'source', or None

if auto_load:
    # Auto-load mode - check if we can load from cache
    if os.path.exists(abstract_features_path) and os.path.exists(retraction_features_path):
        # Check if both files have valid cache
        abstract_cache_path = get_cache_path(abstract_features_path)
        abstract_metadata_path = get_cache_metadata_path(abstract_features_path)
        retraction_cache_path = get_cache_path(retraction_features_path)
        retraction_metadata_path = get_cache_metadata_path(retraction_features_path)
        
        abstract_cached = is_cache_valid(abstract_features_path, abstract_cache_path, abstract_metadata_path)
        retraction_cached = is_cache_valid(retraction_features_path, retraction_cache_path, retraction_metadata_path)
        
        if abstract_cached and retraction_cached:
            should_load = True
            load_source = 'cache'
        else:
            should_load = True
            load_source = 'source'
    else:
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
            data = load_abstract_retraction_data()
            if data:
                st.session_state['abstract_retraction_data'] = data
                cache_status.success("✓ Loaded from cache")
            else:
                load_source = 'source'
        
        if load_source == 'source':
            with st.spinner("Loading abstract and retraction analysis data..."):
                data = load_abstract_retraction_data()
                if data:
                    st.session_state['abstract_retraction_data'] = data
                    cache_status.info("⚙ Loaded from source & cached")
                    st.success("Data loaded successfully and cached!")
                else:
                    st.error("Failed to load data. Please check if data files exist in the abstract_retractionNotice folder.")
                    cache_status.error("✗ Load failed")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        cache_status.error("✗ Error")
else:
    if 'abstract_retraction_data' not in st.session_state:
        cache_status.info("⏸ Waiting for data...")

# Display visualizations if data is available
if 'abstract_retraction_data' in st.session_state and st.session_state['abstract_retraction_data']:
    data = st.session_state['abstract_retraction_data']
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Abstract Analysis", "Retraction Analysis", "Comparison"
    ])
    
    with tab1:
        st.header("Analysis Overview")
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'abstract' in data:
                abstract_df = data['abstract']
                st.metric("Abstract Records", f"{len(abstract_df):,}")
                st.metric("Avg Sentence Length", f"{abstract_df['avg_sentence_length'].mean():.2f}")
                st.metric("Avg Sentiment Score", f"{abstract_df['sentiment_score'].mean():.3f}")
        
        with col2:
            if 'retraction' in data:
                retraction_df = data['retraction']
                st.metric("Retraction Records", f"{len(retraction_df):,}")
                st.metric("Avg Sentence Length", f"{retraction_df['avg_sentence_length'].mean():.2f}")
                st.metric("Avg Sentiment Score", f"{retraction_df['sentiment_score'].mean():.3f}")
        
        with col3:
            if 'differences' in data:
                diff_stats = data['differences']
                st.metric("Shared Metrics", len(diff_stats))
                # Find most significant difference
                max_t_stat = max([abs(v.get('t_statistic', 0)) for v in diff_stats.values()])
                st.metric("Max |T-Statistic|", f"{max_t_stat:.2f}")
        
        # Key findings
        st.subheader("Key Findings")
        st.markdown("""
        This analysis implements a **two-layer text analysis system** for comparing abstracts and retraction notices:
        
        ### Layer 1: Shared Dimensions (Comparable)
        - **Language Complexity**: Sentence length, word length, vocabulary diversity, POS ratios
        - **Readability**: Flesch Reading Ease, Flesch-Kincaid Grade Level, SMOG Index
        - **Tone & Attitude**: Assertive words, hedging words, negation, sentiment scores
        - **Topic & Keywords**: Top keywords, named entities, information density
        
        ### Layer 2: Specific Dimensions
        - **Abstract-specific**: Scientific terms, statistical evidence, structure roles, innovation expressions
        - **Retraction-specific**: Retraction reasons, responsibility subjects, template similarity, investigation terms
        
        ### Analysis Features
        - ✅ Same preprocessing pipeline for both text types
        - ✅ Identical metrics and field names for direct comparison
        - ✅ Statistical significance testing (t-statistics)
        - ✅ Comprehensive visualization suite
        """)
    
    with tab2:
        st.header("Abstract Individual Analysis")
        
        if 'abstract' in data:
            abstract_df = data['abstract']
            
            # Sub-tabs for abstract analysis
            sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
                "Language Complexity", "Readability", "Tone & Attitude", 
                "Structure Roles", "Scientific Features"
            ])
            
            with sub_tab1:
                st.subheader("Language Complexity Distribution")
                
                # Select metrics for complexity
                complexity_metrics = [
                    'avg_sentence_length', 'avg_word_length', 
                    'type_token_ratio', 'passive_voice_ratio'
                ]
                
                # Create box plots
                fig = make_subplots(rows=2, cols=2, 
                                  subplot_titles=[m.replace('_', ' ').title() for m in complexity_metrics])
                
                for idx, metric in enumerate(complexity_metrics):
                    row = (idx // 2) + 1
                    col = (idx % 2) + 1
                    
                    fig.add_trace(
                        go.Box(y=abstract_df[metric].dropna(), name=metric.replace('_', ' ').title(),
                              showlegend=False),
                        row=row, col=col
                    )
                
                fig.update_layout(height=600, showlegend=False,
                                title_text="Language Complexity Distribution in Abstracts")
                st.plotly_chart(fig, use_container_width=True)
            
            with sub_tab2:
                st.subheader("Readability Metrics Distribution")
                
                readability_metrics = [
                    'flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index'
                ]
                
                fig = make_subplots(rows=1, cols=3,
                                  subplot_titles=['Flesch Reading Ease', 'Flesch-Kincaid Grade', 'SMOG Index'])
                
                for idx, metric in enumerate(readability_metrics):
                    data_series = abstract_df[metric].dropna()
                    fig.add_trace(
                        go.Histogram(x=data_series, name=metric.replace('_', ' ').title(),
                                   nbinsx=50, showlegend=False),
                        row=1, col=idx+1
                    )
                
                fig.update_layout(height=400, showlegend=False,
                                title_text="Readability Metrics Distribution in Abstracts")
                st.plotly_chart(fig, use_container_width=True)
            
            with sub_tab3:
                st.subheader("Tone and Attitude Scatter Plot")
                
                # Create scatter plot
                fig = px.scatter(
                    abstract_df,
                    x='assertive_word_count',
                    y='hedging_word_count',
                    color='sentiment_score',
                    color_continuous_scale='RdYlGn',
                    title='Tone and Attitude in Abstracts (Colored by Sentiment Score)',
                    labels={
                        'assertive_word_count': 'Assertive Word Count',
                        'hedging_word_count': 'Hedging Word Count',
                        'sentiment_score': 'Sentiment Score'
                    },
                    hover_data=['record_id']
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with sub_tab4:
                st.subheader("Information Structure Roles")
                
                # Check if structure columns exist
                structure_cols = [
                    'structure_background_ratio', 'structure_method_ratio',
                    'structure_result_ratio', 'structure_conclusion_ratio'
                ]
                
                existing_cols = [col for col in structure_cols if col in abstract_df.columns]
                
                if existing_cols:
                    # Calculate average ratios
                    avg_ratios = [abstract_df[col].mean() for col in existing_cols]
                    labels = [col.replace('structure_', '').replace('_ratio', '').title() 
                             for col in existing_cols]
                    
                    fig = px.bar(
                        x=labels,
                        y=avg_ratios,
                        title='Average Information Structure Roles in Abstracts',
                        labels={'x': 'Structure Role', 'y': 'Average Ratio'},
                        color=avg_ratios,
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=500, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Structure role columns not found in the dataset.")
            
            with sub_tab5:
                st.subheader("Scientific and Innovation Features")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Scatter plot: Scientific terms vs Statistical evidence
                    fig_scatter = px.scatter(
                        abstract_df,
                        x='scientific_terms_count',
                        y='statistical_evidence_count',
                        title='Scientific Terms vs Statistical Evidence',
                        labels={
                            'scientific_terms_count': 'Scientific Terms Count',
                            'statistical_evidence_count': 'Statistical Evidence Count'
                        }
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # Histogram: Innovation expressions
                    fig_hist = px.histogram(
                        abstract_df,
                        x='innovation_expression',
                        title='Innovation Expression Distribution',
                        labels={'innovation_expression': 'Innovation Expression Count'},
                        nbins=20
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("Abstract data not available.")
    
    with tab3:
        st.header("Retraction Individual Analysis")
        
        if 'retraction' in data:
            retraction_df = data['retraction']
            
            # Sub-tabs for retraction analysis
            sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
                "Retraction Reasons", "Responsibility", "Template & Investigation", "Tone Strength"
            ])
            
            with sub_tab1:
                st.subheader("Retraction Reason Distribution")
                
                # Get reason columns
                reason_cols = [
                    'reason_data', 'reason_statistical', 'reason_ethical',
                    'reason_plagiarism', 'reason_authorship', 'reason_image',
                    'reason_reproducibility'
                ]
                
                # Calculate percentages
                reason_counts = {}
                for col in reason_cols:
                    if col in retraction_df.columns:
                        # Count True values
                        if retraction_df[col].dtype == bool:
                            count = retraction_df[col].sum()
                        elif retraction_df[col].dtype == object:
                            count = (retraction_df[col] == True).sum() + \
                                   (retraction_df[col] == 'True').sum() + \
                                   (retraction_df[col].astype(str).str.lower() == 'true').sum()
                        else:
                            count = retraction_df[col].fillna(False).astype(bool).sum()
                        reason_counts[col.replace('reason_', '').title()] = count
                
                if reason_counts:
                    labels = list(reason_counts.keys())
                    values = list(reason_counts.values())
                    total = sum(values) if sum(values) > 0 else 1
                    percentages = [v/total*100 for v in values] if total > 0 else [0] * len(values)
                    
                    fig = px.bar(
                        x=percentages,
                        y=labels,
                        orientation='h',
                        title='Retraction Reason Distribution',
                        labels={'x': 'Percentage (%)', 'y': 'Reason'},
                        color=percentages,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Retraction reason columns not found.")
            
            with sub_tab2:
                st.subheader("Responsibility Subject Distribution")
                
                if 'responsibility_subject' in retraction_df.columns:
                    responsibility_data = retraction_df['responsibility_subject'].fillna('')
                    
                    # Count occurrences
                    author_count = responsibility_data.str.contains('author', case=False, na=False).sum()
                    journal_count = responsibility_data.str.contains('journal|editor', case=False, na=False).sum()
                    institution_count = responsibility_data.str.contains('institution|university', case=False, na=False).sum()
                    other_count = len(responsibility_data) - author_count - journal_count - institution_count
                    
                    # Ensure non-negative
                    author_count = max(0, author_count)
                    journal_count = max(0, journal_count)
                    institution_count = max(0, institution_count)
                    other_count = max(0, other_count)
                    
                    if author_count + journal_count + institution_count + other_count > 0:
                        labels = ['Author', 'Journal/Editor', 'Institution', 'Other/None']
                        sizes = [author_count, journal_count, institution_count, other_count]
                        
                        # Filter out zeros
                        filtered_data = [(l, s) for l, s in zip(labels, sizes) if s > 0]
                        if filtered_data:
                            filtered_labels, filtered_sizes = zip(*filtered_data)
                            
                            fig = px.pie(
                                values=filtered_sizes,
                                names=filtered_labels,
                                title='Responsibility Subject Distribution in Retraction Notices'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No responsibility data found.")
                    else:
                        st.warning("No responsibility data found.")
                else:
                    st.warning("Responsibility subject column not found.")
            
            with sub_tab3:
                st.subheader("Template Similarity vs Investigation Terms")
                
                # Scatter plot with regression line
                fig = px.scatter(
                    retraction_df,
                    x='template_similarity',
                    y='investigation_terms_count',
                    title='Template Similarity vs Investigation Terms in Retraction Notices',
                    labels={
                        'template_similarity': 'Template Similarity',
                        'investigation_terms_count': 'Investigation Terms Count'
                    },
                    trendline="ols"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation
                corr = retraction_df['template_similarity'].corr(retraction_df['investigation_terms_count'])
                st.metric("Correlation", f"{corr:.3f}")
            
            with sub_tab4:
                st.subheader("Negative Tone Strength Distribution")
                
                fig = px.violin(
                    retraction_df,
                    y='tone_negative_strength',
                    title='Negative Tone Strength Distribution in Retraction Notices',
                    labels={'tone_negative_strength': 'Negative Tone Strength'},
                    box=True,
                    points='all'
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean", f"{retraction_df['tone_negative_strength'].mean():.4f}")
                with col2:
                    st.metric("Median", f"{retraction_df['tone_negative_strength'].median():.4f}")
        else:
            st.warning("Retraction data not available.")
    
    with tab4:
        st.header("Comparison Analysis")
        
        if 'abstract' in data and 'retraction' in data:
            abstract_df = data['abstract']
            retraction_df = data['retraction']
            
            # Sub-tabs for comparison
            sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
                "Statistical Significance", "Language Complexity", "Readability",
                "Tone & Sentiment", "Information Density"
            ])
            
            with sub_tab1:
                st.subheader("T-Statistics Comparison")
                
                if 'differences' in data:
                    diff_stats = data['differences']
                    
                    # Extract t-statistics
                    metrics = []
                    t_stats = []
                    
                    for metric, stats_dict in diff_stats.items():
                        if 't_statistic' in stats_dict:
                            metrics.append(metric.replace('_', ' ').title())
                            t_stats.append(abs(stats_dict['t_statistic']))
                    
                    # Sort by t-statistic
                    sorted_data = sorted(zip(metrics, t_stats), key=lambda x: x[1], reverse=True)
                    metrics, t_stats = zip(*sorted_data) if sorted_data else ([], [])
                    
                    if metrics:
                        fig = px.bar(
                            x=t_stats,
                            y=metrics,
                            orientation='h',
                            title='Significance of Differences between Abstracts and Retraction Notices',
                            labels={'x': '|T-Statistic| (Absolute Value)', 'y': 'Metric'},
                            color=t_stats,
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Difference statistics not available.")
            
            with sub_tab2:
                st.subheader("Language Complexity Comparison")
                
                complexity_metrics = [
                    'avg_sentence_length', 'avg_word_length',
                    'type_token_ratio', 'passive_voice_ratio'
                ]
                
                # Prepare data for comparison
                comparison_data = []
                for metric in complexity_metrics:
                    if metric in abstract_df.columns and metric in retraction_df.columns:
                        abs_data = abstract_df[metric].dropna()
                        ret_data = retraction_df[metric].dropna()
                        
                        comparison_data.append({
                            'Metric': metric.replace('_', ' ').title(),
                            'Abstract': abs_data.mean(),
                            'Retraction': ret_data.mean()
                        })
                
                if comparison_data:
                    comp_df = pd.DataFrame(comparison_data)
                    comp_melted = comp_df.melt(
                        id_vars='Metric',
                        value_vars=['Abstract', 'Retraction'],
                        var_name='Type',
                        value_name='Value'
                    )
                    
                    fig = px.bar(
                        comp_melted,
                        x='Metric',
                        y='Value',
                        color='Type',
                        barmode='group',
                        title='Language Complexity Comparison',
                        color_discrete_map={
                            'Abstract': '#3498db',
                            'Retraction': '#e74c3c'
                        }
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            with sub_tab3:
                st.subheader("Readability Comparison")
                
                readability_metrics = [
                    'flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index'
                ]
                
                fig = make_subplots(rows=1, cols=3,
                                  subplot_titles=['Flesch Reading Ease', 'Flesch-Kincaid Grade', 'SMOG Index'])
                
                for idx, metric in enumerate(readability_metrics):
                    if metric in abstract_df.columns and metric in retraction_df.columns:
                        abs_data = abstract_df[metric].dropna()
                        ret_data = retraction_df[metric].dropna()
                        
                        fig.add_trace(
                            go.Histogram(x=abs_data, name='Abstract', opacity=0.6, nbinsx=50),
                            row=1, col=idx+1
                        )
                        fig.add_trace(
                            go.Histogram(x=ret_data, name='Retraction', opacity=0.6, nbinsx=50),
                            row=1, col=idx+1
                        )
                
                fig.update_layout(height=400, barmode='overlay',
                                title_text="Readability Metrics Comparison")
                fig.update_xaxes(title_text="Value", row=1, col=1)
                fig.update_xaxes(title_text="Value", row=1, col=2)
                fig.update_xaxes(title_text="Value", row=1, col=3)
                st.plotly_chart(fig, use_container_width=True)
            
            with sub_tab4:
                st.subheader("Tone Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Scatter plot: Assertive vs Hedging
                    fig_scatter = go.Figure()
                    
                    # Abstract points
                    fig_scatter.add_trace(go.Scatter(
                        x=abstract_df['assertive_word_count'].head(5000),
                        y=abstract_df['hedging_word_count'].head(5000),
                        mode='markers',
                        name='Abstract',
                        marker=dict(color='#3498db', opacity=0.6, size=5)
                    ))
                    
                    # Retraction points
                    fig_scatter.add_trace(go.Scatter(
                        x=retraction_df['assertive_word_count'].head(5000),
                        y=retraction_df['hedging_word_count'].head(5000),
                        mode='markers',
                        name='Retraction',
                        marker=dict(color='#e74c3c', opacity=0.6, size=5)
                    ))
                    
                    fig_scatter.update_layout(
                        title='Tone Comparison: Abstract vs Retraction',
                        xaxis_title='Assertive Word Count',
                        yaxis_title='Hedging Word Count',
                        height=500
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # Sentiment violin plot
                    sentiment_data = []
                    sentiment_data.extend([{'Type': 'Abstract', 'Sentiment': v} 
                                          for v in abstract_df['sentiment_score'].dropna().head(5000)])
                    sentiment_data.extend([{'Type': 'Retraction', 'Sentiment': v} 
                                          for v in retraction_df['sentiment_score'].dropna().head(5000)])
                    sentiment_df_comp = pd.DataFrame(sentiment_data)
                    
                    fig_violin = px.violin(
                        sentiment_df_comp,
                        x='Type',
                        y='Sentiment',
                        title='Sentiment Score Distribution Comparison',
                        color='Type',
                        color_discrete_map={
                            'Abstract': '#3498db',
                            'Retraction': '#e74c3c'
                        },
                        box=True
                    )
                    fig_violin.update_layout(height=500)
                    st.plotly_chart(fig_violin, use_container_width=True)
            
            with sub_tab5:
                st.subheader("Information Density Comparison")
                
                fig = go.Figure()
                
                fig.add_trace(go.Box(
                    y=abstract_df['information_density'].dropna(),
                    name='Abstract',
                    marker_color='#3498db'
                ))
                
                fig.add_trace(go.Box(
                    y=retraction_df['information_density'].dropna(),
                    name='Retraction',
                    marker_color='#e74c3c'
                ))
                
                fig.update_layout(
                    title='Information Density Comparison',
                    yaxis_title='Information Density',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Both abstract and retraction data are required for comparison.")
else:
    st.info("👈 Please load data using the sidebar to view visualizations.")

