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
consistency_analysis_path = os.path.join(abstract_retraction_dir, "consistency_analysis_results.csv")
statistical_tests_path = os.path.join(abstract_retraction_dir, "statistical_tests_results.json")

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
        
        # Load statistical tests results
        if os.path.exists(statistical_tests_path):
            with open(statistical_tests_path, 'r', encoding='utf-8') as f:
                data['statistical_tests'] = json.load(f)
        
        # Load consistency analysis results
        if os.path.exists(consistency_analysis_path):
            cache_path = get_cache_path(consistency_analysis_path)
            metadata_path = get_cache_metadata_path(consistency_analysis_path)
            
            if is_cache_valid(consistency_analysis_path, cache_path, metadata_path):
                cached_df = load_cache(consistency_analysis_path)
                if cached_df is not None:
                    data['consistency'] = cached_df
                else:
                    df = pd.read_csv(consistency_analysis_path)
                    data['consistency'] = df
                    save_cache(consistency_analysis_path, df)
            else:
                df = pd.read_csv(consistency_analysis_path)
                data['consistency'] = df
                save_cache(consistency_analysis_path, df)
        
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
        if os.path.exists(consistency_analysis_path):
            clear_cache(consistency_analysis_path)
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
        
        # Statistical Validation Results
        if 'statistical_tests' in data:
            st.subheader("Statistical Validation Results")
            stats_data = data['statistical_tests']
            
            # Summary metrics
            summary = stats_data.get('summary', {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Tests", f"{summary.get('total_tests', 0)}")
            with col2:
                st.metric("Significant Tests", f"{summary.get('significant_tests', 0)}")
            with col3:
                st.metric("Significance Rate", f"{summary.get('significance_rate', 0):.1f}%")
            
            tests = stats_data.get('tests', {})
            
            # Test 1: Chi-square Test
            if 'chi_square_sentiment' in tests:
                chi2_test = tests['chi_square_sentiment']
                st.markdown("### Chi-square Test: Sentiment Classification Distribution")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create contingency table visualization
                    contingency = chi2_test.get('contingency_table', {})
                    abs_counts = contingency.get('Abstract', {})
                    ret_counts = contingency.get('Retraction', {})
                    
                    # Create bar chart for sentiment distribution
                    sentiment_data = {
                        'Abstract': {
                            'Negative': abs_counts.get('Negative', 0),
                            'Neutral': abs_counts.get('Neutral', 0),
                            'Positive': abs_counts.get('Positive', 0)
                        },
                        'Retraction': {
                            'Negative': ret_counts.get('Negative', 0),
                            'Neutral': ret_counts.get('Neutral', 0),
                            'Positive': ret_counts.get('Positive', 0)
                        }
                    }
                    
                    # Create comparison bar chart
                    categories = ['Negative', 'Neutral', 'Positive']
                    abs_values = [sentiment_data['Abstract'][cat] for cat in categories]
                    ret_values = [sentiment_data['Retraction'][cat] for cat in categories]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=categories,
                        y=abs_values,
                        name='Abstract',
                        marker_color='#3498db'
                    ))
                    fig.add_trace(go.Bar(
                        x=categories,
                        y=ret_values,
                        name='Retraction',
                        marker_color='#e74c3c'
                    ))
                    
                    fig.update_layout(
                        title='Sentiment Classification Distribution: Abstract vs Retraction',
                        xaxis_title='Sentiment Category',
                        yaxis_title='Count',
                        barmode='group',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("📊 **Analysis**: This bar chart compares sentiment classification distributions between abstracts and retraction notices. The chi-square test determines if these distributions are significantly different.")
                
                with col2:
                    st.metric("Chi-square Statistic", f"{chi2_test.get('statistic', 0):.4f}")
                    st.metric("P-value", f"{chi2_test.get('p_value', 0):.6e}")
                    st.metric("Degrees of Freedom", f"{chi2_test.get('degrees_of_freedom', 0)}")
                    st.metric("Cramer's V", f"{chi2_test.get('cramers_v', 0):.4f}")
                    significance = chi2_test.get('significance', 'Not Significant')
                    color = 'green' if significance == 'Significant' else 'gray'
                    st.markdown(f"**Result**: <span style='color:{color}'>{significance}</span>", unsafe_allow_html=True)
                    
                    # Contingency table
                    st.markdown("**Contingency Table:**")
                    contingency_df = pd.DataFrame({
                        'Abstract': [
                            abs_counts.get('Negative', 0),
                            abs_counts.get('Neutral', 0),
                            abs_counts.get('Positive', 0)
                        ],
                        'Retraction': [
                            ret_counts.get('Negative', 0),
                            ret_counts.get('Neutral', 0),
                            ret_counts.get('Positive', 0)
                        ]
                    }, index=['Negative', 'Neutral', 'Positive'])
                    st.dataframe(contingency_df, use_container_width=True)
            
            # Test 2: t-test
            if 't_test_sentiment' in tests and 'abstract' in data and 'retraction' in data:
                ttest = tests['t_test_sentiment']
                st.markdown("### Independent Samples t-test: Sentiment Scores")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Use actual data for visualization
                    abstract_df = data['abstract']
                    retraction_df = data['retraction']
                    
                    abs_sentiment = abstract_df['sentiment_score'].dropna()
                    ret_sentiment = retraction_df['sentiment_score'].dropna()
                    
                    # Create box plot comparison with actual data
                    fig = go.Figure()
                    
                    fig.add_trace(go.Box(
                        y=abs_sentiment.head(10000),  # Limit for performance
                        name='Abstract',
                        marker_color='#3498db',
                        boxmean='sd'
                    ))
                    
                    fig.add_trace(go.Box(
                        y=ret_sentiment.head(10000),  # Limit for performance
                        name='Retraction',
                        marker_color='#e74c3c',
                        boxmean='sd'
                    ))
                    
                    fig.update_layout(
                        title='Sentiment Score Distribution: Abstract vs Retraction',
                        yaxis_title='Sentiment Score',
                        height=500,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show group statistics
                    abs_stats = ttest.get('abstract_stats', {})
                    ret_stats = ttest.get('retraction_stats', {})
                    
                    st.markdown("**Group Statistics:**")
                    stats_df = pd.DataFrame({
                        'Abstract': [
                            f"{abs_stats.get('mean', 0):.4f}",
                            f"{abs_stats.get('std', 0):.4f}",
                            f"{abs_stats.get('count', 0):,}",
                            f"{abs_stats.get('median', 0):.4f}"
                        ],
                        'Retraction': [
                            f"{ret_stats.get('mean', 0):.4f}",
                            f"{ret_stats.get('std', 0):.4f}",
                            f"{ret_stats.get('count', 0):,}",
                            f"{ret_stats.get('median', 0):.4f}"
                        ]
                    }, index=['Mean', 'Std Dev', 'Count', 'Median'])
                    st.dataframe(stats_df, use_container_width=True)
                    
                    st.caption("📊 **Analysis**: This box plot and statistics table compare sentiment score distributions between abstracts and retraction notices. The t-test determines if the mean sentiment scores are significantly different.")
                
                with col2:
                    st.metric("T-statistic", f"{ttest.get('statistic', 0):.4f}")
                    st.metric("P-value", f"{ttest.get('p_value', 0):.6e}")
                    st.metric("Cohen's d", f"{ttest.get('cohens_d', 0):.4f}")
                    effect_size = ttest.get('effect_size_interpretation', 'Unknown')
                    st.metric("Effect Size", effect_size)
                    significance = ttest.get('significance', 'Not Significant')
                    color = 'green' if significance == 'Significant' else 'gray'
                    st.markdown(f"**Result**: <span style='color:{color}'>{significance}</span>", unsafe_allow_html=True)
                    
                    # Mean difference and CI
                    diff_info = ttest.get('difference', {})
                    st.markdown("**Mean Difference:**")
                    st.metric("Difference", f"{diff_info.get('mean_difference', 0):.4f}")
                    st.markdown(f"**95% CI:**<br>[{diff_info.get('ci_lower', 0):.4f}, {diff_info.get('ci_upper', 0):.4f}]", unsafe_allow_html=True)
    
    with tab2:
        st.header("Abstract Individual Analysis")
        
        if 'abstract' in data:
            abstract_df = data['abstract']
            
            # Sub-tabs for abstract analysis
            sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
                "Language Complexity", "Tone & Attitude", 
                "Scientific Features", "Consistency"
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
                st.caption("📊 **Analysis**: This box plot shows the distribution of four key language complexity metrics - average sentence length, word length, vocabulary diversity (type-token ratio), and passive voice usage. It helps identify the complexity patterns in abstract writing.")
            
            with sub_tab2:
                st.subheader("Tone and Attitude Analysis")
                
                # Sentiment Distribution Section
                st.markdown("### Sentiment Score Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram of sentiment scores
                    fig_hist = px.histogram(
                        abstract_df,
                        x='sentiment_score',
                        nbins=50,
                        title='Sentiment Score Distribution',
                        labels={'sentiment_score': 'Sentiment Score', 'count': 'Frequency'},
                        color_discrete_sequence=['#3498db']
                    )
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                    st.caption("📊 **Analysis**: This histogram shows the frequency distribution of sentiment scores across all abstracts, revealing whether most abstracts are positive, neutral, or negative in tone.")
                
                with col2:
                    # Violin plot with box plot
                    fig_violin = px.violin(
                        abstract_df,
                        y='sentiment_score',
                        title='Sentiment Score Distribution (Violin Plot)',
                        labels={'sentiment_score': 'Sentiment Score'},
                        box=True,
                        points='all',
                        color_discrete_sequence=['#3498db']
                    )
                    fig_violin.update_layout(height=400)
                    st.plotly_chart(fig_violin, use_container_width=True)
                    st.caption("📊 **Analysis**: The violin plot provides a detailed view of sentiment score distribution, showing density at different values along with quartiles and median. It helps identify clusters and outliers in sentiment patterns.")
                
                # Sentiment Classification
                st.markdown("### Sentiment Classification")
                col1, col2, col3 = st.columns(3)
                
                # Classify sentiment
                sentiment_scores = abstract_df['sentiment_score'].dropna()
                positive_count = (sentiment_scores > 0.1).sum()
                neutral_count = ((sentiment_scores >= -0.1) & (sentiment_scores <= 0.1)).sum()
                negative_count = (sentiment_scores < -0.1).sum()
                total_count = len(sentiment_scores)
                
                with col1:
                    st.metric("Positive Sentiment", f"{positive_count:,}", 
                             f"{(positive_count/total_count*100):.1f}%")
                with col2:
                    st.metric("Neutral Sentiment", f"{neutral_count:,}", 
                             f"{(neutral_count/total_count*100):.1f}%")
                with col3:
                    st.metric("Negative Sentiment", f"{negative_count:,}", 
                             f"{(negative_count/total_count*100):.1f}%")
                
                # Pie chart for sentiment classification
                sentiment_data = {
                    'Positive': positive_count,
                    'Neutral': neutral_count,
                    'Negative': negative_count
                }
                fig_pie = px.pie(
                    values=list(sentiment_data.values()),
                    names=list(sentiment_data.keys()),
                    title='Sentiment Classification Distribution',
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Neutral': '#f39c12',
                        'Negative': '#e74c3c'
                    }
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
                st.caption("📊 **Analysis**: This pie chart visualizes the proportion of abstracts classified as positive, neutral, or negative based on sentiment scores. It provides a quick overview of the overall emotional tone distribution in the dataset.")
                
                # Tone and Attitude Scatter Plot
                st.markdown("### Tone and Attitude Scatter Plot")
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
                st.caption("📊 **Analysis**: This scatter plot explores the relationship between assertive words (confidence) and hedging words (uncertainty) in abstracts, colored by sentiment score. It reveals how different writing styles correlate with emotional tone.")
                
                # Sentiment vs Text Length
                st.markdown("### Sentiment vs Text Characteristics")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment vs Sentence Length
                    if 'avg_sentence_length' in abstract_df.columns:
                        fig_sent_len = px.scatter(
                            abstract_df,
                            x='avg_sentence_length',
                            y='sentiment_score',
                            color='sentiment_score',
                            color_continuous_scale='RdYlGn',
                            title='Sentiment Score vs Average Sentence Length',
                            labels={
                                'avg_sentence_length': 'Average Sentence Length',
                                'sentiment_score': 'Sentiment Score'
                            },
                            trendline="ols"
                        )
                        fig_sent_len.update_layout(height=400)
                        st.plotly_chart(fig_sent_len, use_container_width=True)
                        st.caption("📊 **Analysis**: This scatter plot examines whether longer sentences correlate with different sentiment scores. The trend line helps identify any relationship between sentence complexity and emotional tone.")
                
                with col2:
                    # Sentiment vs Assertive Words
                    fig_sent_assert = px.scatter(
                        abstract_df,
                        x='assertive_word_count',
                        y='sentiment_score',
                        color='sentiment_score',
                        color_continuous_scale='RdYlGn',
                        title='Sentiment Score vs Assertive Word Count',
                        labels={
                            'assertive_word_count': 'Assertive Word Count',
                            'sentiment_score': 'Sentiment Score'
                        },
                        trendline="ols"
                    )
                    fig_sent_assert.update_layout(height=400)
                    st.plotly_chart(fig_sent_assert, use_container_width=True)
                    st.caption("📊 **Analysis**: This chart investigates how the use of assertive language (confident statements) relates to sentiment scores. It helps understand whether confident writing is associated with positive or negative sentiment.")
                
                # Subjectivity Analysis
                if 'subjectivity_score' in abstract_df.columns:
                    st.markdown("### Subjectivity Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Subjectivity distribution
                        fig_subj = px.histogram(
                            abstract_df,
                            x='subjectivity_score',
                            nbins=50,
                            title='Subjectivity Score Distribution',
                            labels={'subjectivity_score': 'Subjectivity Score', 'count': 'Frequency'},
                            color_discrete_sequence=['#9b59b6']
                        )
                        fig_subj.update_layout(height=400)
                        st.plotly_chart(fig_subj, use_container_width=True)
                        st.caption("📊 **Analysis**: This histogram displays the distribution of subjectivity scores, which measure how opinionated or factual the abstracts are. Higher scores indicate more subjective, opinion-based content.")
                    
                    with col2:
                        # Sentiment vs Subjectivity
                        fig_sent_subj = px.scatter(
                            abstract_df,
                            x='subjectivity_score',
                            y='sentiment_score',
                            color='sentiment_score',
                            color_continuous_scale='RdYlGn',
                            title='Sentiment vs Subjectivity',
                            labels={
                                'subjectivity_score': 'Subjectivity Score',
                                'sentiment_score': 'Sentiment Score'
                            },
                            trendline="ols"
                        )
                        fig_sent_subj.update_layout(height=400)
                        st.plotly_chart(fig_sent_subj, use_container_width=True)
                        st.caption("📊 **Analysis**: This scatter plot explores the relationship between subjectivity (opinion vs fact) and sentiment (positive vs negative). It helps understand if more subjective writing tends to be more emotional.")
                
                # Tone Metrics Summary
                st.markdown("### Tone Metrics Summary Statistics")
                tone_metrics = ['assertive_word_count', 'hedging_word_count', 'negation_count', 'modal_word_count']
                existing_metrics = [m for m in tone_metrics if m in abstract_df.columns]
                
                if existing_metrics:
                    summary_data = []
                    for metric in existing_metrics:
                        summary_data.append({
                            'Metric': metric.replace('_', ' ').title(),
                            'Mean': abstract_df[metric].mean(),
                            'Median': abstract_df[metric].median(),
                            'Std': abstract_df[metric].std(),
                            'Min': abstract_df[metric].min(),
                            'Max': abstract_df[metric].max()
                        })
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                
                # Sentiment Statistics
                st.markdown("### Sentiment Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Sentiment", f"{abstract_df['sentiment_score'].mean():.3f}")
                with col2:
                    st.metric("Median Sentiment", f"{abstract_df['sentiment_score'].median():.3f}")
                with col3:
                    st.metric("Std Deviation", f"{abstract_df['sentiment_score'].std():.3f}")
                with col4:
                    st.metric("Range", f"{abstract_df['sentiment_score'].max() - abstract_df['sentiment_score'].min():.3f}")
            
            with sub_tab3:
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
                    st.caption("📊 **Analysis**: This scatter plot examines the relationship between scientific terminology and statistical evidence in abstracts. It helps identify abstracts that are more data-driven and statistically rigorous.")
                
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
                    st.caption("📊 **Analysis**: This histogram shows the distribution of innovation-related expressions (e.g., 'novel', 'first', 'improvement') in abstracts. It reveals how frequently abstracts emphasize their innovative contributions.")
            
            with sub_tab4:
                st.subheader("Abstract Consistency Analysis")
                st.markdown("**Goal**: Understand semantic stability and language consistency within abstracts.")
                
                if 'consistency' in data:
                    consistency_df = data['consistency']
                    valid_df = consistency_df[
                        (consistency_df['abs_keyword_concentration'] > 0) |
                        (consistency_df['abs_sentiment_consistency'] > 0)
                    ].copy()
                    
                    if len(valid_df) > 0:
                        # 1. Keyword Concentration Distribution
                        st.markdown("### Keyword Concentration Distribution")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig_keyword = px.histogram(
                                valid_df,
                                x='abs_keyword_concentration',
                                nbins=50,
                                title='Abstract Keyword Concentration Distribution',
                                labels={
                                    'abs_keyword_concentration': 'Keyword Concentration',
                                    'count': 'Frequency'
                                },
                                color_discrete_sequence=['#3498db']
                            )
                            fig_keyword.update_layout(height=500)
                            st.plotly_chart(fig_keyword, use_container_width=True)
                            st.caption("📊 **Analysis**: This histogram shows the distribution of keyword concentration in abstracts. Higher concentration indicates that abstracts focus on a few core topics, suggesting clear research focus and strong thematic consistency.")
                        
                        with col2:
                            st.metric("Mean Concentration", f"{valid_df['abs_keyword_concentration'].mean():.3f}")
                            st.metric("Median Concentration", f"{valid_df['abs_keyword_concentration'].median():.3f}")
                            st.metric("Std Deviation", f"{valid_df['abs_keyword_concentration'].std():.3f}")
                        
                        # 2. Sentiment Consistency Distribution
                        st.markdown("### Sentiment Consistency Distribution")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig_sent_cons = px.histogram(
                                valid_df,
                                x='abs_sentiment_consistency',
                                nbins=50,
                                title='Abstract Sentiment Consistency Distribution',
                                labels={
                                    'abs_sentiment_consistency': 'Sentiment Consistency Score',
                                    'count': 'Frequency'
                                },
                                color_discrete_sequence=['#3498db']
                            )
                            fig_sent_cons.update_layout(height=500)
                            st.plotly_chart(fig_sent_cons, use_container_width=True)
                            st.caption("📊 **Analysis**: This histogram displays sentiment consistency scores for abstracts. Higher scores indicate lower variance in sentiment across sentences, suggesting more stable emotional tone and internal language consistency.")
                        
                        with col2:
                            st.metric("Mean Consistency", f"{valid_df['abs_sentiment_consistency'].mean():.3f}")
                            st.metric("Median Consistency", f"{valid_df['abs_sentiment_consistency'].median():.3f}")
                            st.metric("Mean Variance", f"{valid_df['abs_sentiment_variance'].mean():.4f}")
                        
                        # 3. Tone Balance: Assertive vs Hedging
                        st.markdown("### Tone Balance: Assertive vs Hedging Words")
                        fig_tone = px.scatter(
                            valid_df,
                            x='abs_assertive_count',
                            y='abs_hedging_count',
                            title='Abstract Tone Balance: Assertive vs Hedging Words',
                            labels={
                                'abs_assertive_count': 'Assertive Word Count',
                                'abs_hedging_count': 'Hedging Word Count'
                            },
                            color='abs_tone_balance',
                            color_continuous_scale='RdYlGn',
                            hover_data=['record_id']
                        )
                        fig_tone.update_layout(height=600)
                        st.plotly_chart(fig_tone, use_container_width=True)
                        st.caption("📊 **Analysis**: This scatter plot compares assertive words (confident statements) with hedging words (uncertainty expressions) in abstracts. The color scale shows tone balance - yellow/green indicates more assertive language, while red indicates more hedging language. This reflects language expression uniformity.")
                    else:
                        st.warning("No valid consistency data available for abstracts.")
                else:
                    st.warning("Consistency analysis data not available.")
        else:
            st.warning("Abstract data not available.")
    
    with tab3:
        st.header("Retraction Individual Analysis")
        
        if 'retraction' in data:
            retraction_df = data['retraction']
            
            # Sub-tabs for retraction analysis
            sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
                "Retraction Reasons", "Responsibility", "Template & Investigation", "Tone Strength", "Consistency"
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
                    st.caption("📊 **Analysis**: This horizontal bar chart shows the percentage distribution of different retraction reasons (data issues, statistical errors, ethical concerns, plagiarism, etc.). It helps identify the most common causes for retractions.")
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
                            st.caption("📊 **Analysis**: This pie chart shows who is typically held responsible in retraction notices - authors, journals/editors, institutions, or others. It reveals accountability patterns in retraction statements.")
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
                st.caption("📊 **Analysis**: This scatter plot examines the relationship between template similarity (how standardized the retraction notice is) and investigation terms (formality of the language). The trend line shows if more formal investigations use standardized templates.")
                
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
                st.caption("📊 **Analysis**: This violin plot shows the distribution of negative tone strength in retraction notices. It reveals how harshly worded retractions are, with the width indicating the density of notices at different negative tone levels.")
                
                # Statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean", f"{retraction_df['tone_negative_strength'].mean():.4f}")
                with col2:
                    st.metric("Median", f"{retraction_df['tone_negative_strength'].median():.4f}")
            
            with sub_tab5:
                st.subheader("Retraction Consistency Analysis")
                st.markdown("**Goal**: Understand consistency in wording and semantics within retraction notices.")
                
                if 'consistency' in data:
                    consistency_df = data['consistency']
                    valid_df = consistency_df[
                        (consistency_df['ret_sentiment_consistency'] > 0) |
                        (consistency_df['ret_negative_word_count'] > 0)
                    ].copy()
                    
                    if len(valid_df) > 0:
                        # 1. Negative Word Distribution
                        st.markdown("### Negative Word Distribution")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig_neg = px.histogram(
                                valid_df,
                                x='ret_negative_word_count',
                                nbins=50,
                                title='Retraction Negative Word Distribution',
                                labels={
                                    'ret_negative_word_count': 'Negative Word Count',
                                    'count': 'Frequency'
                                },
                                color_discrete_sequence=['#e74c3c']
                            )
                            fig_neg.update_layout(height=500)
                            st.plotly_chart(fig_neg, use_container_width=True)
                            st.caption("📊 **Analysis**: This histogram shows the distribution of negative words (e.g., invalid, unreliable, concern) in retraction notices. Higher frequency indicates more standardized language and official tone, reflecting internal language consistency.")
                        
                        with col2:
                            st.metric("Mean Negative Words", f"{valid_df['ret_negative_word_count'].mean():.2f}")
                            st.metric("Median Negative Words", f"{valid_df['ret_negative_word_count'].median():.0f}")
                            st.metric("Max Negative Words", f"{valid_df['ret_negative_word_count'].max():.0f}")
                        
                        # 2. Template Phrase Distribution
                        st.markdown("### Template Phrase Distribution")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig_template = px.histogram(
                                valid_df,
                                x='ret_template_phrase_count',
                                nbins=20,
                                title='Retraction Template Phrase Count Distribution',
                                labels={
                                    'ret_template_phrase_count': 'Template Phrase Count',
                                    'count': 'Frequency'
                                },
                                color_discrete_sequence=['#e74c3c']
                            )
                            fig_template.update_layout(height=500)
                            st.plotly_chart(fig_template, use_container_width=True)
                            st.caption("📊 **Analysis**: This histogram displays the distribution of template phrases (e.g., 'retract', 'the editors', 'following publication', 'concerns were') in retraction notices. Higher counts indicate more standardized, template-like language, suggesting stronger consistency across different retraction notices.")
                        
                        with col2:
                            st.metric("Mean Template Phrases", f"{valid_df['ret_template_phrase_count'].mean():.2f}")
                            st.metric("Median Template Phrases", f"{valid_df['ret_template_phrase_count'].median():.0f}")
                            st.metric("Max Template Phrases", f"{valid_df['ret_template_phrase_count'].max():.0f}")
                        
                        # 3. Sentiment Consistency Distribution
                        st.markdown("### Sentiment Consistency Distribution")
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig_ret_sent = px.histogram(
                                valid_df,
                                x='ret_sentiment_consistency',
                                nbins=50,
                                title='Retraction Sentiment Consistency Distribution',
                                labels={
                                    'ret_sentiment_consistency': 'Sentiment Consistency Score',
                                    'count': 'Frequency'
                                },
                                color_discrete_sequence=['#e74c3c']
                            )
                            fig_ret_sent.update_layout(height=500)
                            st.plotly_chart(fig_ret_sent, use_container_width=True)
                            st.caption("📊 **Analysis**: This histogram shows sentiment consistency scores for retraction notices. Retraction notices typically concentrate in the negative sentiment range, and lower variance (higher consistency scores) indicates stronger language consistency.")
                        
                        with col2:
                            st.metric("Mean Consistency", f"{valid_df['ret_sentiment_consistency'].mean():.3f}")
                            st.metric("Median Consistency", f"{valid_df['ret_sentiment_consistency'].median():.3f}")
                            st.metric("Mean Variance", f"{valid_df['ret_sentiment_variance'].mean():.4f}")
                    else:
                        st.warning("No valid consistency data available for retractions.")
                else:
                    st.warning("Consistency analysis data not available.")
        else:
            st.warning("Retraction data not available.")
    
    with tab4:
        st.header("Comparison Analysis")
        
        if 'abstract' in data and 'retraction' in data:
            abstract_df = data['abstract']
            retraction_df = data['retraction']
            
            # Sub-tabs for comparison
            sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5, sub_tab6 = st.tabs([
                "Statistical Significance", "Language Complexity", "Readability",
                "Tone & Sentiment", "Information Density", "Consistency"
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
                        st.caption("📊 **Analysis**: This horizontal bar chart displays the absolute values of t-statistics for each metric, sorted by significance. Higher values indicate more statistically significant differences between abstracts and retraction notices.")
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
                    st.caption("📊 **Analysis**: This grouped bar chart compares language complexity metrics between abstracts and retraction notices side by side. It highlights differences in sentence structure, vocabulary, and writing style between the two text types.")
            
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
                st.caption("📊 **Analysis**: These overlapping histograms compare readability scores between abstracts and retraction notices. The overlay shows where the distributions overlap and where they differ, revealing differences in reading difficulty.")
            
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
                    st.caption("📊 **Analysis**: This scatter plot compares the use of assertive vs hedging words between abstracts and retraction notices. Different colors help distinguish writing styles and tone strategies used in each text type.")
                
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
                    st.caption("📊 **Analysis**: This violin plot compares sentiment score distributions between abstracts and retraction notices side by side. It reveals differences in emotional tone, showing whether abstracts are generally more positive than retraction notices.")
                
                # Sentiment Distribution Histogram Comparison
                st.markdown("### Sentiment Distribution Histogram Comparison")
                fig_hist_comp = go.Figure()
                
                fig_hist_comp.add_trace(go.Histogram(
                    x=abstract_df['sentiment_score'].dropna(),
                    name='Abstract',
                    opacity=0.6,
                    nbinsx=50,
                    marker_color='#3498db'
                ))
                
                fig_hist_comp.add_trace(go.Histogram(
                    x=retraction_df['sentiment_score'].dropna(),
                    name='Retraction',
                    opacity=0.6,
                    nbinsx=50,
                    marker_color='#e74c3c'
                ))
                
                fig_hist_comp.update_layout(
                    title='Sentiment Score Distribution: Abstract vs Retraction',
                    xaxis_title='Sentiment Score',
                    yaxis_title='Frequency',
                    barmode='overlay',
                    height=500
                )
                st.plotly_chart(fig_hist_comp, use_container_width=True)
                st.caption("📊 **Analysis**: This overlapping histogram compares sentiment score distributions between abstracts and retraction notices. The overlay shows where the distributions overlap and where they differ, revealing differences in emotional tone patterns.")
                
                # Sentiment Classification Comparison
                st.markdown("### Sentiment Classification Comparison")
                col1, col2 = st.columns(2)
                
                # Calculate sentiment classification for abstracts
                abstract_sentiment_scores = abstract_df['sentiment_score'].dropna()
                abstract_positive = (abstract_sentiment_scores > 0.1).sum()
                abstract_neutral = ((abstract_sentiment_scores >= -0.1) & (abstract_sentiment_scores <= 0.1)).sum()
                abstract_negative = (abstract_sentiment_scores < -0.1).sum()
                
                # Calculate sentiment classification for retractions
                retraction_sentiment_scores = retraction_df['sentiment_score'].dropna()
                retraction_positive = (retraction_sentiment_scores > 0.1).sum()
                retraction_neutral = ((retraction_sentiment_scores >= -0.1) & (retraction_sentiment_scores <= 0.1)).sum()
                retraction_negative = (retraction_sentiment_scores < -0.1).sum()
                
                with col1:
                    # Abstract sentiment pie chart
                    fig_pie_abs = px.pie(
                        values=[abstract_positive, abstract_neutral, abstract_negative],
                        names=['Positive', 'Neutral', 'Negative'],
                        title='Abstract Sentiment Classification',
                        color_discrete_map={
                            'Positive': '#2ecc71',
                            'Neutral': '#f39c12',
                            'Negative': '#e74c3c'
                        }
                    )
                    fig_pie_abs.update_layout(height=400)
                    st.plotly_chart(fig_pie_abs, use_container_width=True)
                
                with col2:
                    # Retraction sentiment pie chart
                    fig_pie_ret = px.pie(
                        values=[retraction_positive, retraction_neutral, retraction_negative],
                        names=['Positive', 'Neutral', 'Negative'],
                        title='Retraction Sentiment Classification',
                        color_discrete_map={
                            'Positive': '#2ecc71',
                            'Neutral': '#f39c12',
                            'Negative': '#e74c3c'
                        }
                    )
                    fig_pie_ret.update_layout(height=400)
                    st.plotly_chart(fig_pie_ret, use_container_width=True)
                
                st.caption("📊 **Analysis**: These side-by-side pie charts compare sentiment classification distributions between abstracts and retraction notices. They reveal differences in the proportion of positive, neutral, and negative sentiment in each text type.")
                
                # Sentiment Statistics Comparison
                st.markdown("### Sentiment Statistics Comparison")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Sentiment", 
                             f"{abstract_df['sentiment_score'].mean():.3f}",
                             f"{retraction_df['sentiment_score'].mean():.3f}",
                             delta_color="off")
                    st.caption("Abstract / Retraction")
                
                with col2:
                    st.metric("Median Sentiment",
                             f"{abstract_df['sentiment_score'].median():.3f}",
                             f"{retraction_df['sentiment_score'].median():.3f}",
                             delta_color="off")
                    st.caption("Abstract / Retraction")
                
                with col3:
                    st.metric("Std Deviation",
                             f"{abstract_df['sentiment_score'].std():.3f}",
                             f"{retraction_df['sentiment_score'].std():.3f}",
                             delta_color="off")
                    st.caption("Abstract / Retraction")
                
                with col4:
                    abs_range = abstract_df['sentiment_score'].max() - abstract_df['sentiment_score'].min()
                    ret_range = retraction_df['sentiment_score'].max() - retraction_df['sentiment_score'].min()
                    st.metric("Range",
                             f"{abs_range:.3f}",
                             f"{ret_range:.3f}",
                             delta_color="off")
                    st.caption("Abstract / Retraction")
                
                # Sentiment vs Text Characteristics Comparison
                st.markdown("### Sentiment vs Text Characteristics Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Sentiment vs Sentence Length
                    if 'avg_sentence_length' in abstract_df.columns and 'avg_sentence_length' in retraction_df.columns:
                        fig_sent_len_comp = go.Figure()
                        
                        fig_sent_len_comp.add_trace(go.Scatter(
                            x=abstract_df['avg_sentence_length'].dropna().head(3000),
                            y=abstract_df['sentiment_score'].dropna().head(3000),
                            mode='markers',
                            name='Abstract',
                            marker=dict(color='#3498db', opacity=0.5, size=4)
                        ))
                        
                        fig_sent_len_comp.add_trace(go.Scatter(
                            x=retraction_df['avg_sentence_length'].dropna().head(3000),
                            y=retraction_df['sentiment_score'].dropna().head(3000),
                            mode='markers',
                            name='Retraction',
                            marker=dict(color='#e74c3c', opacity=0.5, size=4)
                        ))
                        
                        fig_sent_len_comp.update_layout(
                            title='Sentiment vs Sentence Length Comparison',
                            xaxis_title='Average Sentence Length',
                            yaxis_title='Sentiment Score',
                            height=400
                        )
                        st.plotly_chart(fig_sent_len_comp, use_container_width=True)
                        st.caption("📊 **Analysis**: This scatter plot compares how sentiment scores relate to sentence length in abstracts vs retraction notices. It reveals if longer sentences correlate differently with sentiment in each text type.")
                
                with col2:
                    # Sentiment vs Assertive Words Comparison
                    fig_sent_assert_comp = go.Figure()
                    
                    fig_sent_assert_comp.add_trace(go.Scatter(
                        x=abstract_df['assertive_word_count'].dropna().head(3000),
                        y=abstract_df['sentiment_score'].dropna().head(3000),
                        mode='markers',
                        name='Abstract',
                        marker=dict(color='#3498db', opacity=0.5, size=4)
                    ))
                    
                    fig_sent_assert_comp.add_trace(go.Scatter(
                        x=retraction_df['assertive_word_count'].dropna().head(3000),
                        y=retraction_df['sentiment_score'].dropna().head(3000),
                        mode='markers',
                        name='Retraction',
                        marker=dict(color='#e74c3c', opacity=0.5, size=4)
                    ))
                    
                    fig_sent_assert_comp.update_layout(
                        title='Sentiment vs Assertive Words Comparison',
                        xaxis_title='Assertive Word Count',
                        yaxis_title='Sentiment Score',
                        height=400
                    )
                    st.plotly_chart(fig_sent_assert_comp, use_container_width=True)
                    st.caption("📊 **Analysis**: This scatter plot compares how assertive language relates to sentiment scores in abstracts vs retraction notices. It shows if confident writing correlates differently with emotional tone in each text type.")
                
                # Sentiment Box Plot Comparison
                st.markdown("### Sentiment Distribution Box Plot Comparison")
                fig_box = go.Figure()
                
                fig_box.add_trace(go.Box(
                    y=abstract_df['sentiment_score'].dropna(),
                    name='Abstract',
                    marker_color='#3498db',
                    boxmean='sd'
                ))
                
                fig_box.add_trace(go.Box(
                    y=retraction_df['sentiment_score'].dropna(),
                    name='Retraction',
                    marker_color='#e74c3c',
                    boxmean='sd'
                ))
                
                fig_box.update_layout(
                    title='Sentiment Score Distribution: Box Plot Comparison',
                    yaxis_title='Sentiment Score',
                    height=500
                )
                st.plotly_chart(fig_box, use_container_width=True)
                st.caption("📊 **Analysis**: This box plot comparison shows quartiles, medians, and outliers for sentiment scores in abstracts vs retraction notices. It provides a clear statistical summary of distribution differences.")
                
                # Subjectivity Comparison (if available)
                if 'subjectivity_score' in abstract_df.columns and 'subjectivity_score' in retraction_df.columns:
                    st.markdown("### Subjectivity Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Subjectivity distribution comparison
                        fig_subj_comp = go.Figure()
                        
                        fig_subj_comp.add_trace(go.Histogram(
                            x=abstract_df['subjectivity_score'].dropna(),
                            name='Abstract',
                            opacity=0.6,
                            nbinsx=50,
                            marker_color='#3498db'
                        ))
                        
                        fig_subj_comp.add_trace(go.Histogram(
                            x=retraction_df['subjectivity_score'].dropna(),
                            name='Retraction',
                            opacity=0.6,
                            nbinsx=50,
                            marker_color='#e74c3c'
                        ))
                        
                        fig_subj_comp.update_layout(
                            title='Subjectivity Score Distribution Comparison',
                            xaxis_title='Subjectivity Score',
                            yaxis_title='Frequency',
                            barmode='overlay',
                            height=400
                        )
                        st.plotly_chart(fig_subj_comp, use_container_width=True)
                        st.caption("📊 **Analysis**: This overlapping histogram compares subjectivity scores between abstracts and retraction notices. It reveals differences in how opinionated vs factual each text type is.")
                    
                    with col2:
                        # Sentiment vs Subjectivity Comparison
                        fig_sent_subj_comp = go.Figure()
                        
                        fig_sent_subj_comp.add_trace(go.Scatter(
                            x=abstract_df['subjectivity_score'].dropna().head(3000),
                            y=abstract_df['sentiment_score'].dropna().head(3000),
                            mode='markers',
                            name='Abstract',
                            marker=dict(color='#3498db', opacity=0.5, size=4)
                        ))
                        
                        fig_sent_subj_comp.add_trace(go.Scatter(
                            x=retraction_df['subjectivity_score'].dropna().head(3000),
                            y=retraction_df['sentiment_score'].dropna().head(3000),
                            mode='markers',
                            name='Retraction',
                            marker=dict(color='#e74c3c', opacity=0.5, size=4)
                        ))
                        
                        fig_sent_subj_comp.update_layout(
                            title='Sentiment vs Subjectivity Comparison',
                            xaxis_title='Subjectivity Score',
                            yaxis_title='Sentiment Score',
                            height=400
                        )
                        st.plotly_chart(fig_sent_subj_comp, use_container_width=True)
                        st.caption("📊 **Analysis**: This scatter plot compares the relationship between subjectivity and sentiment in abstracts vs retraction notices. It shows if more subjective writing correlates differently with emotional tone in each text type.")
            
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
                st.caption("📊 **Analysis**: This box plot compares information density (noun density) between abstracts and retraction notices. It shows how informationally dense each text type is, revealing differences in how much factual content is packed into the writing.")
            
            with sub_tab6:
                st.subheader("Cross Consistency Analysis")
                st.markdown("**Goal**: Compare semantic and structural consistency between abstracts and retraction notices.")
                
                if 'consistency' in data:
                    consistency_df = data['consistency']
                    valid_df = consistency_df[
                        (consistency_df['overall_consistency'] > 0) | 
                        (consistency_df['tfidf_similarity'] > 0) |
                        (consistency_df['jaccard_similarity'] > 0)
                    ].copy()
                    
                    if len(valid_df) > 0:
                        # 1. Semantic Consistency: TF-IDF vs Jaccard
                        st.markdown("### Semantic Consistency: TF-IDF vs Jaccard Similarity")
                        fig_semantic = px.scatter(
                            valid_df,
                            x='tfidf_similarity',
                            y='jaccard_similarity',
                            title='Semantic Consistency: TF-IDF Similarity vs Jaccard Similarity',
                            labels={
                                'tfidf_similarity': 'TF-IDF Cosine Similarity',
                                'jaccard_similarity': 'Jaccard Similarity (Keyword Overlap)'
                            },
                            color='overall_consistency',
                            color_continuous_scale='Viridis',
                            hover_data=['record_id'],
                            size_max=10
                        )
                        fig_semantic.update_layout(height=600)
                        st.plotly_chart(fig_semantic, use_container_width=True)
                        st.caption("📊 **Analysis**: This scatter plot compares TF-IDF similarity (cosine similarity) with Jaccard similarity (keyword overlap) between abstracts and retraction notices. Points in the upper-right corner represent samples with high semantic consistency, while points in the lower-left represent samples with large semantic differences.")
                        
                        # 2. Overall Consistency Distribution
                        st.markdown("### Overall Consistency Score Distribution")
                        # Classify consistency levels
                        valid_df['consistency_level'] = pd.cut(
                            valid_df['overall_consistency'],
                            bins=[0, 0.3, 0.6, 1.0],
                            labels=['Low', 'Medium', 'High']
                        )
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig_overall = px.histogram(
                                valid_df,
                                x='overall_consistency',
                                nbins=50,
                                color='consistency_level',
                                title='Overall Consistency Score Distribution',
                                labels={
                                    'overall_consistency': 'Overall Consistency Score',
                                    'count': 'Frequency'
                                },
                                color_discrete_map={
                                    'Low': '#e74c3c',
                                    'Medium': '#f39c12',
                                    'High': '#2ecc71'
                                }
                            )
                            fig_overall.update_layout(height=500)
                            st.plotly_chart(fig_overall, use_container_width=True)
                            st.caption("📊 **Analysis**: This histogram displays the distribution of overall consistency scores (composed of semantic similarity + keyword intersection + entity matching) for all records. Colors indicate consistency levels: red (low), yellow (medium), green (high).")
                        
                        with col2:
                            low_count = (valid_df['consistency_level'] == 'Low').sum()
                            med_count = (valid_df['consistency_level'] == 'Medium').sum()
                            high_count = (valid_df['consistency_level'] == 'High').sum()
                            total = len(valid_df)
                            
                            st.metric("Low Consistency", f"{low_count:,}", f"{(low_count/total*100):.1f}%")
                            st.metric("Medium Consistency", f"{med_count:,}", f"{(med_count/total*100):.1f}%")
                            st.metric("High Consistency", f"{high_count:,}", f"{(high_count/total*100):.1f}%")
                            st.metric("Mean Consistency", f"{valid_df['overall_consistency'].mean():.3f}")
                        
                        # 3. Tone Shift Analysis
                        st.markdown("### Tone Shift Analysis")
                        fig_tone_shift = go.Figure()
                        
                        fig_tone_shift.add_trace(go.Box(
                            y=valid_df['sentiment_shift'],
                            name='Δ Sentiment Score',
                            marker_color='#3498db',
                            boxmean='sd'
                        ))
                        
                        fig_tone_shift.add_trace(go.Box(
                            y=valid_df['assertive_shift'],
                            name='Δ Assertive Words',
                            marker_color='#e74c3c',
                            boxmean='sd'
                        ))
                        
                        fig_tone_shift.add_trace(go.Box(
                            y=valid_df['hedging_shift'],
                            name='Δ Hedging Words',
                            marker_color='#9b59b6',
                            boxmean='sd'
                        ))
                        
                        fig_tone_shift.update_layout(
                            title='Tone Shift Analysis: Changes from Abstract to Retraction',
                            yaxis_title='Shift Value (Retraction - Abstract)',
                            height=500
                        )
                        st.plotly_chart(fig_tone_shift, use_container_width=True)
                        st.caption("📊 **Analysis**: This box plot shows three tone shift indicators: Δ Sentiment Score, Δ Assertive Words, and Δ Hedging Words (all calculated as Retraction - Abstract). Box positions shifted to negative values indicate that retraction notices are more negative and cautious compared to abstracts.")
                        
                        # 4. Entity Consistency Heatmap
                        st.markdown("### Entity Consistency Heatmap")
                        # Create bins for heatmap
                        valid_df['tfidf_bin'] = pd.cut(
                            valid_df['tfidf_similarity'],
                            bins=10,
                            labels=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
                        )
                        valid_df['entity_bin'] = pd.cut(
                            valid_df['entity_match_rate'],
                            bins=10,
                            labels=[f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
                        )
                        
                        heatmap_data = valid_df.groupby(['tfidf_bin', 'entity_bin']).size().reset_index(name='count')
                        
                        if len(heatmap_data) > 0 and heatmap_data['count'].sum() > 0:
                            pivot_table = heatmap_data.pivot(index='entity_bin', columns='tfidf_bin', values='count').fillna(0)
                            
                            fig_heatmap = px.imshow(
                                pivot_table.values,
                                labels=dict(
                                    x="TF-IDF Similarity (Binned)",
                                    y="Entity Match Rate (Binned)",
                                    color="Count"
                                ),
                                title='Semantic Similarity × Entity Match Rate Heatmap',
                                color_continuous_scale='Viridis',
                                x=pivot_table.columns,
                                y=pivot_table.index
                            )
                            fig_heatmap.update_layout(height=600)
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                            st.caption("📊 **Analysis**: This heatmap shows the relationship between TF-IDF similarity (x-axis) and entity match rate (y-axis). The upper-right cells represent samples with both high semantic similarity and high entity matching, indicating the most consistent samples. Darker colors indicate more samples.")
                        else:
                            st.info("Insufficient data to create heatmap. Need more variation in TF-IDF similarity and entity match rates.")
                        
                        # Summary Statistics
                        st.markdown("### Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Mean TF-IDF Similarity", f"{valid_df['tfidf_similarity'].mean():.3f}")
                        with col2:
                            st.metric("Mean Jaccard Similarity", f"{valid_df['jaccard_similarity'].mean():.3f}")
                        with col3:
                            st.metric("Mean Entity Match Rate", f"{valid_df['entity_match_rate'].mean():.3f}")
                        with col4:
                            st.metric("Mean Overall Consistency", f"{valid_df['overall_consistency'].mean():.3f}")
                    else:
                        st.warning("No valid consistency data available for cross comparison.")
                else:
                    st.warning("Consistency analysis data not available.")
        else:
            st.warning("Both abstract and retraction data are required for comparison.")
else:
    st.info("👈 Please load data using the sidebar to view visualizations.")

