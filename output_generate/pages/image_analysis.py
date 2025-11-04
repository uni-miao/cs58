"""
Image Analysis Dashboard Page

This page provides interactive visualizations for DOI image analysis data.
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

st.title("📊 Image Analysis Dashboard")

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
default_csv = os.path.join(datasets_dir, "enhanced_doi_analysis_v3_summary.csv")

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
    
    # Define key analysis columns for image analysis
    numeric_cols = [
        'total_images', 'unique_urls', 'unique_domains',
        'forensics_suspicious', 'reverse_search_suspicious', 
        'phash_duplicates', 'multi_source_reuse',
        'exif_editing_software', 'high_ela_images',
        'avg_authenticity_score'
    ]
    
    # Filter to only columns that exist
    available_numeric_cols = [col for col in numeric_cols if col in df_results.columns]
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Image Analysis", "Suspicious Patterns", 
        "Correlation Analysis", "DOI Comparison", "Data Table"
    ])
    
    with tab1:
        st.header("Overview")
        
        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)
        if 'total_images' in df_results.columns:
            col1.metric("Total Images", f"{df_results['total_images'].sum():,}")
            col2.metric("Total DOIs", f"{len(df_results):,}")
        if 'avg_authenticity_score' in df_results.columns:
            col3.metric("Avg Authenticity", f"{df_results['avg_authenticity_score'].mean():.3f}")
        if 'forensics_suspicious' in df_results.columns:
            col4.metric("Total Suspicious", f"{df_results['forensics_suspicious'].sum():,}")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        if available_numeric_cols:
            st.dataframe(df_results[available_numeric_cols].describe())
        
        # Image count distribution
        if 'total_images' in df_results.columns:
            st.subheader("Image Count Distribution")
            fig = px.histogram(
                df_results,
                x='total_images',
                nbins=50,
                title="Distribution of Images per DOI",
                labels={'total_images': 'Number of Images', 'count': 'Number of DOIs'}
            )
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Image Analysis")
        
        # Analysis dimension selection
        selected_metric = st.selectbox(
            "Select Metric",
            available_numeric_cols,
            index=0 if available_numeric_cols else None
        )
        
        if selected_metric:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                hist_fig = px.histogram(
                    df_results, 
                    x=selected_metric,
                    nbins=30,
                    marginal="rug",
                    title=f"Distribution of {selected_metric.replace('_', ' ').title()}"
                )
                hist_fig.update_layout(bargap=0.1)
                st.plotly_chart(hist_fig, use_container_width=True)
            
            with col2:
                # Box plot
                box_fig = px.box(
                    df_results,
                    y=selected_metric,
                    points="outliers",
                    title=f"Box Plot of {selected_metric.replace('_', ' ').title()}"
                )
                st.plotly_chart(box_fig, use_container_width=True)
        
        # Scatter plot: Images vs Authenticity
        if 'total_images' in df_results.columns and 'avg_authenticity_score' in df_results.columns:
            st.subheader("Images vs Authenticity Score")
            fig = px.scatter(
                df_results,
                x='total_images',
                y='avg_authenticity_score',
                size='total_images',
                hover_data=['doi'],
                trendline="ols",
                title="Relationship Between Image Count and Authenticity Score",
                labels={
                    'total_images': 'Total Images',
                    'avg_authenticity_score': 'Average Authenticity Score'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Suspicious Patterns")
        
        # Show suspicious pattern statistics
        if 'forensics_suspicious' in df_results.columns:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Forensics Suspicious", f"{df_results['forensics_suspicious'].sum():,}")
            if 'phash_duplicates' in df_results.columns:
                with col2:
                    st.metric("Total pHash Duplicates", f"{df_results['phash_duplicates'].sum():,}")
            if 'multi_source_reuse' in df_results.columns:
                with col3:
                    st.metric("Total Multi-Source Reuse", f"{df_results['multi_source_reuse'].sum():,}")
        
        # Suspicious patterns distribution
        if 'forensics_suspicious' in df_results.columns and 'total_images' in df_results.columns:
            st.subheader("Suspicious Pattern Rate")
            df_results['suspicious_rate'] = df_results['forensics_suspicious'] / df_results['total_images']
            fig = px.scatter(
                df_results,
                x='total_images',
                y='suspicious_rate',
                color='forensics_suspicious',
                size='total_images',
                hover_data=['doi'],
                title="Suspicious Pattern Rate vs Total Images",
                labels={
                    'total_images': 'Total Images',
                    'suspicious_rate': 'Suspicious Rate',
                    'forensics_suspicious': 'Suspicious Count'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top suspicious DOIs
        if 'forensics_suspicious' in df_results.columns:
            st.subheader("Top 20 DOIs by Suspicious Count")
            top_suspicious = df_results.nlargest(20, 'forensics_suspicious')[['doi', 'forensics_suspicious', 'total_images', 'avg_authenticity_score']]
            fig = px.bar(
                top_suspicious,
                x='doi',
                y='forensics_suspicious',
                title="Top 20 DOIs with Highest Suspicious Patterns",
                labels={'doi': 'DOI', 'forensics_suspicious': 'Suspicious Count'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top_suspicious)
    
    with tab4:
        st.header("Correlation Analysis")
        
        if len(available_numeric_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = df_results[available_numeric_cols].corr()
            
            # Correlation heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                title="Correlation Matrix of Image Analysis Metrics",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display correlation statistics
            with st.expander("View Detailed Correlation Statistics"):
                st.dataframe(corr_matrix)
        else:
            st.info("Not enough numeric columns for correlation analysis.")
    
    with tab5:
        st.header("DOI Comparison")
        
        # Filter by DOI
        if len(df_results) > 0 and 'doi' in df_results.columns:
            all_dois = df_results['doi'].tolist()
            selected_dois = st.multiselect(
                "Select DOIs to Compare",
                options=all_dois,
                default=all_dois[:min(5, len(all_dois))]
            )
            
            if selected_dois:
                filtered_df = df_results[df_results['doi'].isin(selected_dois)]
                
                # Select metrics to compare
                comparison_cols = st.multiselect(
                    "Select Metrics to Compare",
                    options=available_numeric_cols,
                    default=available_numeric_cols[:min(5, len(available_numeric_cols))] if available_numeric_cols else []
                )
                
                if comparison_cols:
                    # Bar chart comparison
                    fig = go.Figure()
                    
                    for metric in comparison_cols:
                        fig.add_trace(go.Bar(
                            x=filtered_df['doi'],
                            y=filtered_df[metric],
                            name=metric.replace('_', ' ').title()
                        ))
                    
                    fig.update_layout(
                        title="Metric Comparison Across Selected DOIs",
                        xaxis_title="DOI",
                        yaxis_title="Value",
                        barmode='group'
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Radar chart for comparison
                    if len(comparison_cols) > 2:
                        categories = comparison_cols
                        fig = go.Figure()
                        
                        for doi in selected_dois:
                            doc_data = filtered_df[filtered_df['doi'] == doi]
                            if not doc_data.empty:
                                values = [doc_data[cat].iloc[0] for cat in categories]
                                fig.add_trace(go.Scatterpolar(
                                    r=values,
                                    theta=categories,
                                    fill='toself',
                                    name=doi[:50]
                                ))
                        
                        fig.update_layout(
                            title="Radar Chart of Selected Metrics",
                            polar=dict(
                                radialaxis=dict(visible=True)
                            ),
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display comparison table
                    st.subheader("Comparison Table")
                    display_cols = ['doi'] + comparison_cols
                    st.dataframe(filtered_df[display_cols])
    
    with tab6:
        st.header("Data Table")
        
        # Show full data table
        st.dataframe(df_results, use_container_width=True, height=400)
        
        # Download option
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="enhanced_doi_analysis_summary.csv",
            mime="text/csv"
        )

else:
    st.info("Click 'Load Data' in the sidebar to start analyzing your data.")
    
    # Show instructions
    st.header("Instructions")
    st.write("""
    This dashboard provides an interactive interface for analyzing DOI image analysis data.
    
    ### How to use:
    1. **Auto-load mode (default)**: Data loads automatically when page opens
       - If cache exists and is valid, loads instantly from cache
       - If cache is missing or outdated, loads from source and creates cache
       - Uncheck "Auto-load on startup" to disable automatic loading
    
    2. **Manual load mode**: Uncheck "Auto-load on startup"
       - Click "Load Data" button to load data manually
       - Cache is created automatically after first load
    
    3. **Precompute cache** (optional): Run `python cache_utils.py [csv_path]` 
       to pre-generate cache files for faster loading
    
    4. **Explore results** through the various tabs:
       - **Overview**: General statistics and summary metrics
       - **Image Analysis**: Distribution and analysis of image metrics
       - **Suspicious Patterns**: Analysis of suspicious image patterns
       - **Correlation Analysis**: Relationships between different metrics
       - **DOI Comparison**: Compare selected DOIs side by side
       - **Data Table**: View and download raw data
    
    ### Cache Features:
    - **Automatic caching**: Data is automatically cached after loading
    - **Smart validation**: Cache automatically invalidates when source file changes
    - **Fast loading**: Cached data loads instantly (no CSV parsing delay)
    - **Cache management**: Use "Cache Management" section to clear caches
    
    ### Data format:
    The input CSV file should contain columns for:
    - `doi`: DOI identifier
    - `total_images`: Total number of images
    - `forensics_suspicious`: Count of suspicious forensic patterns
    - `avg_authenticity_score`: Average authenticity score
    - And other analysis metrics
    """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 DOI Image Analysis Dashboard")

