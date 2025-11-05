"""
Altmetric Source Analysis Dashboard Page

This page provides interactive visualizations for paper-domain data source analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
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

# Try to import matplotlib_venn for Venn diagrams
try:
    from matplotlib_venn import venn2
    HAS_VENN = True
except ImportError:
    HAS_VENN = False

# Page config is set by parent interactive_dashboard.py
# No need to set it here to avoid conflicts

st.title("📈 Altmetric Source Analysis Dashboard")

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
# Updated to use altmetric_source_data subfolder
altmetric_data_dir = os.path.join(datasets_dir, "altmetric_source_data")
default_csv = os.path.join(altmetric_data_dir, "paper_level_summary.csv")

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

# Load additional dataset if available (datasets_dir already points to parent/datasets)
# Updated to use altmetric_source_data subfolder (altmetric_data_dir already defined above)
paper_domain_pairs_path = os.path.join(altmetric_data_dir, "paper_domain_pairs.csv")
df_paper_domain_pairs = None
if os.path.exists(paper_domain_pairs_path):
    try:
        df_paper_domain_pairs = pd.read_csv(paper_domain_pairs_path)
    except:
        pass

# Load paper_level_summary to get DOI information
paper_level_summary_path = os.path.join(altmetric_data_dir, "paper_level_summary.csv")
df_paper_summary = None
if os.path.exists(paper_level_summary_path):
    try:
        df_paper_summary = pd.read_csv(paper_level_summary_path)
    except:
        pass

# Load temporal mentions dataset for veracity timeline sampling
temporal_mentions_path = os.path.join(altmetric_data_dir, "temporal_mentions_joined.csv")
temporal_mentions_df = None
if os.path.exists(temporal_mentions_path):
    try:
        temporal_mentions_df = pd.read_csv(temporal_mentions_path)
        for date_col in ["OriginalDate_iso", "RetractionDate_iso", "MentionDate_iso"]:
            if date_col in temporal_mentions_df.columns:
                temporal_mentions_df[date_col] = pd.to_datetime(
                    temporal_mentions_df[date_col], errors="coerce"
                )
    except Exception as exc:
        st.sidebar.warning(
            f"Unable to load temporal_mentions_joined.csv: {exc}"[:200]
        )
        temporal_mentions_df = None

# Initialize session state for temporal veracity sampling
if "temporal_veracity_true_sample" not in st.session_state:
    st.session_state["temporal_veracity_true_sample"] = None
if "temporal_veracity_false_sample" not in st.session_state:
    st.session_state["temporal_veracity_false_sample"] = None

# Temporal veracity label configuration
TRUE_LABEL_OPTIONS = [
    "ANY",
    "O_BEFORE",
    "R_AFTER",
    "R_BEFORE_COMENTION",
    "R_BEFORE_EXCL_NORM",
    "R_BEFORE_EXCL_ABNORM",
]
FALSE_LABEL_OPTIONS = [
    "ANY",
    "O_AFTER_COMENTION",
    "O_AFTER_EXCL_NORM",
    "O_AFTER_EXCL_ABNORM",
]
TRUE_LABEL_SET = TRUE_LABEL_OPTIONS[1:]
FALSE_LABEL_SET = FALSE_LABEL_OPTIONS[1:]

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
    
    # Detect which dataset is loaded
    is_paper_level = 'Original_Domains_Count' in df_results.columns
    is_paper_domain_pairs = 'Domain' in df_results.columns and 'Has_Original' in df_results.columns
    
    # Create tabs for Data Source Analysis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Domain Analysis", "Paper-Domain Relationships",
        "Overlap Analysis", "Data Table", "Temporal Veracity (Timeline)"
    ])
    
    with tab1:
        st.header("Data Source Analysis Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        if is_paper_level:
            col1.metric("Total Papers", f"{len(df_results):,}")
            col2.metric("Avg Original Domains", f"{df_results['Original_Domains_Count'].mean():.2f}")
            col3.metric("Avg Retraction Domains", f"{df_results['Retraction_Domains_Count'].mean():.2f}")
            col4.metric("Papers with Overlap", f"{df_results['Overlap_Domains_Count'].gt(0).sum():,}")
        
        if df_paper_domain_pairs is not None:
            st.subheader("Paper-Domain Pairs Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Pairs", f"{len(df_paper_domain_pairs):,}")
            col2.metric("Unique Domains", f"{df_paper_domain_pairs['Domain'].nunique():,}")
            col3.metric("Unique Papers", f"{df_paper_domain_pairs['Record_ID'].nunique():,}")
            
            # Evidence types pie chart (similar to image)
            st.subheader("Evidence Types in Source Analysis")
            evidence_counts = {
                'Multiple Contexts': len(df_paper_domain_pairs[df_paper_domain_pairs['Has_Overlap'] == True]),
                'Original Only': len(df_paper_domain_pairs[df_paper_domain_pairs['Mention_Type'] == 'Original Only']),
                'Retraction Only': len(df_paper_domain_pairs[df_paper_domain_pairs['Mention_Type'] == 'Retraction Only']),
                'Both Present': len(df_paper_domain_pairs[(df_paper_domain_pairs['Has_Original'] == True) & 
                                                           (df_paper_domain_pairs['Has_Retraction'] == True)])
            }
            
            fig_pie = px.pie(
                values=list(evidence_counts.values()),
                names=list(evidence_counts.keys()),
                title="Evidence Types Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Source category distribution
            if 'Source_Category' in df_paper_domain_pairs.columns:
                st.subheader("Source Categories")
                category_counts = df_paper_domain_pairs['Source_Category'].value_counts()
                fig_cat = px.bar(
                    x=category_counts.values,
                    y=category_counts.index,
                    orientation='h',
                    title="Source Categories Distribution",
                    labels={'x': 'Count', 'y': 'Category'}
                )
                st.plotly_chart(fig_cat, use_container_width=True)
    
    with tab2:
        st.header("Domain Analysis")
        
        if df_paper_domain_pairs is not None:
            # Top domains analysis
            domain_stats = df_paper_domain_pairs.groupby('Domain').agg({
                'Record_ID': 'count',
                'Has_Original': 'sum',
                'Has_Retraction': 'sum',
                'Has_Overlap': 'sum'
            }).reset_index()
            domain_stats.columns = ['Domain', 'Total_Mentions', 'Original_Count', 'Retraction_Count', 'Overlap_Count']
            domain_stats = domain_stats.sort_values('Total_Mentions', ascending=False)
            
            # Top 20 domains
            st.subheader("Top 20 Domains by Mention Count")
            top_domains = domain_stats.head(20)
            fig_domains = px.bar(
                top_domains,
                x='Total_Mentions',
                y='Domain',
                orientation='h',
                title="Top 20 Domains with Most Mentions",
                labels={'Total_Mentions': 'Number of Mentions', 'Domain': 'Domain'}
            )
            fig_domains.update_layout(height=500)
            st.plotly_chart(fig_domains, use_container_width=True)
            
            # Domain type distribution
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Domains by Original Mentions")
                fig_orig = px.bar(
                    top_domains.head(10),
                    x='Original_Count',
                    y='Domain',
                    orientation='h',
                    title="Top 10 Domains - Original Mentions",
                    labels={'Original_Count': 'Original Mentions'}
                )
                st.plotly_chart(fig_orig, use_container_width=True)
            
            with col2:
                st.subheader("Domains by Retraction Mentions")
                fig_ret = px.bar(
                    top_domains.head(10),
                    x='Retraction_Count',
                    y='Domain',
                    orientation='h',
                    title="Top 10 Domains - Retraction Mentions",
                    labels={'Retraction_Count': 'Retraction Mentions'}
                )
                st.plotly_chart(fig_ret, use_container_width=True)
            
            # Display domain statistics table
            st.subheader("Domain Statistics")
            st.dataframe(domain_stats.head(30), use_container_width=True)
    
    with tab3:
        st.header("Paper-Domain Relationships")
        
        if df_paper_domain_pairs is not None:
            # Select a paper to analyze
            unique_papers = df_paper_domain_pairs.groupby('Record_ID')['Title'].first().reset_index()
            unique_papers['Paper_Label'] = unique_papers.apply(
                lambda x: f"{x['Record_ID']}: {x['Title'][:60]}...", axis=1
            )
            
            selected_paper_id = st.selectbox(
                "Select Paper (Record ID)",
                options=unique_papers['Record_ID'].tolist(),
                format_func=lambda x: unique_papers[unique_papers['Record_ID'] == x]['Paper_Label'].iloc[0]
            )
            
            if selected_paper_id:
                paper_data = df_paper_domain_pairs[df_paper_domain_pairs['Record_ID'] == selected_paper_id]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Domain Mentions", len(paper_data))
                with col2:
                    st.metric("Original Domains", paper_data['Has_Original'].sum())
                with col3:
                    st.metric("Retraction Domains", paper_data['Has_Retraction'].sum())
                
                # Domains for this paper
                st.subheader("Domain Mentions for Selected Paper")
                display_cols = ['Domain', 'Has_Original', 'Has_Retraction', 'Has_Overlap', 
                               'Source_Category', 'Mention_Type']
                st.dataframe(paper_data[display_cols], use_container_width=True)
            
            # Papers per domain heatmap (top domains)
            if 'df_paper_domain_pairs' in locals() and df_paper_domain_pairs is not None:
                domain_stats = df_paper_domain_pairs.groupby('Domain').size().sort_values(ascending=False)
                top_20_domains = domain_stats.head(20).index.tolist()
                
                st.subheader("Papers per Domain (Top 20 Domains)")
                papers_by_domain = df_paper_domain_pairs[df_paper_domain_pairs['Domain'].isin(top_20_domains)]
                paper_domain_matrix = pd.crosstab(
                    papers_by_domain['Record_ID'], 
                    papers_by_domain['Domain']
                )
                
                if len(paper_domain_matrix) > 0:
                    fig_heatmap = px.imshow(
                        paper_domain_matrix.values[:50],  # Limit to first 50 papers
                        labels=dict(x="Domain", y="Paper", color="Mention Count"),
                        x=paper_domain_matrix.columns,
                        y=[f"Paper {i}" for i in range(min(50, len(paper_domain_matrix)))],
                        aspect="auto",
                        title="Paper-Domain Mention Matrix (Top 50 Papers)"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab4:
        st.header("Overlap Analysis")
        
        # Paper-Domain Pairs Overlap (Venn Diagram)
        if df_paper_domain_pairs is not None:
            st.subheader("Paper-Domain Pairs Overlap")
            st.write("*Same Paper × Same Media: Original vs Retraction*")
            
            # Calculate overlap statistics
            original_only_pairs = len(df_paper_domain_pairs[
                (df_paper_domain_pairs['Has_Original'] == True) & 
                (df_paper_domain_pairs['Has_Retraction'] == False)
            ])
            retraction_only_pairs = len(df_paper_domain_pairs[
                (df_paper_domain_pairs['Has_Original'] == False) & 
                (df_paper_domain_pairs['Has_Retraction'] == True)
            ])
            overlap_pairs = len(df_paper_domain_pairs[
                (df_paper_domain_pairs['Has_Original'] == True) & 
                (df_paper_domain_pairs['Has_Retraction'] == True)
            ])
            
            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Only", f"{original_only_pairs:,}")
            with col2:
                st.metric("Retraction Only", f"{retraction_only_pairs:,}")
            with col3:
                st.metric("Both", f"{overlap_pairs:,}")
            
            # Create Venn diagram
            if HAS_VENN:
                fig_venn, ax = plt.subplots(figsize=(10, 8))
                venn2(subsets=(original_only_pairs, retraction_only_pairs, overlap_pairs),
                      set_labels=('Original Only', 'Retraction Only'),
                      set_colors=('lightcoral', 'lightblue'),
                      alpha=0.7,
                      ax=ax)
                ax.set_title('Paper-Domain Pairs Overlap\nSame Paper × Same Media: Original vs Retraction',
                           fontsize=16, fontweight='bold', pad=20)
                # Add text annotations
                ax.text(0.5, 0.95, f'Original Only: {original_only_pairs:,}',
                       transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold')
                ax.text(0.5, 0.05, f'Retraction Only: {retraction_only_pairs:,}',
                       transform=ax.transAxes, ha='center', va='bottom', fontsize=12, fontweight='bold')
                ax.text(0.5, 0.5, f'Both: {overlap_pairs:,}',
                       transform=ax.transAxes, ha='center', va='center', fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_venn)
                plt.close(fig_venn)
            else:
                # Fallback: Create Venn-like visualization using Plotly
                st.info("💡 Install matplotlib-venn for Venn diagram visualization. Showing statistics above.")
            
            # DOI-specific Analysis Section
            st.markdown("---")
            st.subheader("DOI-Specific Distribution Analysis")
            st.write("*Select a DOI to view its data distribution*")
            
            # Get unique Record_IDs (DOIs) from paper_domain_pairs
            if 'Record_ID' in df_paper_domain_pairs.columns:
                unique_record_ids = sorted(df_paper_domain_pairs['Record_ID'].unique().tolist())
                
                # Create a mapping from Record_ID to DOI if available
                record_id_to_doi = {}
                if df_paper_summary is not None and 'Record_ID' in df_paper_summary.columns:
                    # Try to find DOI columns (could be OriginalPaperDOI, RetractionDOI, or just DOI)
                    doi_columns = [col for col in df_paper_summary.columns if 'DOI' in col.upper()]
                    if doi_columns:
                        for _, row in df_paper_summary.iterrows():
                            record_id = row['Record_ID']
                            # Try to get any available DOI
                            doi_value = None
                            for doi_col in doi_columns:
                                if pd.notna(row[doi_col]) and str(row[doi_col]).strip():
                                    doi_value = str(row[doi_col]).strip()
                                    break
                            if doi_value:
                                record_id_to_doi[record_id] = doi_value
                
                # Create display format for selectbox: "ID: X, DOI: Y" or just "ID: X"
                def format_record_id_option(record_id):
                    if record_id in record_id_to_doi:
                        return f"ID: {record_id}, DOI: {record_id_to_doi[record_id]}"
                    else:
                        return f"ID: {record_id}"
                
                # Create DOI selector with formatted options
                formatted_options = {format_record_id_option(rid): rid for rid in unique_record_ids}
                selected_option = st.selectbox(
                    "Select DOI (Record ID)",
                    options=list(formatted_options.keys()),
                    index=0,
                    help="Choose a DOI to analyze its domain distribution"
                )
                selected_doi = formatted_options[selected_option]
                
                if selected_doi:
                    # Filter data for selected DOI
                    doi_data = df_paper_domain_pairs[df_paper_domain_pairs['Record_ID'] == selected_doi]
                    
                    # Get DOI string for display
                    actual_doi = record_id_to_doi.get(selected_doi, "N/A")
                    display_label = f"ID: {selected_doi}"
                    if actual_doi != "N/A":
                        display_label += f", DOI: {actual_doi}"
                    
                    # Display DOI information prominently
                    st.info(f"**Selected:** {display_label}")
                    
                    # Calculate statistics for this DOI
                    doi_original_only = len(doi_data[
                        (doi_data['Has_Original'] == True) & 
                        (doi_data['Has_Retraction'] == False)
                    ])
                    doi_retraction_only = len(doi_data[
                        (doi_data['Has_Original'] == False) & 
                        (doi_data['Has_Retraction'] == True)
                    ])
                    doi_both = len(doi_data[
                        (doi_data['Has_Original'] == True) & 
                        (doi_data['Has_Retraction'] == True)
                    ])
                    doi_total = len(doi_data)
                    
                    # Display metrics for selected DOI
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Domains", f"{doi_total:,}")
                    with col2:
                        st.metric("Original Only", f"{doi_original_only:,}")
                    with col3:
                        st.metric("Retraction Only", f"{doi_retraction_only:,}")
                    with col4:
                        st.metric("Both", f"{doi_both:,}")
                    
                    # Create visualizations for selected DOI
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        # Pie chart showing distribution
                        if doi_total > 0:
                            doi_distribution = {
                                'Original Only': doi_original_only,
                                'Retraction Only': doi_retraction_only,
                                'Both': doi_both
                            }
                            # Remove zero values
                            doi_distribution = {k: v for k, v in doi_distribution.items() if v > 0}
                            
                            if doi_distribution:
                                fig_doi_pie = px.pie(
                                    values=list(doi_distribution.values()),
                                    names=list(doi_distribution.keys()),
                                    title=f"Domain Distribution for {display_label}",
                                    color_discrete_map={
                                        'Original Only': 'lightcoral',
                                        'Retraction Only': 'lightblue',
                                        'Both': 'lightgreen'
                                    }
                                )
                                st.plotly_chart(fig_doi_pie, use_container_width=True)
                            
                            # Bar chart comparison
                            fig_doi_bar = px.bar(
                                x=list(doi_distribution.keys()),
                                y=list(doi_distribution.values()),
                                title=f"Domain Counts for {display_label}",
                                labels={'x': 'Category', 'y': 'Count'},
                                color=list(doi_distribution.keys()),
                                color_discrete_map={
                                    'Original Only': 'lightcoral',
                                    'Retraction Only': 'lightblue',
                                    'Both': 'lightgreen'
                                }
                            )
                            st.plotly_chart(fig_doi_bar, use_container_width=True)
                    
                    with col_viz2:
                        # Display domain list for selected DOI
                        st.write(f"**Domains for {display_label}**")
                        
                        # Create expandable sections for different categories
                        if doi_original_only > 0:
                            with st.expander(f"Original Only Domains ({doi_original_only})", expanded=True):
                                original_only_domains = doi_data[
                                    (doi_data['Has_Original'] == True) & 
                                    (doi_data['Has_Retraction'] == False)
                                ][['Domain', 'Source_Category']].drop_duplicates()
                                st.dataframe(original_only_domains, use_container_width=True, hide_index=True)
                        
                        if doi_retraction_only > 0:
                            with st.expander(f"Retraction Only Domains ({doi_retraction_only})", expanded=True):
                                retraction_only_domains = doi_data[
                                    (doi_data['Has_Original'] == False) & 
                                    (doi_data['Has_Retraction'] == True)
                                ][['Domain', 'Source_Category']].drop_duplicates()
                                st.dataframe(retraction_only_domains, use_container_width=True, hide_index=True)
                        
                        if doi_both > 0:
                            with st.expander(f"Overlapping Domains - Both ({doi_both})", expanded=True):
                                both_domains = doi_data[
                                    (doi_data['Has_Original'] == True) & 
                                    (doi_data['Has_Retraction'] == True)
                                ][['Domain', 'Source_Category']].drop_duplicates()
                                st.dataframe(both_domains, use_container_width=True, hide_index=True)
                        
                        # Show paper title if available
                        if 'Title' in doi_data.columns and not doi_data.empty:
                            title = doi_data['Title'].iloc[0]
                            if pd.notna(title):
                                st.info(f"**Paper Title:** {title}")
                    
                    # Create detailed breakdown table
                    st.subheader(f"Detailed Domain Data for {display_label}")
                    display_cols = ['Domain', 'Has_Original', 'Has_Retraction', 'Has_Overlap', 
                                   'Mention_Type', 'Source_Category']
                    available_cols = [col for col in display_cols if col in doi_data.columns]
                    st.dataframe(doi_data[available_cols], use_container_width=True, height=400)
                    
                    # Source category distribution
                    if 'Source_Category' in doi_data.columns:
                        st.subheader("Source Category Distribution")
                        category_counts = doi_data['Source_Category'].value_counts()
                        fig_category = px.bar(
                            x=category_counts.index,
                            y=category_counts.values,
                            title=f"Source Categories for {display_label}",
                            labels={'x': 'Source Category', 'y': 'Count'},
                            color=category_counts.values,
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_category, use_container_width=True)
            
            # Media Coverage Distribution Heatmap
            st.subheader("Media Coverage Distribution Heatmap")
            st.write("*Normalized Percentages: Top 20 Domains × Mention Types*")
            
            # Get top 20 domains by total mentions
            if 'Domain' in df_paper_domain_pairs.columns and 'Mention_Type' in df_paper_domain_pairs.columns:
                domain_total = df_paper_domain_pairs.groupby('Domain').size().sort_values(ascending=False)
                top_20_domains = domain_total.head(20).index.tolist()
                
                # Filter data for top 20 domains
                top_domains_data = df_paper_domain_pairs[df_paper_domain_pairs['Domain'].isin(top_20_domains)]
                
                # Create cross-tabulation
                mention_type_counts = pd.crosstab(
                    top_domains_data['Domain'],
                    top_domains_data['Mention_Type'],
                    normalize='index'  # Normalize by row (domain)
                )
                
                # Ensure all mention types are present
                mention_types = ['Original Only', 'Retraction Only', 'Both']
                for mt in mention_types:
                    if mt not in mention_type_counts.columns:
                        mention_type_counts[mt] = 0.0
                
                # Reorder columns
                mention_type_counts = mention_type_counts[mention_types]
                
                # Reorder rows by total mentions (descending)
                mention_type_counts = mention_type_counts.reindex(top_20_domains)
                
                # Create heatmap
                fig_heatmap = px.imshow(
                    mention_type_counts.values,
                    labels=dict(x="Mention Type", y="Domain", color="Normalized Percentage"),
                    x=mention_type_counts.columns.tolist(),
                    y=mention_type_counts.index.tolist(),
                    aspect="auto",
                    color_continuous_scale='RdBu_r',  # Reversed: blue (low) to red (high)
                    text_auto='.2f',
                    zmin=0.0,
                    zmax=1.0,
                    title="Media Coverage Distribution Heatmap (Normalized Percentages: Top 20 Domains × Mention Types)"
                )
                fig_heatmap.update_xaxes(side="bottom")
                fig_heatmap.update_layout(
                    height=600,
                    xaxis_title="Mention Type",
                    yaxis_title="Domain",
                    coloraxis_colorbar=dict(title="Normalized<br>Percentage")
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Display data table
                with st.expander("View Raw Data", expanded=False):
                    st.dataframe(mention_type_counts, use_container_width=True)
        
        if is_paper_level:
            # Overlap rate distribution
            st.subheader("Overlap Rate Distribution")
            fig_overlap = px.histogram(
                df_results,
                x='Overlap_Rate',
                nbins=50,
                title="Distribution of Domain Overlap Rates",
                labels={'Overlap_Rate': 'Overlap Rate', 'count': 'Number of Papers'}
            )
            fig_overlap.update_layout(bargap=0.1)
            st.plotly_chart(fig_overlap, use_container_width=True)
            
            # Papers with/without overlap
            overlap_summary = {
                'No Overlap': len(df_results[df_results['Overlap_Domains_Count'] == 0]),
                'Has Overlap': len(df_results[df_results['Overlap_Domains_Count'] > 0])
            }
            
            col1, col2 = st.columns(2)
            with col1:
                fig_overlap_pie = px.pie(
                    values=list(overlap_summary.values()),
                    names=list(overlap_summary.keys()),
                    title="Papers with/without Domain Overlap"
                )
                st.plotly_chart(fig_overlap_pie, use_container_width=True)
            
            with col2:
                st.subheader("Overlap Statistics")
                overlap_stats = pd.DataFrame({
                    'Metric': ['Total Papers', 'Papers with Overlap', 'Avg Overlap Rate', 
                              'Max Overlap Domains', 'Papers with 100% Overlap'],
                    'Value': [
                        len(df_results),
                        df_results['Overlap_Domains_Count'].gt(0).sum(),
                        f"{df_results['Overlap_Rate'].mean():.3f}",
                        df_results['Overlap_Domains_Count'].max(),
                        (df_results['Overlap_Rate'] == 1.0).sum()
                    ]
                })
                st.dataframe(overlap_stats, use_container_width=True, hide_index=True)
            
            # Top papers by overlap
            st.subheader("Top 20 Papers by Overlap Domains")
            top_overlap = df_results.nlargest(20, 'Overlap_Domains_Count')[
                ['Record_ID', 'Title', 'Original_Domains_Count', 'Retraction_Domains_Count', 
                 'Overlap_Domains_Count', 'Overlap_Rate']
            ]
            fig_top_overlap = px.bar(
                top_overlap,
                x='Overlap_Domains_Count',
                y='Record_ID',
                orientation='h',
                title="Papers with Highest Domain Overlap",
                labels={'Overlap_Domains_Count': 'Overlap Domain Count', 'Record_ID': 'Paper Record ID'}
            )
            st.plotly_chart(fig_top_overlap, use_container_width=True)
            st.dataframe(top_overlap, use_container_width=True)
    
    with tab5:
        st.header("Data Table")

        # Show current dataset
        st.subheader("Current Dataset")
        st.dataframe(df_results, use_container_width=True, height=400)

        # Option to load paper_domain_pairs
        if df_paper_domain_pairs is not None and not is_paper_domain_pairs:
            st.subheader("Paper-Domain Pairs Dataset")
            with st.expander("View Paper-Domain Pairs Data", expanded=False):
                st.dataframe(df_paper_domain_pairs, use_container_width=True, height=300)

        # Download option
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="Download Current Data as CSV",
            data=csv,
            file_name="data_source_analysis.csv",
            mime="text/csv"
        )

    with tab6:
        st.header("Temporal Veracity (Timeline)")
        st.write(
            "Draw paired TRUE and FALSE news samples to inspect how publication, retraction, "
            "and mention dates align across the lifecycle of a research output."
        )

        if temporal_mentions_df is None:
            st.warning(
                "The temporal_mentions_joined.csv dataset is not available. Place it in the datasets "
                "directory to enable timeline sampling."
            )
        elif "label" not in temporal_mentions_df.columns:
            st.error(
                "The temporal mentions dataset is missing the required 'label' column. "
                "Please verify the CSV schema."
            )
        else:
            control_col_true, control_col_false = st.columns(2)
            temporal_true_label = control_col_true.selectbox(
                "TRUE Label",
                TRUE_LABEL_OPTIONS,
                index=0,
                key="temporal_true_label_filter",
            )
            temporal_false_label = control_col_false.selectbox(
                "FALSE Label",
                FALSE_LABEL_OPTIONS,
                index=0,
                key="temporal_false_label_filter",
            )

            temporal_seed_text = st.text_input(
                "Seed (optional, integer)",
                value="",
                key="temporal_seed_text",
            )

            button_col_seed, button_col_random = st.columns(2)
            trigger_sample_with_seed = button_col_seed.button("Sample with Seed")
            trigger_sample_random = button_col_random.button("Sample Random")

            true_labels_filter = TRUE_LABEL_SET if temporal_true_label == "ANY" else [temporal_true_label]
            false_labels_filter = (
                FALSE_LABEL_SET if temporal_false_label == "ANY" else [temporal_false_label]
            )

            if trigger_sample_with_seed or trigger_sample_random:
                seed_value = None
                should_sample = False

                if trigger_sample_with_seed:
                    seed_text = temporal_seed_text.strip()
                    if seed_text:
                        try:
                            seed_value = int(seed_text)
                            should_sample = True
                        except ValueError:
                            st.warning("Seed must be an integer to enable deterministic sampling.")
                    else:
                        st.warning("Enter a seed value to reproduce the same samples.")

                if trigger_sample_random:
                    should_sample = True

                if should_sample:
                    true_pool = temporal_mentions_df[
                        temporal_mentions_df["label"].isin(true_labels_filter)
                    ]
                    false_pool = temporal_mentions_df[
                        temporal_mentions_df["label"].isin(false_labels_filter)
                    ]

                    pool_errors = []
                    if true_pool.empty:
                        pool_errors.append("TRUE pool is empty for the selected filter.")
                    if false_pool.empty:
                        pool_errors.append("FALSE pool is empty for the selected filter.")

                    if pool_errors:
                        for msg in pool_errors:
                            st.warning(msg)
                    else:
                        true_random_state = seed_value if seed_value is not None else None
                        false_random_state = (
                            seed_value + 1 if seed_value is not None else None
                        )

                        st.session_state["temporal_veracity_true_sample"] = (
                            true_pool.sample(n=1, random_state=true_random_state)
                            .iloc[0]
                            .copy()
                        )
                        st.session_state["temporal_veracity_false_sample"] = (
                            false_pool.sample(n=1, random_state=false_random_state)
                            .iloc[0]
                            .copy()
                        )
                        st.success("Sampled 1 TRUE and 1 FALSE item.")

            true_sample = st.session_state.get("temporal_veracity_true_sample")
            false_sample = st.session_state.get("temporal_veracity_false_sample")

            if (
                true_sample is not None
                and true_sample.get("label") not in true_labels_filter
            ):
                true_sample = None
                st.session_state["temporal_veracity_true_sample"] = None

            if (
                false_sample is not None
                and false_sample.get("label") not in false_labels_filter
            ):
                false_sample = None
                st.session_state["temporal_veracity_false_sample"] = None

            def format_field(value):
                if isinstance(value, pd.Timestamp):
                    if pd.isna(value):
                        return "—"
                    return value.strftime("%Y-%m-%d")
                if value is None:
                    return "—"
                if pd.isna(value):
                    return "—"
                value_str = str(value).strip()
                return value_str if value_str else "—"

            def build_timeline(sample_series):
                event_fields = [
                    ("Original Publication", "OriginalDate_iso"),
                    ("Retraction", "RetractionDate_iso"),
                    ("Mention", "MentionDate_iso"),
                ]
                timeline_points = []
                for event_name, column in event_fields:
                    if column in sample_series.index:
                        value = sample_series.get(column)
                        if isinstance(value, pd.Timestamp):
                            dt_value = value
                        else:
                            dt_value = pd.to_datetime(value, errors="coerce")
                        if pd.notna(dt_value):
                            timeline_points.append((event_name, dt_value))

                if not timeline_points:
                    return None, []

                timeline_points.sort(key=lambda item: item[1])
                y_positions = [0] * len(timeline_points)
                mode = "markers" if len(timeline_points) == 1 else "lines+markers"
                color_map = {
                    "Original Publication": "#1f77b4",
                    "Retraction": "#ff7f0e",
                    "Mention": "#2ca02c",
                }
                marker_colors = [
                    color_map.get(point[0], "#636EFA") for point in timeline_points
                ]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=[point[1] for point in timeline_points],
                        y=y_positions,
                        mode=mode,
                        marker=dict(size=12, color=marker_colors),
                        line=dict(color="#636EFA"),
                        text=[point[0] for point in timeline_points],
                        hovertemplate="%{text}<br>%{x|%Y-%m-%d}<extra></extra>",
                    )
                )
                fig.update_layout(
                    title="Timeline",
                    showlegend=False,
                    xaxis_title="Date",
                    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    margin=dict(l=40, r=20, t=60, b=40),
                )
                return fig, timeline_points

            def render_sample(column, sample_series, label_name):
                with column:
                    st.subheader(label_name)
                    if sample_series is None:
                        st.info(
                            "Use the sampling controls above to populate this sample."
                        )
                        return

                    st.markdown("**Basic Info**")
                    st.markdown(f"**Title:** {format_field(sample_series.get('Title'))}")
                    st.markdown(f"**Paper Record ID:** {format_field(sample_series.get('Record ID'))}")
                    st.markdown(
                        f"**Mention Title:** {format_field(sample_series.get('Mention Title'))}"
                    )
                    st.markdown(
                        f"**Mention Date:** {format_field(sample_series.get('MentionDate_iso'))}"
                    )
                    mention_url = format_field(sample_series.get('Mention URL'))
                    if mention_url != "—" and mention_url.lower().startswith("http"):
                        st.markdown(f"**Mention URL:** [{mention_url}]({mention_url})")
                    else:
                        st.markdown("**Mention URL:** —")

                    st.markdown("**Paper Identifiers & Dates**")
                    st.markdown(
                        f"**Original DOI:** {format_field(sample_series.get('OriginalPaperDOI'))}"
                    )
                    st.markdown(
                        f"**Original Date:** {format_field(sample_series.get('OriginalDate_iso'))}"
                    )
                    st.markdown(
                        f"**Retraction DOI:** {format_field(sample_series.get('RetractionDOI'))}"
                    )
                    st.markdown(
                        f"**Retraction Date:** {format_field(sample_series.get('RetractionDate_iso'))}"
                    )

                    timeline_fig, timeline_points = build_timeline(sample_series)
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True)
                        if timeline_points:
                            st.markdown("**Event Order**")
                            for event_name, point in timeline_points:
                                st.markdown(
                                    f"- {event_name}: {point.strftime('%Y-%m-%d')}"
                                )
                    else:
                        st.info("No temporal events with valid dates for this sample.")

                    st.markdown("**Label Metadata**")
                    st.markdown(f"**Label:** {format_field(sample_series.get('label'))}")
                    st.markdown(
                        f"**Mention Source:** {format_field(sample_series.get('mention_source'))}"
                    )
                    st.markdown(
                        f"**Outlet or Author:** {format_field(sample_series.get('Outlet or Author'))}"
                    )

            col_true, col_false = st.columns(2)
            render_sample(col_true, true_sample, "TRUE Sample")
            render_sample(col_false, false_sample, "FALSE Sample")

else:
    st.info("Click 'Load Data' in the sidebar to start analyzing your data.")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Altmetric Source Analysis Dashboard")

