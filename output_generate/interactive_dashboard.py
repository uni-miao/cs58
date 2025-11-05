"""
Main Dashboard Launcher

This is the main entry point that provides navigation to different analysis modes.

To use this dashboard:
- Run: python -m streamlit run interactive_dashboard.py
- Then select an analysis mode from the sidebar
"""

import streamlit as st
import os

# Initialize session state for page tracking
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Read current page from session state first to set page config
current_page = st.session_state.current_page

# Set page config FIRST - must be before any other Streamlit commands
if current_page == 'image':
    st.set_page_config(page_title="Image Analysis Dashboard", layout="wide", page_icon="📷")
elif current_page == 'source':
    st.set_page_config(page_title="Altmetric Source Analysis Dashboard", layout="wide", page_icon="📈")
elif current_page == 'style':
    st.set_page_config(page_title="Altmetric News Analysis Dashboard", layout="wide", page_icon="📝")
elif current_page == 'abstract_retraction':
    st.set_page_config(page_title="Abstract & Retraction Analysis Dashboard", layout="wide", page_icon="📊")
elif current_page == 'retraction_consistency':
    st.set_page_config(page_title="Retraction Consistency Analysis Dashboard", layout="wide", page_icon="🔍")
elif current_page == 'summary':
    st.set_page_config(page_title="Summary", layout="wide", page_icon="🧭")
else:
    st.set_page_config(page_title="Multisource Integrity Dashboard", layout="wide", page_icon="📊")

# Define page themes
page_themes = {
    'home': '📊 Multisource Integrity Dashboard',
    'image': '📷 Image Analysis',
    'source': '📈 Altmetric Source Analysis',
    'style': '📝 Altmetric News Analysis',
    'abstract_retraction': '📊 Abstract & Retraction Analysis',
    'retraction_consistency': '🔍 Retraction Consistency Analysis',
    'summary': '🧭 Summary'
}

# Navigation section - Display current page theme at top
current_theme = page_themes.get(current_page, '📊 Multisource Integrity Dashboard')
st.sidebar.markdown(f"### {current_theme}")
st.sidebar.markdown("---")

# Navigation options with styled boxes
st.sidebar.markdown("**Navigation**")

# Add custom CSS for navigation boxes - style buttons to look like boxes
st.markdown("""
<style>
    /* Hide Streamlit's default multi-page navigation menu */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Style all navigation buttons as boxes */
    div[data-testid="stSidebar"] button[data-testid*="baseButton-secondary"] {
        width: 100%;
        text-align: left;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        background-color: white;
        font-weight: normal;
        transition: all 0.2s ease;
    }
    div[data-testid="stSidebar"] button[data-testid*="baseButton-secondary"]:hover {
        background-color: #f8f8f8;
        border-color: #c0c0c0;
    }
    /* Style active navigation box */
    .nav-active-box {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border: 1px solid #888888;
        background-color: #f0f0f0;
        text-align: left;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Create navigation options
nav_options = [
    ("home", "Home"),
    ("abstract_retraction", "Abstract & Retraction Analysis"),
    ("retraction_consistency", "Retraction Consistency Analysis"),
    ("image", "Image Analysis"),
    ("source", "Altmetric Source Analysis"),
    ("style", "Altmetric News Analysis"),
    ("summary", "Summary")
]

# Display navigation options as styled boxes
for page_key, page_label in nav_options:
    is_active = (current_page == page_key)
    
    if is_active:
        # Active page: show as styled markdown box (gray background)
        st.sidebar.markdown(f'<div class="nav-active-box">{page_label}</div>', unsafe_allow_html=True)
    else:
        # Inactive page: use styled button (white background)
        if st.sidebar.button(page_label, key=f"nav_{page_key}", use_container_width=True):
            if st.session_state.current_page != page_key:
                st.session_state.current_page = page_key
                st.rerun()

st.sidebar.markdown("---")

# Get base directory and datasets directory (for injecting into exec context)
base_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(base_dir, 'datasets')

# Conditional rendering based on current_page
if current_page == 'image':
    st.title("📊 Image Analysis Dashboard")
    # Import and run image analysis with UTF-8 encoding
    pages_dir = os.path.join(base_dir, 'pages')
    image_analysis_path = os.path.join(pages_dir, 'image_analysis.py')
    with open(image_analysis_path, encoding='utf-8') as f:
        code = f.read()
        # Simple removal of page config comment and title line
        code = code.replace('# Page config is set by parent interactive_dashboard.py\n# No need to set it here to avoid conflicts\n\n', '')
        code = code.replace('st.title("📊 Image Analysis Dashboard")', '', 1)  # Remove only first occurrence
        # Inject datasets_dir into exec context
        exec_globals = {'__file__': image_analysis_path, 'DATASETS_DIR': datasets_dir}
        exec(code, exec_globals)
    
elif current_page == 'source':
    st.title("📈 Altmetric Source Analysis Dashboard")
    # Import and run data source analysis with UTF-8 encoding
    pages_dir = os.path.join(base_dir, 'pages')
    data_source_analysis_path = os.path.join(pages_dir, 'data_source_analysis.py')
    with open(data_source_analysis_path, encoding='utf-8') as f:
        code = f.read()
        # Simple removal of page config comment and title line
        code = code.replace('# Page config is set by parent interactive_dashboard.py\n# No need to set it here to avoid conflicts\n\n', '')
        code = code.replace('st.title("📈 Altmetric Source Analysis Dashboard")', '', 1)  # Remove only first occurrence
        # Inject datasets_dir into exec context
        exec_globals = {'__file__': data_source_analysis_path, 'DATASETS_DIR': datasets_dir}
        exec(code, exec_globals)
    
elif current_page == 'style':
    st.title("📝 Altmetric News Analysis Dashboard")
    # Import and run style features analysis with UTF-8 encoding
    pages_dir = os.path.join(base_dir, 'pages')
    style_features_analysis_path = os.path.join(pages_dir, 'style_features_analysis.py')
    with open(style_features_analysis_path, encoding='utf-8') as f:
        code = f.read()
        # Simple removal of page config comment and title line
        code = code.replace('# Page config is set by parent interactive_dashboard.py\n# No need to set it here to avoid conflicts\n\n', '')
        code = code.replace('st.title("📝 Altmetric News Analysis Dashboard")', '', 1)  # Remove only first occurrence
        # Inject datasets_dir into exec context
        exec_globals = {'__file__': style_features_analysis_path, 'DATASETS_DIR': datasets_dir}
        exec(code, exec_globals)
    
    
elif current_page == 'abstract_retraction':
    st.title("📊 Abstract & Retraction Analysis Dashboard")
    # Import and run abstract retraction analysis with UTF-8 encoding
    pages_dir = os.path.join(base_dir, 'pages')
    abstract_retraction_analysis_path = os.path.join(pages_dir, 'abstract_retraction_analysis.py')
    with open(abstract_retraction_analysis_path, encoding='utf-8') as f:
        code = f.read()
        # Simple removal of page config comment and title line
        code = code.replace('# Page config is set by parent interactive_dashboard.py\n# No need to set it here to avoid conflicts\n\n', '')
        # Remove both possible title variations
        code = code.replace('st.title("📊 Abstract & Retraction Notice Analysis Dashboard")', '', 1)  # Remove first occurrence
        code = code.replace('st.title("📊 Abstract & Retraction Analysis Dashboard")', '', 1)  # Remove first occurrence
        # Inject datasets_dir into exec context
        exec_globals = {'__file__': abstract_retraction_analysis_path, 'DATASETS_DIR': datasets_dir}
        exec(code, exec_globals)

elif current_page == 'retraction_consistency':
    st.title("🔍 Retraction Consistency Analysis Dashboard")
    # Import and run retraction consistency analysis with UTF-8 encoding
    pages_dir = os.path.join(base_dir, 'pages')
    retraction_consistency_analysis_path = os.path.join(pages_dir, 'retraction_consistency_analysis.py')
    with open(retraction_consistency_analysis_path, encoding='utf-8') as f:
        code = f.read()
        # Simple removal of page config comment and title line
        code = code.replace('# Page config is set by parent interactive_dashboard.py\n# No need to set it here to avoid conflicts\n\n', '')
        code = code.replace('st.title("🔍 Retraction Consistency Analysis Dashboard")', '', 1)  # Remove first occurrence
        # Inject datasets_dir into exec context
        exec_globals = {'__file__': retraction_consistency_analysis_path, 'DATASETS_DIR': datasets_dir}
        exec(code, exec_globals)
    
elif current_page == 'summary':
    st.title("Summary")
    # Import and run news veracity summary with UTF-8 encoding
    pages_dir = os.path.join(base_dir, 'pages')
    summary_path = os.path.join(pages_dir, 'news_veracity_summary.py')
    with open(summary_path, encoding='utf-8') as f:
        code = f.read()
        # Remove page config comment and title line
        code = code.replace('# Page config is set by parent interactive_dashboard.py\n# No need to set it here to avoid conflicts\n\n', '')
        code = code.replace('st.title("Summary")', '', 1)
        exec_globals = {'__file__': summary_path, 'DATASETS_DIR': datasets_dir}
        exec(code, exec_globals)
    
else:
    # Home page
    st.title("📊 Multisource Integrity Dashboard")
    st.markdown("---")
    st.header("Welcome!")
    st.markdown("""
    This dashboard provides comprehensive analysis tools for DOI-related research data.
    
    ### Available Analysis Modes:
    
    **📷 Image Analysis**
    - Visualize image authenticity metrics
    - Detect suspicious patterns
    - Explore correlations between different image characteristics
    - Compare multiple DOIs side by side
    
    **📈 Altmetric Source Analysis**
    - Analyze media coverage distribution
    - Explore paper-domain relationships
    - Visualize overlap patterns with Venn diagrams
    - Track domain-level statistics
    
    **📝 Altmetric News Analysis**
    - Compare sentiment scores across news sources
    - Analyze sensationality and subjectivity distributions
    - Visualize retraction language frequency
    - Explore style differences between sources
    
    **📊 Abstract & Retraction Analysis**
    - Two-layer text analysis system (shared + specific dimensions)
    - Language complexity, readability, and tone comparison
    - Abstract-specific: scientific terms, structure roles, innovation
    - Retraction-specific: reasons, responsibility, template similarity
    - Statistical significance testing with t-statistics
    
    **🔍 Retraction Consistency Analysis**
    - Before vs after retraction comparison analysis
    - Semantic similarity and entity overlap analysis
    - Reason consistency and sentiment analysis
    - Multi-dimensional consistency pattern exploration
    
    ### Features:
    - ⚡ **Fast Loading**: Automatic caching for instant data access
    - 🔄 **Auto-reload**: Smart cache invalidation when data changes
    - 📊 **Interactive Charts**: Powered by Plotly for rich visualizations
    - 💾 **Data Export**: Download analyzed results as CSV
    
    ### Getting Started:
    1. Select an analysis mode from the sidebar
    2. Enable "Auto-load on startup" for automatic data loading
    3. Or manually load data using the sidebar controls
    4. Explore the various tabs and visualizations
    
    ---
    
    **Note**: Make sure your data files are in the `datasets` directory before starting analysis.
    """)
    
    # Quick start cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.info("""
        **📷 Image Analysis**
        
        Perfect for analyzing image-related metrics:
        - Authenticity scores
        - Suspicious pattern detection
        - Image distribution analysis
        """)
    
    with col2:
        st.info("""
        **📈 Altmetric Source Analysis**
        
        Perfect for analyzing source-level data:
        - Domain statistics
        - Paper-domain relationships
        - Overlap visualization
        """)
    
    with col3:
        st.info("""
        **📝 Altmetric News Analysis**
        
        Perfect for analyzing style characteristics:
        - Sentiment analysis
        - Sensationality scores
        - Subjectivity metrics
        - Retraction language patterns
        """)
    
    with col4:
        st.info("""
        **📊 Abstract & Retraction Analysis**
        
        Two-layer analysis system:
        - Shared dimensions comparison
        - Abstract-specific features
        - Retraction-specific features
        - Statistical significance testing
        """)
    
    with col5:
        st.info("""
        **🔍 Retraction Consistency Analysis**
        
        Perfect for analyzing consistency patterns:
        - Before vs after retraction comparison
        - Semantic similarity analysis
        - Entity overlap patterns
        - Sentiment consistency tracking
        """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Multisource Integrity Dashboard")
