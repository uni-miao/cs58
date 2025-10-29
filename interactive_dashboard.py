import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from PIL import Image
import io

# Import original analysis functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from abstract_consistency_analysis_fixed import analyze_abstract_consistency, correlation_analysis

# Set page config
st.set_page_config(page_title="Academic Abstract Consistency Analysis", layout="wide")

st.title("Academic Abstract Consistency Analysis Dashboard")

# Add debug information
st.write("Dashboard is loaded. If you can see this message, the interface is working correctly.")
st.write("Please configure settings in the sidebar and click 'Run Analysis' to begin.")

# Sidebar configuration
st.sidebar.header("Analysis Settings")
csv_path = st.sidebar.text_input("Dataset Path", value=r"D:\研究生课程文件\25-S2\5703\abstract对比\数据集合\merged_dois_dataset.csv")

# Data range settings
col1, col2 = st.sidebar.columns(2)
with col1:
    start_idx = st.number_input("Start Row", min_value=0, value=0)
with col2:
    max_rows = st.number_input("Max Rows (0 for all)", min_value=0, value=10)
    end_idx = None if max_rows == 0 else start_idx + max_rows

# Run analysis button
if st.sidebar.button("Run Analysis"):
    with st.spinner("Analyzing data, please wait..."):
        # Execute analysis
        results = analyze_abstract_consistency(csv_path, start_idx, end_idx)
        
        if results:
            # Convert results to DataFrame
            df_results = pd.DataFrame(results)
            
            # Cache results for later use
            st.session_state['df_results'] = df_results
            st.success(f"Analysis completed! Processed {len(df_results)} records")
        else:
            st.error("Analysis yielded no results. Please check your data path and range settings")

# Display visualizations if results are available
if 'df_results' in st.session_state:
    df_results = st.session_state['df_results']
    
    # Define consistency dimension columns
    consistency_cols = [
        'Abstract-Title Consistency', 
        'Abstract-Body Consistency', 
        'Title-Body Consistency',
        'Overall Consistency'
    ]
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Distribution Analysis", "Correlation Analysis", "Document Comparison", "Data Table"])
    
    with tab1:
        st.header("Consistency Overview")
        
        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Consistency Mean", f"{df_results['Overall Consistency'].mean():.3f}")
        col2.metric("Abstract-Title Consistency Mean", f"{df_results['Abstract-Title Consistency'].mean():.3f}")
        col3.metric("Abstract-Body Consistency Mean", f"{df_results['Abstract-Body Consistency'].mean():.3f}")
        col4.metric("Title-Body Consistency Mean", f"{df_results['Title-Body Consistency'].mean():.3f}")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(df_results[consistency_cols].describe())
        
        # Overall summary chart
        st.subheader("Consistency Dimensions Overview")
        fig = px.box(
            pd.melt(df_results[consistency_cols], value_vars=consistency_cols),
            x="variable", 
            y="value",
            title="Distribution of Consistency Scores",
            labels={"variable": "Dimension", "value": "Consistency Score"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Distribution Analysis")
        
        # Distribution selection
        selected_dimension = st.selectbox(
            "Select Dimension",
            consistency_cols,
            index=3  # Default to Overall Consistency
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            hist_fig = px.histogram(
                df_results, 
                x=selected_dimension,
                nbins=20,
                marginal="rug",
                title=f"Distribution of {selected_dimension}"
            )
            hist_fig.update_layout(bargap=0.1)
            st.plotly_chart(hist_fig, use_container_width=True)
        
        with col2:
            # Box plot with points
            box_fig = px.box(
                df_results,
                y=selected_dimension,
                points="all",
                title=f"Box Plot of {selected_dimension}"
            )
            st.plotly_chart(box_fig, use_container_width=True)
        
        # Comparison of all distributions
        st.subheader("Comparison of All Consistency Dimensions")
        
        # Create KDE plot
        fig = go.Figure()
        for col in consistency_cols[:3]:  # Exclude Overall to reduce clutter
            fig.add_trace(go.Violin(
                x=df_results[col],
                name=col.split()[0],  # Just use the first part of the name
                box_visible=True,
                meanline_visible=True
            ))
        
        fig.update_layout(
            title="Violin Plot of Consistency Dimensions",
            xaxis_title="Consistency Score",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Correlation Analysis")
        
        # Calculate correlation matrix
        corr_matrix = df_results[consistency_cols].corr()
        
        # Correlation heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            title="Correlation Matrix of Consistency Dimensions",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot matrix
        st.subheader("Scatter Plot Matrix")
        
        fig = px.scatter_matrix(
            df_results[consistency_cols],
            dimensions=consistency_cols,
            title="Relationships Between Consistency Dimensions"
        )
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Regression analysis
        st.subheader("Regression Analysis")
        
        x_dim = st.selectbox("Select X Dimension", consistency_cols[:3])
        
        fig = px.scatter(
            df_results,
            x=x_dim,
            y="Overall Consistency",
            trendline="ols",
            title=f"{x_dim} vs Overall Consistency"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation statistics
        with st.expander("View Detailed Correlation Statistics"):
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1))
            
            # Statistical significance
            st.subheader("Statistical Significance (p-values)")
            import scipy.stats as stats
            p_values = pd.DataFrame(index=consistency_cols, columns=consistency_cols)
            
            for i in consistency_cols:
                for j in consistency_cols:
                    if i != j:
                        corr, p_value = stats.pearsonr(df_results[i], df_results[j])
                        p_values.loc[i, j] = p_value
                    else:
                        p_values.loc[i, j] = 0
            
            st.dataframe(p_values.style.background_gradient(cmap='YlGnBu', axis=None))
    
    with tab4:
        st.header("Document Comparison")
        
        # Filter by DOI
        if len(df_results) > 0:
            all_dois = df_results['DOI'].tolist()
            selected_dois = st.multiselect(
                "Select DOIs to Compare",
                options=all_dois,
                default=all_dois[:min(5, len(all_dois))]
            )
            
            if selected_dois:
                filtered_df = df_results[df_results['DOI'].isin(selected_dois)]
                
                # Document comparison chart
                fig = go.Figure()
                
                for i, doi in enumerate(selected_dois):
                    doc_data = filtered_df[filtered_df['DOI'] == doi]
                    if not doc_data.empty:
                        doc_title = doc_data['Title'].iloc[0]
                        short_title = doc_title[:30] + '...' if len(doc_title) > 30 else doc_title
                        
                        fig.add_trace(go.Bar(
                            x=consistency_cols,
                            y=doc_data[consistency_cols].values[0],
                            name=short_title
                        ))
                
                fig.update_layout(
                    title="Consistency Comparison Across Selected Documents",
                    xaxis_title="Consistency Dimension",
                    yaxis_title="Consistency Score",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Radar chart for comparison
                categories = consistency_cols
                fig = go.Figure()
                
                for i, doi in enumerate(selected_dois):
                    doc_data = filtered_df[filtered_df['DOI'] == doi]
                    if not doc_data.empty:
                        doc_title = doc_data['Title'].iloc[0]
                        short_title = doc_title[:30] + '...' if len(doc_title) > 30 else doc_title
                        
                        fig.add_trace(go.Scatterpolar(
                            r=doc_data[categories].values[0],
                            theta=categories,
                            fill='toself',
                            name=short_title
                        ))
                
                fig.update_layout(
                    title="Radar Chart of Document Consistency",
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Data Table")
        
        # Show data with DOI, Title, and consistency scores
        st.dataframe(df_results)
        
        # Download option
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="abstract_consistency_results.csv",
            mime="text/csv"
        )

else:
    st.info("Click 'Run Analysis' in the sidebar to start analyzing data.")
    
    # Display sample image to verify rendering is working
    st.subheader("Sample Visualization Preview")
    st.write("Below is an example of what visualizations will look like after analysis:")
    
    # Create a simple sample plot
    import matplotlib.pyplot as plt
    import numpy as np
    import io
    from PIL import Image
    
    # Generate sample data
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title('Sample Visualization')
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Display the image
    st.image(buf)
    
    # Show instructions
    st.header("Instructions")
    st.write("""
    This dashboard provides an interactive interface for analyzing academic abstract consistency.
    
    ### How to use:
    1. Set the path to your dataset in the sidebar
    2. Specify the range of data to analyze
    3. Click 'Run Analysis' to process the data
    4. Explore the results through the various tabs:
        - Overview: General statistics and summary
        - Distribution Analysis: Detailed distribution visualizations
        - Correlation Analysis: Relationships between consistency dimensions
        - Document Comparison: Compare selected documents
        - Data Table: View and download raw data
    
    ### Required data format:
    The input CSV file should contain columns for DOI, Title, page_title, text, and final_original_abstract.
    """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Abstract Analysis Dashboard")
