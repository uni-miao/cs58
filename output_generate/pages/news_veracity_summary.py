"""
News Veracity Summary Page

This page aggregates key indicators related to news veracity across available datasets
and presents a concise, decision-focused summary.
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

# Page title (parent config may set title already)
st.title("Summary")

# Sidebar configuration
st.sidebar.header("Summary Settings")

# Get datasets_dir injected by parent or compute fallback
try:
    datasets_dir = DATASETS_DIR  # type: ignore
except NameError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    datasets_dir = os.path.join(parent_dir, "datasets")

# Dataset paths
style_csv = os.path.join(datasets_dir, "style_features_data.csv")
text_dir = os.path.join(datasets_dir, "text_analysis")
sentiment_stats_csv = os.path.join(text_dir, "text_sentiment_stats.csv")
structure_stats_csv = os.path.join(text_dir, "text_structure_stats.csv")
summary_csv = os.path.join(text_dir, "text_analysis_summary.csv")

# Auto-load toggle
auto_load = st.sidebar.checkbox(
    "Auto-load on startup", value=True,
    help="Automatically load datasets when page opens."
)

# Cache management
st.sidebar.markdown("---")
with st.sidebar.expander("Cache Management", expanded=False):
    if st.button("Clear All Caches (Summary Page)"):
        clear_cache()
        st.success("All caches cleared!")
        st.session_state['news_summary_loaded'] = False

# Helper: safe load with cache
def safe_load(csv_path):
    if not os.path.exists(csv_path):
        return None
    cache_path = get_cache_path(csv_path)
    metadata_path = get_cache_metadata_path(csv_path)
    try:
        if is_cache_valid(csv_path, cache_path, metadata_path):
            df = load_cache(csv_path)
            if df is not None:
                return df
        # fallback to source
        df = load_data_from_source(csv_path)
        return df
    except Exception:
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return None

# Load datasets
should_load = auto_load or st.sidebar.button("Load/Reload Summary Data")

if should_load:
    with st.spinner("Loading datasets and computing indicators..."):
        df_style = safe_load(style_csv)
        df_sent_stats = safe_load(sentiment_stats_csv)
        df_struct_stats = safe_load(structure_stats_csv)
        df_summary = safe_load(summary_csv)
        st.session_state['news_summary'] = {
            'style': df_style,
            'sent_stats': df_sent_stats,
            'struct_stats': df_struct_stats,
            'summary': df_summary,
        }
        st.session_state['news_summary_loaded'] = True

# Guard: availability
data_bundle = st.session_state.get('news_summary') if st.session_state.get('news_summary_loaded') else None

if not data_bundle:
    st.info("Enable 'Auto-load on startup' or click 'Load/Reload Summary Data' to begin.")
else:
    df_style = data_bundle.get('style')
    df_sent_stats = data_bundle.get('sent_stats')
    df_struct_stats = data_bundle.get('struct_stats')
    df_summary = data_bundle.get('summary')

    st.markdown("---")
    st.header("Summary: Key Indicators (DOI Image News Distribution and Sentiment Analysis)")

    # Compute high-level metrics with robust fallbacks
    # 1) Sensationality, Subjectivity, Retraction-language prevalence
    sens_by_source = None
    subj_by_source = None
    retract_lang_share = None
    if df_style is not None and len(df_style) > 0:
        if 'sensationality_score' in df_style.columns and 'source' in df_style.columns:
            sens_by_source = df_style.groupby('source')['sensationality_score'].mean().reset_index()
        if 'subjectivity_score' in df_style.columns and 'source' in df_style.columns:
            subj_by_source = df_style.groupby('source')['subjectivity_score'].mean().reset_index()
        if 'has_retraction_language' in df_style.columns and 'source' in df_style.columns:
            retract_lang_share = (
                df_style.groupby('source')['has_retraction_language']
                .mean()
                .reset_index()
                .rename(columns={'has_retraction_language': 'share'})
            )

    # 2) Sentiment compound difference
    compound_by_source = None
    if df_sent_stats is not None and set(['compound_mean', 'source']).issubset(df_sent_stats.columns):
        compound_by_source = df_sent_stats[['source', 'compound_mean']]
    elif df_summary is not None and set(['source', 'avg_compound_sentiment']).issubset(df_summary.columns):
        compound_by_source = df_summary[['source', 'avg_compound_sentiment']].rename(
            columns={'avg_compound_sentiment': 'compound_mean'}
        )

    # 3) Structure (avg sentence length, num words)
    struct_by_source = None
    if df_struct_stats is not None and set(['source', 'avg_sentence_len', 'num_words']).issubset(df_struct_stats.columns):
        struct_by_source = df_struct_stats[['source', 'avg_sentence_len', 'num_words']]
    elif df_summary is not None and set(['source', 'avg_sentence_len', 'avg_num_words']).issubset(df_summary.columns):
        struct_by_source = df_summary[['source', 'avg_sentence_len', 'avg_num_words']].rename(
            columns={'avg_num_words': 'num_words'}
        )

    # 4) Build a combined, linked view by source
    combined_by_source = None
    try:
        parts = []
        if compound_by_source is not None:
            parts.append(compound_by_source.rename(columns={'compound_mean': 'compound_mean'}))
        if sens_by_source is not None:
            parts.append(sens_by_source.rename(columns={'sensationality_score': 'avg_sensationality'}))
        if subj_by_source is not None:
            parts.append(subj_by_source.rename(columns={'subjectivity_score': 'avg_subjectivity'}))
        if retract_lang_share is not None:
            parts.append(retract_lang_share.rename(columns={'share': 'retraction_lang_share'}))
        if struct_by_source is not None:
            parts.append(struct_by_source)
        # Merge all on 'source'
        if parts:
            combined_by_source = parts[0]
            for p in parts[1:]:
                combined_by_source = pd.merge(combined_by_source, p, on='source', how='outer')
            # Order columns for readability
            preferred_cols = [
                'source', 'compound_mean', 'avg_sensationality', 'avg_subjectivity',
                'retraction_lang_share', 'avg_sentence_len', 'num_words'
            ]
            existing = [c for c in preferred_cols if c in combined_by_source.columns]
            rest = [c for c in combined_by_source.columns if c not in existing]
            combined_by_source = combined_by_source[existing + rest]
    except Exception:
        combined_by_source = None

    # Metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        if compound_by_source is not None and set(['Original', 'Retraction']).issubset(set(compound_by_source['source'])):
            orig = float(compound_by_source[compound_by_source['source'] == 'Original']['compound_mean'].iloc[0])
            retr = float(compound_by_source[compound_by_source['source'] == 'Retraction']['compound_mean'].iloc[0])
            st.metric("Compound Sentiment (Original)", f"{orig:.3f}")
            st.metric("Compound Sentiment (Retraction)", f"{retr:.3f}")
            st.metric("Δ Compound (O - R)", f"{(orig - retr):.3f}")
        elif compound_by_source is not None:
            st.metric("Compound Sentiment (available)", f"{compound_by_source['compound_mean'].mean():.3f}")
        else:
            st.info("Sentiment stats unavailable")
    with col2:
        if sens_by_source is not None:
            st.metric("Avg Sensationality (Original)", f"{sens_by_source[sens_by_source['source']=='Original']['sensationality_score'].mean():.2f}" if 'Original' in set(sens_by_source['source']) else "-")
            st.metric("Avg Sensationality (Retraction)", f"{sens_by_source[sens_by_source['source']=='Retraction']['sensationality_score'].mean():.2f}" if 'Retraction' in set(sens_by_source['source']) else "-")
        else:
            st.info("Sensationality unavailable")
    with col3:
        if retract_lang_share is not None:
            label = "Share with Retraction Language"
            if 'Retraction' in set(retract_lang_share['source']):
                val = float(retract_lang_share[retract_lang_share['source']=='Retraction']['share'].mean())
                st.metric(f"{label} (Retraction)", f"{val:.0%}")
            if 'Original' in set(retract_lang_share['source']):
                val = float(retract_lang_share[retract_lang_share['source']=='Original']['share'].mean())
                st.metric(f"{label} (Original)", f"{val:.0%}")
        else:
            st.info("Retraction-language flag unavailable")

    # (Visual Highlights moved to bottom)

    # Narrative insights
    st.markdown("---")
    st.header("Analyst Notes")
    notes = []
    try:
        if compound_by_source is not None and set(['Original','Retraction']).issubset(set(compound_by_source['source'])):
            o = float(compound_by_source[compound_by_source['source']=='Original']['compound_mean'].iloc[0])
            r = float(compound_by_source[compound_by_source['source']=='Retraction']['compound_mean'].iloc[0])
            if o > r:
                notes.append("Original texts tend to have more positive compound sentiment than Retraction texts.")
            elif r > o:
                notes.append("Retraction texts tend to have more positive compound sentiment than Original texts.")
    except Exception:
        pass
    try:
        if sens_by_source is not None and set(sens_by_source['source']):
            if 'Original' in set(sens_by_source['source']) and 'Retraction' in set(sens_by_source['source']):
                so = float(sens_by_source[sens_by_source['source']=='Original']['sensationality_score'].mean())
                sr = float(sens_by_source[sens_by_source['source']=='Retraction']['sensationality_score'].mean())
                if so > sr:
                    notes.append("Original coverage appears more sensational on average.")
                elif sr > so:
                    notes.append("Retraction coverage appears more sensational on average.")
    except Exception:
        pass
    try:
        if retract_lang_share is not None and 'Retraction' in set(retract_lang_share['source']):
            v = float(retract_lang_share[retract_lang_share['source']=='Retraction']['share'].mean())
            if v > 0.1:
                notes.append("Retraction-related language frequently appears in retraction coverage.")
    except Exception:
        pass

    if notes:
        for n in notes:
            st.write(f"- {n}")
    else:
        st.write("- No strong directional differences detected based on available aggregates.")

    # Raw data access (optional)
    st.markdown("---")
    if combined_by_source is not None and len(combined_by_source) > 0:
        st.subheader("Linked Indicators by Source (Combined)")
        st.dataframe(combined_by_source, use_container_width=True)

        # Scatter: Sensationality vs Sentiment, bubble by Subjectivity
        try:
            if set(['avg_sensationality', 'compound_mean', 'avg_subjectivity', 'source']).issubset(combined_by_source.columns):
                fig_scatter = px.scatter(
                    combined_by_source,
                    x='avg_sensationality', y='compound_mean', color='source',
                    size='avg_subjectivity',
                    title='Linked View: Sensationality vs Sentiment (bubble=Subjectivity)'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception:
            pass

        # Multi-metric grouped bars
        try:
            cols_for_bars = [c for c in ['compound_mean', 'avg_sensationality', 'avg_subjectivity', 'retraction_lang_share'] if c in combined_by_source.columns]
            if cols_for_bars:
                bar_df = combined_by_source.melt(id_vars='source', value_vars=cols_for_bars, var_name='metric', value_name='value')
                fig_multi = px.bar(bar_df, x='source', y='value', color='metric', barmode='group',
                                   title='Linked View: Key Indicators by Source')
                st.plotly_chart(fig_multi, use_container_width=True)
        except Exception:
            pass

    with st.expander("View Aggregated Tables", expanded=False):
        if sens_by_source is not None:
            st.subheader("Sensationality by Source")
            st.dataframe(sens_by_source, use_container_width=True)
        if subj_by_source is not None:
            st.subheader("Subjectivity by Source")
            st.dataframe(subj_by_source, use_container_width=True)
        if compound_by_source is not None:
            st.subheader("Compound Sentiment by Source")
            st.dataframe(compound_by_source, use_container_width=True)
        if struct_by_source is not None:
            st.subheader("Structure by Source")
            st.dataframe(struct_by_source, use_container_width=True)

    # Visual Highlights (moved to bottom)
    st.markdown("---")
    st.header("Visual Highlights")

    # Charts: Sensationality & Subjectivity
    charts_row1 = st.columns(2)
    with charts_row1[0]:
        if sens_by_source is not None and len(sens_by_source) > 0:
            fig = px.bar(sens_by_source, x='source', y='sensationality_score', title="Average Sensationality by Source",
                         color='source', labels={'sensationality_score': 'Avg Sensationality', 'source': 'Source'})
            st.plotly_chart(fig, use_container_width=True)
    with charts_row1[1]:
        if subj_by_source is not None and len(subj_by_source) > 0:
            fig = px.bar(subj_by_source, x='source', y='subjectivity_score', title="Average Subjectivity by Source",
                         color='source', labels={'subjectivity_score': 'Avg Subjectivity', 'source': 'Source'})
            st.plotly_chart(fig, use_container_width=True)

    # Charts: Sentiment compound
    if compound_by_source is not None and len(compound_by_source) > 0:
        fig = px.bar(compound_by_source, x='source', y='compound_mean', title="Compound Sentiment by Source",
                     color='source', labels={'compound_mean': 'Compound (Mean)', 'source': 'Source'})
        st.plotly_chart(fig, use_container_width=True)

    # Charts: Structure
    if struct_by_source is not None and len(struct_by_source) > 0:
        fig = go.Figure()
        if 'avg_sentence_len' in struct_by_source.columns:
            fig.add_trace(go.Bar(name='Avg Sentence Len', x=struct_by_source['source'], y=struct_by_source['avg_sentence_len']))
        if 'num_words' in struct_by_source.columns:
            fig.add_trace(go.Bar(name='Num Words', x=struct_by_source['source'], y=struct_by_source['num_words']))
        fig.update_layout(barmode='group', title="Structure by Source")
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 News Veracity Summary")


