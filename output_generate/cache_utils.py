"""
Cache Management Utilities for Dashboard

This module provides caching functions for efficient data loading.
It can be used both as a module (imported by pages) and as a script (precompute cache).

Usage as module:
    from cache_utils import load_cache, save_cache, is_cache_valid

Usage as script:
    python cache_utils.py [csv_path]
    python cache_utils.py datasets/  # Precompute all CSV files in datasets directory
"""

import os
import hashlib
import json
import time
import pandas as pd
import sys
from pathlib import Path

# ==================== Core Cache Functions ====================

def get_file_signature(file_path):
    """Generate file signature based on path, size, and modification time"""
    try:
        stat = os.stat(file_path)
        # Use path, size, and mtime to create signature
        signature_str = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(signature_str.encode()).hexdigest()[:12]
    except:
        # Fallback to path-based signature
        return hashlib.md5(str(file_path).encode()).hexdigest()[:12]

def get_cache_dir():
    """Get or create cache directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(current_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_cache_path(csv_path):
    """Generate cache file path"""
    cache_dir = get_cache_dir()
    file_sig = get_file_signature(csv_path)
    cache_filename = f"doi_analysis__{file_sig}.parquet"
    return os.path.join(cache_dir, cache_filename)

def get_cache_metadata_path(csv_path):
    """Generate cache metadata file path"""
    cache_dir = get_cache_dir()
    file_sig = get_file_signature(csv_path)
    metadata_filename = f"doi_analysis__{file_sig}.json"
    return os.path.join(cache_dir, metadata_filename)

def is_cache_valid(csv_path, cache_path, metadata_path):
    """Check if cache is valid (file exists and source hasn't changed)"""
    if not os.path.exists(metadata_path):
        return False
    
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if source file has changed
        if not os.path.exists(csv_path):
            return False
        
        source_stat = os.stat(csv_path)
        cached_sig = metadata.get('file_signature', '')
        current_sig = get_file_signature(csv_path)
        
        # Check if file signature matches
        if cached_sig != current_sig:
            return False
        
        # Check modification time (optional additional check)
        if metadata.get('source_mtime') and metadata['source_mtime'] != source_stat.st_mtime:
            return False
        
        # Check if cache file exists (may be .parquet or .pkl)
        cache_format = metadata.get('cache_format', 'parquet')
        if cache_format == 'pkl':
            cache_path = cache_path.replace('.parquet', '.pkl')
        
        if not os.path.exists(cache_path):
            return False
        
        return True
    except:
        return False

def save_cache(csv_path, df_results):
    """Save DataFrame to cache"""
    try:
        cache_path = get_cache_path(csv_path)
        metadata_path = get_cache_metadata_path(csv_path)
        actual_cache_path = cache_path  # Track the actual file saved
        
        # Save DataFrame as parquet (with fallback to pickle if pyarrow not available)
        try:
            df_results.to_parquet(cache_path, index=False, engine='pyarrow')
        except Exception as e:
            # Fallback to pickle if parquet fails
            cache_path_pkl = cache_path.replace('.parquet', '.pkl')
            df_results.to_pickle(cache_path_pkl)
            actual_cache_path = cache_path_pkl
        
        # Save metadata
        source_stat = os.stat(csv_path)
        metadata = {
            'file_path': csv_path,
            'file_signature': get_file_signature(csv_path),
            'source_mtime': source_stat.st_mtime,
            'source_size': source_stat.st_size,
            'cached_at': time.time(),
            'record_count': len(df_results),
            'cache_format': 'parquet' if actual_cache_path.endswith('.parquet') else 'pkl'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        # Try to import streamlit to show warning if available
        try:
            import streamlit as st
            st.warning(f"Failed to save cache: {str(e)}")
        except:
            pass
        return False

def load_cache(csv_path):
    """Load DataFrame from cache"""
    try:
        cache_path = get_cache_path(csv_path)
        metadata_path = get_cache_metadata_path(csv_path)
        
        # Check metadata for cache format
        cache_format = 'parquet'  # default
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    cache_format = metadata.get('cache_format', 'parquet')
            except:
                pass
        
        # Load based on format
        if cache_format == 'pkl':
            cache_path = cache_path.replace('.parquet', '.pkl')
        
        if not os.path.exists(cache_path):
            return None
        
        if cache_path.endswith('.parquet'):
            df_results = pd.read_parquet(cache_path, engine='pyarrow')
        else:
            df_results = pd.read_pickle(cache_path)
        
        return df_results
    except Exception as e:
        return None

def clear_cache(csv_path=None):
    """Clear cache files. If csv_path is None, clear all caches"""
    cache_dir = get_cache_dir()
    if csv_path:
        # Clear specific cache
        cache_path = get_cache_path(csv_path)
        metadata_path = get_cache_metadata_path(csv_path)
        for path in [cache_path, metadata_path, 
                     cache_path.replace('.parquet', '.pkl')]:
            if os.path.exists(path):
                os.remove(path)
    else:
        # Clear all caches
        for file in os.listdir(cache_dir):
            if file.startswith('doi_analysis__'):
                os.remove(os.path.join(cache_dir, file))

def load_data_from_source(csv_path):
    """Load data from source CSV file"""
    df_results = pd.read_csv(csv_path)
    # Save to cache for next time
    save_cache(csv_path, df_results)
    return df_results

# ==================== Precompute Functions ====================

def precompute_cache(csv_path):
    """Precompute and save cache for a given CSV file"""
    print(f"\n{'='*60}")
    print(f"Precomputing cache for: {csv_path}")
    print(f"{'='*60}\n")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found: {csv_path}")
        return False
    
    # Check if cache is already valid
    cache_path = get_cache_path(csv_path)
    metadata_path = get_cache_metadata_path(csv_path)
    
    if is_cache_valid(csv_path, cache_path, metadata_path):
        print(f"✓ Cache already exists and is valid!")
        print(f"  Cache file: {cache_path}")
        print(f"  Use 'Clear Cache' in dashboard or delete cache to regenerate.\n")
        return True
    
    # Load data from source
    print(f"📂 Loading data from source CSV...")
    try:
        df_results = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df_results):,} records")
    except Exception as e:
        print(f"❌ Error loading CSV: {str(e)}")
        return False
    
    # Save to cache
    print(f"💾 Saving to cache...")
    success = save_cache(csv_path, df_results)
    
    if success:
        print(f"✓ Cache saved successfully!")
        print(f"  Cache directory: {get_cache_dir()}")
        print(f"  Cache file: {get_cache_path(csv_path)}")
        print(f"  Metadata file: {get_cache_metadata_path(csv_path)}")
        print(f"\n🎉 Precomputation complete! Dashboard will load instantly.\n")
        return True
    else:
        print(f"❌ Failed to save cache\n")
        return False

def precompute_all_default_datasets():
    """Precompute all default datasets"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, "datasets")
    
    print(f"📋 Precomputing all default datasets...\n")
    default_files = [
        os.path.join(datasets_dir, "enhanced_doi_analysis_v3_summary.csv"),
        os.path.join(datasets_dir, "paper_level_summary.csv"),
        os.path.join(datasets_dir, "paper_domain_pairs.csv"),
        os.path.join(datasets_dir, "style_features_data.csv"),
        os.path.join(datasets_dir, "abstract_retractionNotice", "abstract_features.csv"),
        os.path.join(datasets_dir, "abstract_retractionNotice", "retraction_features.csv"),
    ]
    
    success_count = 0
    for csv_path in default_files:
        if os.path.exists(csv_path):
            if precompute_cache(csv_path):
                success_count += 1
        else:
            print(f"⚠ Skipping (file not found): {csv_path}\n")
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(default_files)} files cached successfully")
    print(f"{'='*60}\n")
    return success_count

def precompute_directory(directory_path):
    """Precompute all CSV files in a directory"""
    print(f"📁 Directory provided. Will process all CSV files...\n")
    csv_files = list(Path(directory_path).glob("*.csv"))
    
    # Also search in subdirectories
    csv_files.extend(list(Path(directory_path).glob("**/*.csv")))
    
    if not csv_files:
        print(f"❌ No CSV files found in directory: {directory_path}")
        return 0
    
    success_count = 0
    for csv_file in csv_files:
        if precompute_cache(str(csv_file)):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary: {success_count}/{len(csv_files)} files cached successfully")
    print(f"{'='*60}\n")
    return success_count

# ==================== Main Function (Script Mode) ====================

def main():
    """Main function - called when script is run directly"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, "datasets")
    
    # Get CSV path from command line or use defaults
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        # Expand user path if needed
        csv_path = os.path.expanduser(csv_path)
        
        # Check if it's a directory (precompute all CSV files)
        if os.path.isdir(csv_path):
            precompute_directory(csv_path)
        else:
            # Single file
            precompute_cache(csv_path)
    else:
        # No argument provided - precompute all default datasets
        precompute_all_default_datasets()

if __name__ == "__main__":
    main()
