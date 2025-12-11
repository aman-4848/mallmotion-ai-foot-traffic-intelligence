"""
Feature Analysis Script
Analyzes features before and after engineering
Can be run as: python features/feature_analysis.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
FEATURE_ANALYSIS_DIR = RESULTS_DIR / "feature_analysis"
FEATURE_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_before_engineering(df):
    """Analyze data before feature engineering"""
    print("\n" + "=" * 60)
    print("BEFORE FEATURE ENGINEERING")
    print("=" * 60)
    
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {}
    }
    
    # Categorical summary
    for col in df.select_dtypes(include=['object']).columns:
        analysis['categorical_summary'][col] = {
            'unique_count': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    print(f"\nDataset Shape: {analysis['shape']}")
    print(f"\nColumns ({len(analysis['columns'])}):")
    for col in analysis['columns']:
        dtype = analysis['dtypes'][col]
        missing = analysis['missing_values'][col]
        missing_pct = analysis['missing_percentage'][col]
        print(f"  - {col}: {dtype} (Missing: {missing} ({missing_pct:.2f}%))")
    
    print(f"\nNumeric Columns Summary:")
    if analysis['numeric_summary']:
        print(df.describe())
    else:
        print("  No numeric columns found")
    
    print(f"\nCategorical Columns Summary:")
    for col, info in analysis['categorical_summary'].items():
        print(f"  - {col}: {info['unique_count']} unique values")
        print(f"    Top values: {list(info['top_values'].keys())[:3]}")
    
    return analysis

def analyze_after_engineering(df_original, df_engineered):
    """Analyze data after feature engineering"""
    print("\n" + "=" * 60)
    print("AFTER FEATURE ENGINEERING")
    print("=" * 60)
    
    new_columns = [col for col in df_engineered.columns if col not in df_original.columns]
    
    analysis = {
        'original_shape': df_original.shape,
        'engineered_shape': df_engineered.shape,
        'new_columns': new_columns,
        'new_columns_count': len(new_columns),
        'columns_added': len(df_engineered.columns) - len(df_original.columns),
        'missing_values': df_engineered.isnull().sum().to_dict(),
        'numeric_features': list(df_engineered.select_dtypes(include=[np.number]).columns),
        'categorical_features': list(df_engineered.select_dtypes(include=['object']).columns)
    }
    
    print(f"\nOriginal Shape: {analysis['original_shape']}")
    print(f"Engineered Shape: {analysis['engineered_shape']}")
    print(f"\nNew Features Created: {analysis['new_columns_count']}")
    print(f"\nNew Columns:")
    for col in new_columns[:20]:  # Show first 20
        dtype = df_engineered[col].dtype
        missing = analysis['missing_values'][col]
        print(f"  - {col}: {dtype} (Missing: {missing})")
    if len(new_columns) > 20:
        print(f"  ... and {len(new_columns) - 20} more")
    
    print(f"\nTotal Numeric Features: {len(analysis['numeric_features'])}")
    print(f"Total Categorical Features: {len(analysis['categorical_features'])}")
    
    return analysis

def create_visualizations(df_original, df_engineered):
    """Create visualization plots"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Missing values comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    missing_before = df_original.isnull().sum()
    missing_after = df_engineered.isnull().sum()
    
    if missing_before.sum() > 0:
        missing_before[missing_before > 0].plot(kind='bar', ax=axes[0], color='red')
        axes[0].set_title('Missing Values - Before Engineering')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
    
    if missing_after.sum() > 0:
        missing_after[missing_after > 0].head(20).plot(kind='bar', ax=axes[1], color='orange')
        axes[1].set_title('Missing Values - After Engineering (Top 20)')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
        axes[1].set_title('Missing Values - After Engineering')
    
    plt.tight_layout()
    plt.savefig(FEATURE_ANALYSIS_DIR / "missing_values_comparison.png", dpi=150, bbox_inches='tight')
    print("  ✓ Saved: missing_values_comparison.png")
    plt.close()
    
    # 2. Feature count comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Original', 'Engineered']
    counts = [len(df_original.columns), len(df_engineered.columns)]
    bars = ax.bar(categories, counts, color=['skyblue', 'lightgreen'])
    ax.set_ylabel('Number of Features')
    ax.set_title('Feature Count Comparison')
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FEATURE_ANALYSIS_DIR / "feature_count_comparison.png", dpi=150, bbox_inches='tight')
    print("  ✓ Saved: feature_count_comparison.png")
    plt.close()
    
    # 3. Numeric features distribution (sample)
    numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        n_cols = min(6, len(numeric_cols))
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols[:n_cols]):
            df_engineered[col].hist(bins=30, ax=axes[i], color='steelblue', edgecolor='black')
            axes[i].set_title(f'{col}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(n_cols, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Distribution of Sample Numeric Features (After Engineering)', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(FEATURE_ANALYSIS_DIR / "numeric_features_distribution.png", dpi=150, bbox_inches='tight')
        print("  ✓ Saved: numeric_features_distribution.png")
        plt.close()

def main():
    """Main analysis function"""
    print("=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    
    # Load data
    print("\nLoading processed data...")
    df_original = load_processed_data()
    
    # Analyze before
    analysis_before = analyze_before_engineering(df_original)
    
    # Engineer features
    print("\n" + "=" * 60)
    print("RUNNING FEATURE ENGINEERING")
    print("=" * 60)
    fe = FeatureEngineer()
    df_engineered = fe.engineer_features(df_original)
    
    # Analyze after
    analysis_after = analyze_after_engineering(df_original, df_engineered)
    
    # Create visualizations
    create_visualizations(df_original, df_engineered)
    
    # Save summary
    summary = {
        'before': {
            'shape': analysis_before['shape'],
            'columns_count': len(analysis_before['columns']),
            'missing_values_total': sum(analysis_before['missing_values'].values())
        },
        'after': {
            'shape': analysis_after['engineered_shape'],
            'columns_count': len(df_engineered.columns),
            'new_features': analysis_after['new_columns_count'],
            'missing_values_total': sum(analysis_after['missing_values'].values())
        }
    }
    
    import json
    with open(FEATURE_ANALYSIS_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {FEATURE_ANALYSIS_DIR}")
    print(f"  - missing_values_comparison.png")
    print(f"  - feature_count_comparison.png")
    print(f"  - numeric_features_distribution.png")
    print(f"  - summary.json")
    print("=" * 60)

if __name__ == "__main__":
    main()

