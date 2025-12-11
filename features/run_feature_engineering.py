"""
Run Feature Engineering Pipeline
Executes feature engineering and saves the engineered dataset
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_FILE = DATA_DIR / "engineered_features.csv"
CONFIG_FILE = Path(__file__).parent / "feature_config.yaml"

def main():
    """Run feature engineering pipeline"""
    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)
    
    # Load processed data
    print("\n1. Loading processed data...")
    df = load_processed_data()
    print(f"   Original shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Initialize feature engineer
    print("\n2. Initializing feature engineer...")
    fe = FeatureEngineer(config_path=CONFIG_FILE)
    
    # Detect column types
    print("\n3. Detecting column types...")
    column_types = fe.detect_column_types(df)
    print(f"   Datetime columns: {column_types['datetime']}")
    print(f"   Categorical columns: {column_types['categorical']}")
    print(f"   Numeric columns: {column_types['numeric']}")
    print(f"   ID columns: {column_types['id']}")
    
    # Engineer features
    print("\n4. Engineering features...")
    df_engineered = fe.engineer_features(df)
    
    print(f"\n5. Final engineered dataset shape: {df_engineered.shape}")
    print(f"   New columns created: {len(df_engineered.columns) - len(df.columns)}")
    
    # Save engineered features
    print(f"\n6. Saving engineered features to: {OUTPUT_FILE}")
    df_engineered.to_csv(OUTPUT_FILE, index=False)
    print("   ✓ Saved successfully!")
    
    # Summary
    print("\n" + "=" * 60)
    print("Feature Engineering Summary")
    print("=" * 60)
    print(f"Original columns: {len(df.columns)}")
    print(f"Engineered columns: {len(df_engineered.columns)}")
    print(f"New features added: {len(df_engineered.columns) - len(df.columns)}")
    print(f"\nOutput file: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()

