"""
Feature Engineering Verification Script
Checks if feature engineering is complete before model training
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from streamlit_app.utils.data_loader import load_processed_data
from features.feature_engineering import FeatureEngineer

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DATA_FILE = DATA_DIR / "merged data set.csv"
ENGINEERED_DATA_FILE = DATA_DIR / "engineered_features.csv"

def verify_feature_engineering():
    """Verify feature engineering is complete"""
    print("=" * 70)
    print("FEATURE ENGINEERING VERIFICATION")
    print("=" * 70)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Processed data exists
    print("\n[CHECK 1] Processed data file exists...")
    total_checks += 1
    if PROCESSED_DATA_FILE.exists():
        print(f"   ✓ Found: {PROCESSED_DATA_FILE}")
        checks_passed += 1
    else:
        print(f"   ✗ Missing: {PROCESSED_DATA_FILE}")
        print("   → Please ensure processed data is in data/processed/")
        return False
    
    # Check 2: Load and verify processed data
    print("\n[CHECK 2] Loading processed data...")
    total_checks += 1
    try:
        df_original = load_processed_data()
        print(f"   ✓ Loaded successfully")
        print(f"   - Shape: {df_original.shape}")
        print(f"   - Columns: {len(df_original.columns)}")
        print(f"   - Missing values: {df_original.isnull().sum().sum():,}")
        checks_passed += 1
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return False
    
    # Check 3: Feature engineering module available
    print("\n[CHECK 3] Feature engineering module available...")
    total_checks += 1
    try:
        fe = FeatureEngineer()
        print(f"   ✓ FeatureEngineer initialized")
        checks_passed += 1
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Check 4: Column type detection
    print("\n[CHECK 4] Column type detection...")
    total_checks += 1
    try:
        column_types = fe.detect_column_types(df_original)
        print(f"   ✓ Column types detected:")
        print(f"     - Datetime: {len(column_types['datetime'])} columns")
        print(f"     - Categorical: {len(column_types['categorical'])} columns")
        print(f"     - Numeric: {len(column_types['numeric'])} columns")
        print(f"     - ID: {len(column_types['id'])} columns")
        checks_passed += 1
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Check 5: Engineered features file exists
    print("\n[CHECK 5] Engineered features file exists...")
    total_checks += 1
    if ENGINEERED_DATA_FILE.exists():
        print(f"   ✓ Found: {ENGINEERED_DATA_FILE}")
        checks_passed += 1
        
        # Check 6: Load and verify engineered data
        print("\n[CHECK 6] Verifying engineered features...")
        total_checks += 1
        try:
            df_engineered = pd.read_csv(ENGINEERED_DATA_FILE)
            print(f"   ✓ Loaded successfully")
            print(f"   - Shape: {df_engineered.shape}")
            print(f"   - Columns: {len(df_engineered.columns)}")
            print(f"   - New features: {len(df_engineered.columns) - len(df_original.columns)}")
            print(f"   - Missing values: {df_engineered.isnull().sum().sum():,}")
            
            # Check for expected feature types
            numeric_features = df_engineered.select_dtypes(include=['number']).columns.tolist()
            print(f"   - Numeric features: {len(numeric_features)}")
            
            # Check for temporal features
            temporal_features = [col for col in df_engineered.columns 
                               if any(x in col.lower() for x in ['hour', 'day', 'week', 'month', 'year', 'is_', 'sin', 'cos'])]
            print(f"   - Temporal features: {len(temporal_features)}")
            
            checks_passed += 1
        except Exception as e:
            print(f"   ✗ Error loading engineered data: {e}")
            return False
    else:
        print(f"   ⚠ Not found: {ENGINEERED_DATA_FILE}")
        print("   → Feature engineering has not been run yet")
        print("\n   Would you like to run feature engineering now?")
        response = input("   Type 'yes' to run feature engineering: ").strip().lower()
        
        if response == 'yes':
            print("\n" + "=" * 70)
            print("RUNNING FEATURE ENGINEERING...")
            print("=" * 70)
            try:
                df_engineered = fe.engineer_features(df_original)
                df_engineered.to_csv(ENGINEERED_DATA_FILE, index=False)
                print(f"\n   ✓ Feature engineering complete!")
                print(f"   ✓ Saved to: {ENGINEERED_DATA_FILE}")
                checks_passed += 1
            except Exception as e:
                print(f"   ✗ Error during feature engineering: {e}")
                return False
        else:
            print("\n   → Please run feature engineering first:")
            print("     python features/run_feature_engineering.py")
            return False
    
    # Final Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Checks passed: {checks_passed}/{total_checks}")
    
    if checks_passed == total_checks:
        print("\n✓ ALL CHECKS PASSED!")
        print("✓ Feature engineering is complete and ready for model training!")
        print("\nYou can now proceed with model training:")
        print("  - python training/train_classification.py")
        print("  - python training/train_clustering.py")
        print("  - python training/train_forecasting.py")
        return True
    else:
        print("\n✗ SOME CHECKS FAILED")
        print("Please fix the issues above before proceeding with model training.")
        return False

if __name__ == "__main__":
    success = verify_feature_engineering()
    sys.exit(0 if success else 1)

